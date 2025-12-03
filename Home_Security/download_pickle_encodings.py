import os
import pickle
import time
import threading
import io
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "20"))

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------------------------------------
# Local encodings folder
# --------------------------------------------------
LOCAL_ENCODING_DIR = "encodings"
os.makedirs(LOCAL_ENCODING_DIR, exist_ok=True)

# Global caches
enc_family = []
names_family = []

enc_criminals = []
names_criminals = []

cache_lock = threading.Lock()

# --------------------------------------------------
# Download & SAVE pickle file locally
# --------------------------------------------------
def download_and_save_pickle(remote_path, local_file):
    try:
        res = supabase.storage.from_("sentinal_ai_home_security").download(remote_path)
        file_bytes = res

        # SAVE LOCALLY
        with open(local_file, "wb") as f:
            f.write(file_bytes)

        print(f"⬇ Saved LOCAL: {local_file}")

        return pickle.load(io.BytesIO(file_bytes))

    except Exception as e:
        print(f"❌ Failed: {remote_path} → {e}")
        return None


# --------------------------------------------------
# Load all encodings into RAM
# --------------------------------------------------
def load_all_encodings():
    global enc_family, names_family, enc_criminals, names_criminals

    print("🔄 Fetching & saving encodings...")

    family = download_and_save_pickle(
        "encodings/family_encodings.p",
        os.path.join(LOCAL_ENCODING_DIR, "family_encodings.p")
    )

    criminal = download_and_save_pickle(
        "encodings/criminal_encodings.p",
        os.path.join(LOCAL_ENCODING_DIR, "criminal_encodings.p")
    )

    if family:
        with cache_lock:
            enc_family = family["encodings"]
            names_family = family["names"]

    if criminal:
        with cache_lock:
            enc_criminals = criminal["encodings"]
            names_criminals = criminal["names"]

    print(f"✔ Loaded → Family: {len(enc_family)} | Criminals: {len(enc_criminals)}")


def poller_loop():
    while True:
        load_all_encodings()
        time.sleep(POLL_INTERVAL)


def start_polling_encodings():
    load_all_encodings()
    t = threading.Thread(target=poller_loop, daemon=True)
    t.start()
    print(f"📡 Poller Started (every {POLL_INTERVAL}s)")


def get_family_encodings():
    with cache_lock:
        return enc_family, names_family


def get_criminal_encodings():
    with cache_lock:
        return enc_criminals, names_criminals
    
    
if __name__ == "__main__":
    start_polling_encodings()
    time.sleep(5)  # allow fetch

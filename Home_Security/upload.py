


import os
import pickle
import uuid
import face_recognition
import numpy as np
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# Supabase Credentials
# -------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------
# Dataset Paths
# -------------------------------
FAMILY_DIR = "dataset/family"
CRIMINAL_DIR = "dataset/criminals"

# -------------------------------
# Helper Functions
# -------------------------------
def encode_images_in_folder(folder_path):
    encodings = []
    names = []

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        name = os.path.splitext(file)[0]

        print(f"Encoding: {file}")

        try:
            img = face_recognition.load_image_file(path)
            encode = face_recognition.face_encodings(img)

            if len(encode) == 0:
                print(f"❌ No face found in {file}, skipping.")
                continue

            encodings.append(encode[0])
            names.append(name)

        except Exception as e:
            print(f"Error processing {file}: {e}")

    return encodings, names


def save_pickle_file(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"✔ Pickle created: {filename}")


def upload_pickle_to_supabase(local_file, remote_path):
    with open(local_file, "rb") as f:
        supabase.storage.from_("sentinal_ai_home_security").upload(
            path=remote_path,
            file=f,
            file_options={"content-type": "application/octet-stream"}
        )

    public_url = supabase.storage.from_("sentinal_ai_home_security").get_public_url(remote_path)
    print(f"⬆ Uploaded: {remote_path}")
    print(f"🌐 URL: {public_url}")
    return public_url


# -------------------------------
# MAIN SCRIPT
# -------------------------------
if __name__ == "__main__":
    print("\n==============")
    print("FAMILY ENCODINGS")
    print("==============")

    family_encodings, family_names = encode_images_in_folder(FAMILY_DIR)
    family_data = {"encodings": family_encodings, "names": family_names}
    save_pickle_file("family_encodings.p", family_data)
    upload_pickle_to_supabase("family_encodings.p", "encodings/family_encodings.p")

    print("\n==============")
    print("CRIMINAL ENCODINGS")
    print("==============")

    criminal_encodings, criminal_names = encode_images_in_folder(CRIMINAL_DIR)
    criminal_data = {"encodings": criminal_encodings, "names": criminal_names}
    save_pickle_file("criminal_encodings.p", criminal_data)
    upload_pickle_to_supabase("criminal_encodings.p", "encodings/criminal_encodings.p")

    print("\nAll encodings generated & uploaded successfully! 🎉")

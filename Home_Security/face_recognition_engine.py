import cv2
import numpy as np
import time
from datetime import datetime
import pyttsx3
from supabase import create_client, Client
import os
import pickle
from dotenv import load_dotenv

load_dotenv()

# --------------------------
# Supabase Setup
# --------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------------
# TTS Setup
# --------------------------
engine = pyttsx3.init()

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except:
        pass

# --------------------------
# Load Local Encodings
# --------------------------
with open("encodings/family_encodings.p", "rb") as f:
    fam = pickle.load(f)
    enc_family_local = fam["encodings"]
    names_family_local = fam["names"]

with open("encodings/criminal_encodings.p", "rb") as f:
    crim = pickle.load(f)
    enc_criminal_local = crim["encodings"]
    names_criminal_local = crim["names"]

# --------------------------
# Supabase Logging Functions
# --------------------------
def insert_visitor_log(name=None, recognized_as="unknown", notes=None, image_url=None):
    try:
        supabase.table("visitor_logs").insert({
            "name": name,
            "recognized_as": recognized_as,
            "notes": notes,
            "image_url": image_url
        }).execute()
    except Exception as e:
        print("❌ Failed to insert visitor log:", e)


def insert_alert(alert_type, message, image_url=None):
    try:
        supabase.table("alerts").insert({
            "alert_type": alert_type,
            "message": message,
            "image_url": image_url
        }).execute()
    except Exception as e:
        print("❌ Failed to insert alert:", e)

# --------------------------
# Upload Image to Supabase
# --------------------------
def upload_visitor_image(frame, filename):
    try:
        ret, buffer = cv2.imencode(".jpg", frame)
        file_bytes = buffer.tobytes()

        path = f"visitors/{filename}"

        supabase.storage.from_("sentinal_ai_home_security").upload(
            path=path,
            file=file_bytes,
            file_options={"content-type": "image/jpeg"}
        )

        public_url = supabase.storage.from_("sentinal_ai_home_security").get_public_url(path)
        print(f"📸 Uploaded visitor image: {public_url}")

        return public_url

    except Exception as e:
        print("❌ Failed to upload visitor image:", e)
        return None

# --------------------------
# Matching Helper
# --------------------------
def match_face(encodeFace, family_enc, names_family, criminal_enc, names_criminal):
    # Family
    if len(family_enc) > 0:
        dists_f = np.linalg.norm(family_enc - encodeFace, axis=1)
        min_dist_f = np.min(dists_f)
        if min_dist_f < 0.50:
            idx = np.argmin(dists_f)
            return "family", names_family[idx], min_dist_f
    
    # Criminal
    if len(criminal_enc) > 0:
        dists_c = np.linalg.norm(criminal_enc - encodeFace, axis=1)
        min_dist_c = np.min(dists_c)
        if min_dist_c < 0.50:
            idx = np.argmin(dists_c)
            return "criminal", names_criminal[idx], min_dist_c

    return "unknown", "Unknown", None


# --------------------------
# Main Camera Loop
# --------------------------
def main():
    print("📡 Starting Sentinal AI - Secure Homes Face Recognition Engine…")

    cap = cv2.VideoCapture(0)
    last_announced = {}
    last_captured = {}
    CAPTURE_INTERVAL = 60  # seconds

    while True:
        family_enc = enc_family_local
        names_family = names_family_local
        criminal_enc = enc_criminal_local
        names_criminal = names_criminal_local

        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        import face_recognition
        faces = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, faces)

        for encodeFace, (top, right, bottom, left) in zip(encodings, faces):
            
            match_type, name, dist = match_face(
                encodeFace,
                np.array(family_enc, dtype='float'),
                names_family,
                np.array(criminal_enc, dtype='float'),
                names_criminal
            )

            # ----------------------------------------------------
            # 📸 Photo Capture Logic (Unknown + Criminal)
            # ----------------------------------------------------
            current_time = time.time()
            should_capture = False

            if match_type == "criminal":
                should_capture = True
                visitor_key = name
            elif match_type == "unknown":
                should_capture = True
                visitor_key = f"unknown_{top}"

            if should_capture:
                if visitor_key not in last_captured or (current_time - last_captured[visitor_key]) > CAPTURE_INTERVAL:
                    filename = f"{visitor_key}_{int(current_time)}.jpg"
                    image_url = upload_visitor_image(frame, filename)

                    insert_visitor_log(
                        name=name if match_type == "criminal" else None,
                        recognized_as=match_type,
                        notes=f"Photo captured dist={dist}",
                        image_url=image_url
                    )

                    last_captured[visitor_key] = current_time

            # ----------------------------------------------------
            # Scale bounding box
            # ----------------------------------------------------
            top *= 4; right *= 4; bottom *= 4; left *= 4

            # ----------------------------------------------------
            # Voice + Logging
            # ----------------------------------------------------
            if match_type == "family":
                color = (0, 255, 0)
                label = f"{name} (Family)"

                if name not in last_announced or time.time() - last_announced[name] > 10:
                    speak(f"Welcome home {name}")
                    last_announced[name] = time.time()

                insert_visitor_log(name=name, recognized_as="family", notes=f"dist={dist}")

            elif match_type == "criminal":
                color = (0, 0, 255)
                label = f"{name} (CRIMINAL)"

                if name not in last_announced or time.time() - last_announced[name] > 10:
                    speak("Known criminal detected outside the home. Should I alert the authorities?")
                    last_announced[name] = time.time()

                    insert_alert("criminal_detected", f"{name} detected at {datetime.now()}",image_url=image_url)
                    insert_visitor_log(name=name, recognized_as="criminal", notes=f"dist={dist}")

            else:
                color = (0, 165, 255)
                label = "Unknown"
                # Unknown already logged above

            # Draw box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("SentinalAI - Secure Homes", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

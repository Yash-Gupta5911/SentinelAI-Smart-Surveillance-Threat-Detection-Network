# ------------------------------------------------------
# Real-time Object Detection + Tracking + MySQL Logging
# ------------------------------------------------------
from ultralytics import YOLO
import cv2, time, pymysql, os
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- CONFIG ---
VIDEO_SOURCE = 0  # or "C:/Users/LENOVO/Desktop/s2/vehicles.mp4" or RTSP URL
YOLO_MODEL = "yolov8n.pt"
SAVE_SNAPS = "./snapshots"
DB_CONFIG = dict(host="localhost", user="root", password="2004", db="sentinel", charset='utf8mb4')

os.makedirs(SAVE_SNAPS, exist_ok=True)

# --- DB Helper Functions ---
def connect_db():
    try:
        con = pymysql.connect(**DB_CONFIG)
        return con
    except Exception as e:
        print(f"❌ [DB] Connection Error: {e}")
        return None

def insert_detection(device_id, ts, obj_type, conf, bbox, track_id, snap_path):
    con = connect_db()
    if not con:
        return False
    try:
        cur = con.cursor()
        sql = """INSERT INTO detections (device_id, timestamp, object_type, confidence,
                 bbox_x, bbox_y, bbox_w, bbox_h, track_id, snapshot_path)
                 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        x, y, w, h = bbox
        cur.execute(sql, (device_id, ts, obj_type, conf, x, y, w, h, track_id, snap_path))
        con.commit()
        cur.close()
        con.close()
        print(f"🟢 [DB] Record inserted for {obj_type} (Track {track_id}) at {ts}")
        return True
    except Exception as e:
        print(f"❌ [DB] Insert Error: {e}")
        return False

# --- Model + Tracker Init ---
print("🚀 Loading YOLO model...")
model = YOLO(YOLO_MODEL)
tracker = DeepSort(max_age=30)
print("✅ YOLO model loaded successfully!")

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("❌ Error: Could not open video source.")
    exit()

device_id = 1
print("🎥 Starting detection... Press 'Q' to stop.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Stream ended or cannot read frame.")
        break

    results = model(frame)[0]
    detections_for_tracker = []

    for r in results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())  # ✅ Fixed
        conf = float(r.conf[0])
        cls = int(r.cls[0])
        label = model.names[cls]
        w = x2 - x1
        h = y2 - y1
        detections_for_tracker.append(([x1, y1, x2, y2], conf, label))
        print(f"🔍 Detected: {label} | Conf: {conf:.2f}")

    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    for t in tracks:
        if not t.is_confirmed():
            continue
        tid = t.track_id
        ltrb = t.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        w = x2 - x1
        h = y2 - y1
        label = t.det_class or "object"
        conf = t.det_conf or 0.0

        # Save snapshot
        snap_name = f"{SAVE_SNAPS}/{device_id}_{tid}_{int(time.time())}.jpg"
        cv2.imwrite(snap_name, frame[y1:y2, x1:x2])

        # Insert into DB
        success = insert_detection(device_id, ts, label, conf, (x1, y1, w, h), tid, snap_name)

        # Console log summary
        status = "✅ Saved" if success else "⚠️ Not Saved"
        print(f"📦 [{status}] {label} (Track {tid}) | Conf: {conf:.2f} | Snapshot: {os.path.basename(snap_name)}")

        # Draw bounding boxes
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}-{tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("SentinelAI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n🛑 Detection stopped. Exiting cleanly.")

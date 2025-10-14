import os, shutil

snapshots_folder = "snapshots"
if os.path.exists(snapshots_folder):
    for file in os.listdir(snapshots_folder):
        file_path = os.path.join(snapshots_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print("🧹 All snapshot files deleted successfully.")
else:
    print("⚠️ Folder not found.")
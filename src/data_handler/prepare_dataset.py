import os
import subprocess
from config import DATA_DIR
import json
import cv2
from pathlib import Path
import shutil


# Copied from framecut.py (See Anti-UAV-RGBT)
def video2image(video_path, type:str="jpg"):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success = True
    print('read video in', video_path)

    if type == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif type == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 1]

    while success:
        success, frame = cap.read()
        if not success:
            break
        
        #print("Frame %d" %frame_count)
        cv2.imwrite(video_path.parent / "{}.{}".format(str(frame_count).zfill(6), type), frame, params)
        frame_count = frame_count + 1
    cap.release()

# Copied from framecut.py (See Anti-UAV-RGBT)
def get_all_dir(root_dir):
    video_pathes = []
    for dir in os.listdir(root_dir):
        dir = os.path.join(root_dir, dir)
        if os.path.isdir(dir):
            video_pathes.extend(get_all_dir(dir))
        if dir[-4:] == '.mp4':
            video_pathes.append(Path(dir))
    return video_pathes


def prepare_anti_uav_300():
    ANTI_UAV_300_DIR_NAME = "Anti-UAV-RGBT"
    ANTI_UAV_300_DIR_PATH = DATA_DIR / ANTI_UAV_300_DIR_NAME
    ZIP_NAME = "Anti-UAV-RGBT.zip"
    archive_path = os.path.join(DATA_DIR, ZIP_NAME)

    if os.path.exists(ANTI_UAV_300_DIR_PATH):
        shutil.rmtree(ANTI_UAV_300_DIR_PATH)

    print(f"=== 1. Prüfe das Archiv in {DATA_DIR} ===")
    if not os.path.exists(archive_path):
        print(f"\n[!] FEHLER: Die Datei '{ZIP_NAME}' wurde nicht in '{DATA_DIR}' gefunden.")
        print(f"Lade das Dataset einmalig manuell im Browser herunter und lege es als '{ZIP_NAME}' in den Ordner '{DATA_DIR}/'.")
        exit(1)

    print("=== 2. Entpacke Archiv mit 'unzip' ===")
    # Ruft den echten Bash-Befehl 'unzip' auf
    subprocess.run(["unzip", "-q", str(archive_path), "-d", str(ANTI_UAV_300_DIR_PATH)], check=True)
    print("Entpacken erfolgreich abgeschlossen.")

    print("=== 3. Bereinige train, test und val (lösche visible.mp4 & visible.json) ===")
    # Nutzt den echten Bash-Befehl 'find' mit -delete
    for split in ["train", "test", "val"]:
        split_dir = os.path.join(ANTI_UAV_300_DIR_PATH, split)
        if os.path.exists(split_dir):
            print(f"Bereinige Verzeichnis: {split_dir}")
            find_cmd = f"find '{split_dir}' -type f \\( -name 'visible.mp4' -o -name 'visible.json' \\) -delete"
            subprocess.run(find_cmd, shell=True, check=True)
        else:
            print(f"Hinweis: Ordner '{split_dir}' existiert nicht (wird übersprungen).")

    print("=== 4. Split video into frames ===")
    video_pathes = get_all_dir(ANTI_UAV_300_DIR_PATH)
    for vid, video_path in enumerate(video_pathes):
        video2image(video_path)
        os.unlink(video_path)

    print("=== 5. Update formatting of some json-files ===")
    for file_name in ["test.json", "train.json", "val.json"]:
        file_path = ANTI_UAV_300_DIR_PATH / "label_new" / file_name

        if not file_path.exists():
            print(f"File not found. Not formatted. File: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f) 

        # 2. Strukturiert und formatiert in eine neue Datei speichern
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                data, 
                f, 
                indent=4,           # Sorgt für die saubere Einrückung (Strukturierung)
                ensure_ascii=False  # Verhindert, dass Umlaute als Code (z.B. \u00fc) angezeigt werden
            )

    print(f"\n=== Fertig! Datensatz in {DATA_DIR} ist bereit und bereinigt. ===")

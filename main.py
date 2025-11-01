import cv2
import time
import os
from pathlib import Path

# try to import load_dotenv, fallback to tiny loader if missing
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(dotenv_path=None):
        p = Path(dotenv_path or ".env")
        if not p.exists():
            return
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('\'"'))

from src.camera import Camera
from src.common import clear_image_folder, clear_old_images
from src.detection import ArucoDetector

# load env
env_path = Path(".env")
load_dotenv(dotenv_path=env_path)

# read env with sensible defaults
IMAGE_DIR = os.getenv("IMAGE_DIR") or "captured_images"
IMAGE_EXT = os.getenv("IMAGE_EXT") or "jpg"
CAMERA_ID_RAW = os.getenv("CAMERA_ID", "0")
MAX_IMAGE = int(os.getenv("MAX_IMAGE") or 50)

# optional requested resolution from env
_cam_w = os.getenv("CAMERA_WIDTH")
_cam_h = os.getenv("CAMERA_HEIGHT")
CAMERA_WIDTH = int(_cam_w) if _cam_w and _cam_w.isdigit() else None
CAMERA_HEIGHT = int(_cam_h) if _cam_h and _cam_h.isdigit() else None

# parse camera id (try int, else keep as string path)
try:
    CAMERA_ID = int(CAMERA_ID_RAW)
except Exception:
    CAMERA_ID = CAMERA_ID_RAW

print(f"DEBUG: IMAGE_DIR={IMAGE_DIR}, IMAGE_EXT={IMAGE_EXT}, CAMERA_ID={CAMERA_ID}, MAX_IMAGE={MAX_IMAGE}, WIDTH={CAMERA_WIDTH}, HEIGHT={CAMERA_HEIGHT}")

def update_in_real_time():
    # ensure folder exists
    os.makedirs(IMAGE_DIR, exist_ok=True)
    clear_image_folder(IMAGE_DIR)

    # init camera
    camera = Camera(CAMERA_ID, IMAGE_DIR, IMAGE_EXT, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
    detector = ArucoDetector(camera)

    try:
        if not camera.is_opened():
            print("Error: camera not opened")
            return

        info = camera.get_camera_info() if hasattr(camera, "get_camera_info") else {}
        print(f"Camera info: {info}")

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        last_capture = 0.0

        while True:
            now = time.time()
            if now - last_capture >= 0.2:
                clear_old_images(IMAGE_DIR, MAX_IMAGE)
                file_path = camera.take_picture()
                if file_path:
                    image_id = os.path.splitext(os.path.basename(file_path))[0]
                    detector.analyze_image(image_id)
                last_capture = now

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        clear_image_folder(IMAGE_DIR)

if __name__ == "__main__":
    update_in_real_time()
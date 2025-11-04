import cv2
import time
import os
from pathlib import Path
import numpy as np

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

def parse_float_array_env(name, default=None, sep=None, dtype=np.float64, shape=None):
    """
    Parse a float array environment variable with a default fallback.
    """
    v = os.getenv(name)
    if v is None:
        if default is None:
            return None
        arr = np.array(default, dtype=dtype)
        return arr.reshape(shape) if shape is not None else arr

    try:
        parts = v.split(sep) if sep is not None else v.split()
        nums = [float(x) for x in parts if x != ""]
        arr = np.array(nums, dtype=dtype)
        if shape is not None:
            arr = arr.reshape(shape)
        return arr
    except Exception:
        if default is None:
            raise
        arr = np.array(default, dtype=dtype)
        return arr.reshape(shape) if shape is not None else arr

def parse_int_env(name, default=None):
    """
    Parse an integer environment variable with a default fallback.
    """
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default

# read env with sensible defaults
IMAGE_DIR = os.getenv("IMAGE_DIR") or "captured_images"
IMAGE_EXT = os.getenv("IMAGE_EXT") or "jpg"
CAMERA_ID_RAW = os.getenv("CAMERA_ID", "0")
MAX_IMAGE = parse_int_env("MAX_IMAGE", 50)

# optional requested resolution from env
CAMERA_WIDTH = parse_int_env("CAMERA_WIDTH", None)
CAMERA_HEIGHT = parse_int_env("CAMERA_HEIGHT", None)
DIST_COEFFS = parse_float_array_env("DIST_COEFFS", default=None, shape=(5,))
CAMERA_MATRIX = parse_float_array_env("CAMERA_MATRIX", default=None, shape=(3, 3))

CHESSBOARD_ROWS = parse_int_env("CHESSBOARD_ROWS", 6)
CHESSBOARD_COLS = parse_int_env("CHESSBOARD_COLS", 9)
SQUARE_SIZE_CM = float(os.getenv("SQUARE_SIZE_CM") or "1.0")
NUM_CALIB_IMAGES = parse_int_env("NUM_CALIB_IMAGES", 20)

# parse camera id (try int, else keep as string path)
try:
    CAMERA_ID = int(CAMERA_ID_RAW)
except Exception:
    CAMERA_ID = CAMERA_ID_RAW

MARKER_SIZE_CM = float(os.getenv("MARKER_SIZE_CM") or "2.4")
ASSUMED_HFOV_DEG = float(os.getenv("ASSUMED_HFOV_DEG") or "90.0")

print(f"DEBUG: IMAGE_DIR={IMAGE_DIR}, IMAGE_EXT={IMAGE_EXT}, CAMERA_ID={CAMERA_ID}, MAX_IMAGE={MAX_IMAGE}, WIDTH={CAMERA_WIDTH}, HEIGHT={CAMERA_HEIGHT}, MARKER_SIZE_CM={MARKER_SIZE_CM}, HFOV={ASSUMED_HFOV_DEG}")


def calibrate_camera():
    camera = Camera(CAMERA_ID, IMAGE_DIR, IMAGE_EXT, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
    camera_matrix, dist_coeffs, rvecs, tvecs = camera.get_dist_coeffs(
        chessboard_size=(CHESSBOARD_COLS, CHESSBOARD_ROWS),
        square_size=SQUARE_SIZE_CM,
        num_images=NUM_CALIB_IMAGES
    )

    if camera_matrix is not None:
        print("Calibration completed successfully")
    else:
        print("Calibration failed")


def update_in_real_time():
    """
    Capture images from camera in real-time, analyze them for Aruco markers,
    and display the results. Cleans up images after use.
    """
    # ensure folder exists
    os.makedirs(IMAGE_DIR, exist_ok=True)
    clear_image_folder(IMAGE_DIR)

    # init camera
    camera = Camera(CAMERA_ID, IMAGE_DIR, IMAGE_EXT, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
    detector = ArucoDetector(camera, MARKER_SIZE_CM, camera_matrix=CAMERA_MATRIX, dist_coeffs=DIST_COEFFS, assumed_hfov_deg=ASSUMED_HFOV_DEG)

    try:
        if not camera.is_opened():
            print("Error: camera not opened")
            return

        info = camera.get_camera_info()
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
    #calibrate_camera()
import cv2
import os
from pathlib import Path
import numpy as np

try:
    from dotenv import load_dotenv
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
from src.detection import ArucoDetector

load_dotenv(dotenv_path=Path(".env"))


def parse_float_array(name, default=None, shape=None):
    v = os.getenv(name)
    if not v:
        return np.array(default, dtype=np.float64).reshape(shape) if default else None
    try:
        arr = np.array([float(x) for x in v.split()], dtype=np.float64)
        return arr.reshape(shape) if shape else arr
    except Exception:
        return np.array(default, dtype=np.float64).reshape(shape) if default else None


def parse_int(name, default):
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


CAMERA_ID = parse_int("CAMERA_ID", 0)
CAMERA_WIDTH = parse_int("CAMERA_WIDTH", 1920)
CAMERA_HEIGHT = parse_int("CAMERA_HEIGHT", 1080)
MARKER_SIZE_CM = float(os.getenv("MARKER_SIZE_CM", "2.4"))
ASSUMED_HFOV_DEG = float(os.getenv("ASSUMED_HFOV_DEG", "90.0"))
DIST_COEFFS = parse_float_array("DIST_COEFFS", shape=(5,))
CAMERA_MATRIX = parse_float_array("CAMERA_MATRIX", shape=(3, 3))
CHESSBOARD_ROWS = parse_int("CHESSBOARD_ROWS", 6)
CHESSBOARD_COLS = parse_int("CHESSBOARD_COLS", 9)
SQUARE_SIZE_CM = float(os.getenv("SQUARE_SIZE_CM", "1.0"))
NUM_CALIB_IMAGES = parse_int("NUM_CALIB_IMAGES", 20)

def calibrate_camera():
    camera = Camera(CAMERA_ID, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
    if not camera.is_opened():
        print("Erreur: impossible d'ouvrir la caméra")
        return

    camera_matrix, dist_coeffs, _, _ = camera.calibrate(
        chessboard_size=(CHESSBOARD_COLS, CHESSBOARD_ROWS),
        square_size=SQUARE_SIZE_CM,
        num_images=NUM_CALIB_IMAGES
    )

    camera.release()

    if camera_matrix is not None:
        print("Calibration réussie")
        print(f"CAMERA_MATRIX=\"{' '.join(map(str, camera_matrix.ravel()))}\"")
        print(f"DIST_COEFFS=\"{' '.join(map(str, dist_coeffs.ravel()))}\"")
    else:
        print("Échec de la calibration")


def update_in_real_time():
    camera = Camera(CAMERA_ID, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
    if not camera.is_opened():
        print("Erreur: impossible d'ouvrir la caméra")
        return

    detector = ArucoDetector(
        camera,
        MARKER_SIZE_CM,
        camera_matrix=CAMERA_MATRIX,
        dist_coeffs=DIST_COEFFS,
        assumed_hfov_deg=ASSUMED_HFOV_DEG
    )

    print(f"Caméra: {camera.get_camera_info()}")
    cv2.namedWindow("ArUco Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Arena", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                continue

            annotated_frame = detector.analyze_frame(frame, show_arena=True, arena_window_name="Arena")
            cv2.imshow("ArUco Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    finally:
        # nettoyer les ressources matplotlib
        if detector._arena_fig is not None:
            import matplotlib.pyplot as plt
            plt.close(detector._arena_fig)
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    update_in_real_time()
    #calibrate_camera()

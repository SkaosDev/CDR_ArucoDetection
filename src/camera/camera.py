import cv2
import os
import time
import numpy as np
from typing import Optional, Tuple, List

class Camera:
    def __init__(
        self,
        camera_id,
        image_dir,
        file_ext,
        width: Optional[int] = None,
        height: Optional[int] = None,
        backends: Optional[List[int]] = None,
        warmup_frames: int = 5,
    ):
        self.camera_id = camera_id
        self.image_dir = image_dir
        self.file_ext = file_ext
        self.warmup_frames = warmup_frames
        self.cam = None
        self._backend_used = None

        # simplify backend list construction
        if backends is not None:
            default_backends = backends
        else:
            default_backends = []
            for attr in ("CAP_DSHOW", "CAP_MSMF"):
                if hasattr(cv2, attr):
                    default_backends.append(getattr(cv2, attr))
            default_backends.append(cv2.CAP_ANY)

        self._open_with_backends(self.camera_id, default_backends)

        if not self.is_opened():
            print("Warning: camera not opened during init")
            return

        # try requested resolution or adapt from detected/default
        cur_w, cur_h = self.get_resolution()
        if width is not None or height is not None:
            w = width if width is not None else cur_w or 1280
            h = height if height is not None else cur_h or 720
            actual = self.adapt_resolution(w, h)
            print(f"Requested resolution: {w}x{h} | actual: {actual[0]}x{actual[1]}")
        else:
            if cur_w and cur_h:
                actual = self.adapt_resolution(cur_w, cur_h)
                print(f"Detected resolution: {cur_w}x{cur_h} | actual: {actual[0]}x{actual[1]}")
            else:
                actual = self.adapt_resolution(1280, 720)
                print(f"No detected resolution, tried default | actual: {actual[0]}x{actual[1]}")

    def _open_with_backends(self, camera_id, backends: List[int]):
        """
        Try to open camera with multiple backends.
        """
        # try multiple backends to get best behavior on Windows
        for b in backends:
            try:
                if b is None or b == cv2.CAP_ANY:
                    cap = cv2.VideoCapture(camera_id)
                else:
                    cap = cv2.VideoCapture(camera_id, b)
                if cap is not None and cap.isOpened():
                    self.cam = cap
                    self._backend_used = b
                    return
                # ensure release if opened failed
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
            except Exception:
                pass
        # final attempt with plain constructor
        try:
            cap = cv2.VideoCapture(camera_id)
            if cap is not None and cap.isOpened():
                self.cam = cap
                self._backend_used = None
        except Exception:
            self.cam = None

    def get_file_info(self):
        """
        Get image storage info (directory, extension).
        """
        return self.image_dir, self.file_ext

    def is_opened(self) -> bool:
        """
        Check if camera is opened.
        """
        return self.cam is not None and self.cam.isOpened()

    def get_resolution(self) -> Tuple[int, int]:
        """
        Get current camera resolution (width, height).
        """
        if not self.is_opened():
            return (0, 0)
        w = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return (w, h)

    def get_camera_info(self):
        """
        Get camera properties as a dictionary.
        """
        if not self.is_opened():
            return {}
        w, h = self.get_resolution()
        fps = float(self.cam.get(cv2.CAP_PROP_FPS) or 0.0)
        try:
            fourcc_int = int(self.cam.get(cv2.CAP_PROP_FOURCC) or 0)
            fourcc = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]).strip() or None
        except Exception:
            fourcc = None
        return {"width": w, "height": h, "fps": fps, "fourcc": fourcc, "backend": self._backend_used}

    def _warmup_and_read(self, frames: int = None):
        """
        Read frames to warmup camera.
        """
        if frames is None:
            frames = self.warmup_frames
        if not self.cam:
            return
        for _ in range(frames):
            try:
                self.cam.read()
            except Exception:
                pass

    def set_resolution(self, width: int, height: int) -> Tuple[int, int]:
        """
        Try to set resolution; perform a small warmup and return actual resolution.
        """
        if not self.is_opened():
            return (0, 0)
        # apply settings
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        # small pause so driver applies changes
        time.sleep(0.05)
        self._warmup_and_read()
        # read once more to update properties
        return self.get_resolution()

    def adapt_resolution(self, target_width: int, target_height: int, fallbacks: Optional[List[Tuple[int,int]]] = None) -> Tuple[int,int]:
        """
        Try to set requested resolution; if not possible, try fallbacks.
        """
        if not self.is_opened():
            return (0, 0)

        # try requested first
        actual = self.set_resolution(target_width, target_height)
        if actual == (target_width, target_height):
            return actual

        if not fallbacks:
            fallbacks = [(1920,1080), (1280,720), (1024,768), (800,600), (640,480), (320,240)]

        # try fallbacks
        for w, h in fallbacks:
            actual = self.set_resolution(w, h)
            if actual == (w, h):
                return actual

        # nothing matched; return reported resolution
        return self.get_resolution()

    def take_picture(self):
        """
        Capture an image and save it to disk; return file path or None on error.
        """
        if not self.is_opened():
            print("Error: camera not opened - Cannot take picture")
            return None
        ret, frame = self.cam.read()
        if not ret or frame is None:
            print("Error: failed to take picture")
            return None
        os.makedirs(self.image_dir, exist_ok=True)
        file_id = str(time.time()).replace(".", "_")
        file_path = os.path.join(self.image_dir, f"{file_id}.{self.file_ext}")
        cv2.imwrite(file_path, frame)
        return file_path

    def get_dist_coeffs(self, chessboard_size: Tuple[int, int] = (9, 6), square_size: float = 1.0,
                        num_images: int = 20):
        """
        Calibrate camera using chessboard images to calculate distortion coefficients.

        Args:
            chessboard_size: Number of internal corners (columns, rows)
            square_size: Size of a square in your defined unit (e.g., millimeters)
            num_images: Number of images to capture for calibration

        Returns:
            tuple: (camera_matrix, dist_coeffs, rvecs, tvecs) or None on error
        """
        if not self.is_opened():
            print("Error: camera not opened")
            return None, None, None, None

        # Prepare object points
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size

        objpoints = []
        imgpoints = []

        print(f"Capturing {num_images} images for calibration. Press 'c' to capture, 'q' to finish.")

        captured = 0
        frame_shape = None

        while captured < num_images:
            ret, frame = self.cam.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_shape = gray.shape[::-1]

            ret_chess, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            display_frame = frame.copy()
            if ret_chess:
                cv2.drawChessboardCorners(display_frame, chessboard_size, corners, ret_chess)
                cv2.putText(display_frame, f"Pattern found! Press 'c' to capture ({captured}/{num_images})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No pattern detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Calibration', display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and ret_chess:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                objpoints.append(objp)
                imgpoints.append(corners2)
                captured += 1
                print(f"Image {captured}/{num_images} captured")
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()

        if captured < 3:
            print("Error: need at least 3 images for calibration")
            return None, None, None, None

        print("Calibrating camera...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, frame_shape, None, None
        )

        if ret:
            print("Calibration successful!")
            print(f"Camera matrix:\n{camera_matrix}")
            print(f"Distortion coefficients: {dist_coeffs.ravel()}")
            return camera_matrix, dist_coeffs, rvecs, tvecs
        else:
            print("Calibration failed")
            return None, None, None, None

    def release(self):
        """
        Release camera resources.
        """
        if self.cam:
            try:
                self.cam.release()
            except Exception:
                pass
            self.cam = None

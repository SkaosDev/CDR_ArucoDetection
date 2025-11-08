import cv2
import numpy as np
import math


class ArucoDetector:
    def __init__(self, cam, marker_size_cm: float, camera_matrix: np.ndarray = None,
                 dist_coeffs: np.ndarray = None, assumed_hfov_deg: float = 60.0):
        self.cam = cam
        self.marker_size_m = float(marker_size_cm) / 100.0
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.assumed_hfov_deg = float(assumed_hfov_deg)

        # Initialisation unique du dictionnaire ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Estimation de la matrice caméra si non fournie
        if self.camera_matrix is None:
            w, h = self.cam.get_resolution()
            if w and h:
                fx = fy = (w / 2.0) / math.tan(math.radians(self.assumed_hfov_deg) / 2.0)
                self.camera_matrix = np.array([[fx, 0.0, w / 2.0],
                                               [0.0, fy, h / 2.0],
                                               [0.0, 0.0, 1.0]], dtype=np.float64)
                self.dist_coeffs = np.zeros((5, 1), dtype=np.float64) if self.dist_coeffs is None else self.dist_coeffs

        # Objets 3D du marqueur (réutilisable)
        s = self.marker_size_m
        self.objp = np.array([[-s / 2, s / 2, 0.0], [s / 2, s / 2, 0.0],
                              [s / 2, -s / 2, 0.0], [-s / 2, -s / 2, 0.0]], dtype=np.float64)

    def analyze_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return frame

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        if self.camera_matrix is not None:
            for i in range(len(ids)):
                imgp = corners[i][0].astype(np.float64)
                _, rvec, tvec = cv2.solvePnP(self.objp, imgp, self.camera_matrix,
                                             self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)

                dist_cm = float(np.linalg.norm(tvec)) * 100.0

                try:
                    cv2.aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs,
                                       rvec, tvec, self.marker_size_m * 0.5)
                except Exception:
                    pass

                center = np.mean(corners[i][0], axis=0).astype(int)
                cv2.putText(frame, f"{dist_cm:.1f} cm", (center[0] - 50, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        return frame

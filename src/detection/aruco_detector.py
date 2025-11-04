import cv2
import numpy as np
import math

class ArucoDetector:
    def __init__(self, cam, marker_size_cm: float, camera_matrix: np.ndarray = None, dist_coeffs: np.ndarray = None, assumed_hfov_deg: float = 60.0):
        self.cam = cam
        self.image_dir, self.file_ext = self.cam.get_file_info()
        self.marker_size_cm = float(marker_size_cm)
        self.marker_size_m = self.marker_size_cm / 100.0
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.assumed_hfov_deg = float(assumed_hfov_deg)

        # if no camera matrix provided, try to estimate focal from image width and assumed HFOV
        if self.camera_matrix is None:
            w, h = self.cam.get_resolution()
            if w and h:
                fx = fy = (w / 2.0) / math.tan(math.radians(self.assumed_hfov_deg) / 2.0)
                cx = w / 2.0
                cy = h / 2.0
                self.camera_matrix = np.array([[fx, 0.0, cx],
                                               [0.0, fy, cy],
                                               [0.0, 0.0, 1.0]], dtype=float)
                self.dist_coeffs = np.zeros((5, 1), dtype=float) if self.dist_coeffs is None else self.dist_coeffs
            else:
                self.camera_matrix = None
                self.dist_coeffs = None

    @staticmethod
    def estimate_pose_single_markers_fallback(corners, marker_size_m, camera_matrix, dist_coeffs):
        """
        Replace cv2.aruco.estimatePoseSingleMarkers if missing.
        - corners : list/array of shape (N,1,4,2) return by detectMarkers
        - marker_size_m : real size of markers in meters
        - camera_matrix, dist_coeffs : camera calibration
        Return (rvecs, tvecs) with shapes (N,1,3) and (N,1,3)
        """
        s = float(marker_size_m)
        # Points order: top-left, top-right, bottom-right, bottom-left
        objp = np.array([[-s / 2, s / 2, 0.0],
                         [s / 2, s / 2, 0.0],
                         [s / 2, -s / 2, 0.0],
                         [-s / 2, -s / 2, 0.0]], dtype=np.float64)

        rvecs = []
        tvecs = []

        # choose PnP method with square marker support if available
        flags = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", None)
        if flags is None:
            flags = getattr(cv2, "SOLVEPNP_IPPE", cv2.SOLVEPNP_ITERATIVE)

        for c in corners:
            imgp = np.asarray(c[0], dtype=np.float64)
            try:
                ret, rvec, tvec = cv2.solvePnP(objp, imgp, camera_matrix, dist_coeffs, flags=flags)
                if not ret:
                    ret, rvec, tvec = cv2.solvePnP(objp, imgp, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            except Exception:
                # fallback to default method
                ret, rvec, tvec = cv2.solvePnP(objp, imgp, camera_matrix, dist_coeffs)

            rvecs.append(rvec.reshape(1, 3))
            tvecs.append(tvec.reshape(1, 3))

        rvecs_np = np.array(rvecs, dtype=np.float64)
        tvecs_np = np.array(tvecs, dtype=np.float64)
        return rvecs_np, tvecs_np

    def analyze_image(self, image_id):
        """
        Analyze a single image for Aruco markers, estimate their distance,
        and display the results.
        """
        path = f"{self.image_dir}/{image_id}.{self.file_ext}"
        image = cv2.imread(path)
        if image is None:
            print(f"Error : unable to read image at {path}")
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()

        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            cv2.imshow("Image", image)
            return

        cv2.aruco.drawDetectedMarkers(image, corners, ids)

        # if camera_matrix available, use estimate_pose_single_markers_fallback
        use_pose = self.camera_matrix is not None and self.dist_coeffs is not None
        if use_pose:
            # markerSize in meters
            rvecs, tvecs = self.estimate_pose_single_markers_fallback(corners=corners, marker_size_m=self.marker_size_m, camera_matrix=self.camera_matrix, dist_coeffs=self.dist_coeffs)
            for i in range(len(ids)):
                tvec = tvecs[i][0]
                dist_m = float(np.linalg.norm(tvec))
                dist_cm = dist_m * 100.0
                # draw axis if possible
                try:
                    cv2.aruco.drawAxis(image, self.camera_matrix, self.dist_coeffs, rvecs[i][0], tvecs[i][0], self.marker_size_m * 0.5)
                except Exception:
                    pass
                # put distance text near marker center
                center = np.mean(corners[i][0], axis=0).astype(int)
                cv2.putText(image, f"{dist_cm:.1f} cm", (center[0] - 50, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        else:
            # fallback: simple pinhole relation using apparent width in pixels and estimated focal
            # distance = (real_width * focal) / pixel_width
            h_img, w_img = image.shape[:2]
            focal = (w_img / 2.0) / math.tan(math.radians(self.assumed_hfov_deg) / 2.0)
            for i in range(len(ids)):
                c = corners[i][0]
                # measure marker width in pixels as average of top and bottom edges
                top = np.linalg.norm(c[0] - c[1])
                bottom = np.linalg.norm(c[2] - c[3])
                pixel_width = float((top + bottom) / 2.0)
                if pixel_width <= 0:
                    continue
                dist_m = (self.marker_size_m * focal) / pixel_width
                dist_cm = dist_m * 100.0
                center = np.mean(c, axis=0).astype(int)
                cv2.putText(image, f"{dist_cm:.1f} cm (est)", (center[0] - 60, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Image", image)

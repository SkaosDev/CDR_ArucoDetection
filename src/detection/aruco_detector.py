import cv2

class ArucoDetector:
    def __init__(self, cam):
        self.cam = cam
        self.image_dir, self.file_ext = self.cam.get_file_info()

    def analyze_image(self, image_id):
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

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)

        cv2.imshow("Image", image)
import uuid
import cv2
import numpy as np
import os
import time

IMAGE_DIR = "captured_images"
IMAGE_EXT = "jpg"
CAMERA_ID = 0

class Camera:
    def __init__(self, camera_id):
        self.cam = cv2.VideoCapture(camera_id)
        self.last_image_id = None

    def get_last_image_id(self):
        return self.last_image_id

    def take_picture(self):
        if not self.cam.isOpened():
            print("Error: camera not opened")
            exit()

        ret, frame = self.cam.read()

        if ret:
            os.makedirs(IMAGE_DIR, exist_ok=True)
            file_id = str(uuid.uuid4())
            file_path = os.path.join(IMAGE_DIR, file_id + ".jpg")
            self.last_image_id = file_id

            cv2.imwrite(file_path, frame)
            print(f"Image saved at {file_path}")
        else:
            print("Error: failed to take picture")

    def release(self):
        self.cam.release()

class ArucoDetector:
    def __init__(self):
        self.cam = Camera(CAMERA_ID)

def clear_image_folder(folder_name):
    if not os.path.exists(folder_name):
        print(f"Folder '{folder_name}' don't exist.")
        return

    for filename in os.listdir(folder_name):
        file_path = os.path.join(folder_name, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error: unable to remove image at {file_path}: {e}")

    print(f"All file in '{folder_name}' have been deleted.")

def analyze_image(image_id):
    path = f"{IMAGE_DIR}/{image_id}.{IMAGE_EXT}"
    image = cv2.imread(path)
    if image is None:
        print(f"Error : unable to read image at {path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected  = detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

    cv2.imshow("Image", image)

def update_every_second():
    clear_image_folder(IMAGE_DIR)
    cam = Camera(CAMERA_ID)
    try:
        if not cam.cam.isOpened():
            print("Error: camera not opened")
            return

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

        last_capture = 0.0
        while True:
            now = time.time()
            if now - last_capture >= 0.2:
                cam.take_picture()
                img_id = cam.get_last_image_id()
                if img_id:
                    analyze_image(img_id)
                last_capture = now

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or 'q' to quit
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()
        clear_image_folder(IMAGE_DIR)

update_every_second()
import dlib
import time


class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def predict_landmarks(self, img, dets):
        for detection in dets:
            face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
            yield face_landmarks

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)

        start = time.time()
        dets = self.detector(img, 1)
        diff = time.time() - start

        print(f"Took {diff:.2f} seconds to detect {len(dets)} faces")

        landmarks = list(self.predict_landmarks(img, dets))
        return landmarks


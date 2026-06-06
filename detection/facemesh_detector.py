import cv2
import mediapipe as mp


class FaceMeshDetector:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]

        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def detect_landmarks(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:

            return results.multi_face_landmarks[0]

        return None

    def get_eye_points(self, landmarks, frame_shape):

        h, w, _ = frame_shape

        left_eye = []
        right_eye = []

        for idx in self.LEFT_EYE:

            point = landmarks.landmark[idx]

            left_eye.append(
                (
                    int(point.x * w),
                    int(point.y * h)
                )
            )

        for idx in self.RIGHT_EYE:

            point = landmarks.landmark[idx]

            right_eye.append(
                (
                    int(point.x * w),
                    int(point.y * h)
                )
            )

        return left_eye, right_eye
import mediapipe as mp
import os
import cv2
import json

# Initialize the pose detection model
mp_pose = mp.solutions.pose


class PoseImage:
    def __init__(self, image_path):
        # File paths
        self.image_path = image_path

        # Load and prepare image
        self.image_bgr = cv2.imread(image_path)

        # Convert into RGB
        # since OpenCV loads in BGR but Mediapipe expects RGB
        # BGR : Blue Green, Red : image processing app
        # RGB : image editing and display app
        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)

        self.results = None
        self.landmarks = []

    def run_pose_detection_model(self):
        with mp_pose.Pose(static_image_mode= True) as pose:
            # Extract the pose landmark
            self.results = pose.process(self.image_rgb)
    
    # landmark -> list of 3D points
    # 0-1 for each
    def extract_pose_landmarks(self):
        if self.results and self.results.pose_landmarks:
            for landmark in self.results.pose_landmarks.landmark:
                self.landmarks.append({
                "x" : landmark.x,
                "y" : landmark.y,
                "z" : landmark.z,
                "visibility" : landmark.visibility # how cofident the coordinate it is 
            })
        return self.landmarks
    
    def save_to_json(self, output_path):
        if self.landmarks:
            with open(output_path,  "w") as f:
                json.dump(self.landmarks, f, indent = 2) # json.dump() method is python's built-in that cleans json data

    def draw_and_visualize(self, window_name = "Arabesque"):
        # Mediapipe: draw the landmarks on the image
        # # dispaly image with Open CV

        if self.results and self.results.pose_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(self.image_bgr, self.results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow(window_name, self.image_bgr)
            cv2.waitKey(0) # click any key to go on to the next image
            cv2.destroyAllWindows()


image1_path = PoseImage("/Users/onofuka/Desktop/ballet arabesque.jpeg") # a girl with blur leo
image1_path.run_pose_detection_model()
image1_path.extract_pose_landmarks()
image1_path.save_to_json("ref_poses/right_arabesque/1.json")
image1_path.draw_and_visualize("Arabesque 1")

image2_path = PoseImage("/Users/onofuka/Desktop/bending_towards_front.jpg") # bending towards
image2_path.run_pose_detection_model()
image2_path.extract_pose_landmarks()
image2_path.save_to_json("ref_poses/right_arabesque/2.json")
image2_path.draw_and_visualize("Arabesque 2")

image3_path = PoseImage("/Users/onofuka/Desktop/ballet arabesque 3.jpeg")
image3_path.run_pose_detection_model()
image3_path.extract_pose_landmarks()
image3_path.save_to_json("ref_poses/right_arabesque/3.json")
image3_path.draw_and_visualize("Arabesque 3")

image4_path = PoseImage("/Users/onofuka/Desktop/ballet arabesque 4.jpg")
image4_path.run_pose_detection_model()
image4_path.extract_pose_landmarks()
image4_path.save_to_json("ref_poses/right_arabesque/4.json")
image4_path.draw_and_visualize("Arabesque 4")

image5_path = PoseImage("/Users/onofuka/Desktop/ballet arabesque low leg.jpg")
image5_path.run_pose_detection_model()
image5_path.extract_pose_landmarks()
image5_path.save_to_json("ref_poses/right_arabesque/5.json")
image5_path.draw_and_visualize("Arabesque 5")


image5_path = PoseImage("/Users/onofuka/Desktop/ballet arabesque hide-heel.jpg")
image5_path.run_pose_detection_model()
image5_path.extract_pose_landmarks()
image5_path.save_to_json("ref_poses/right_arabesque/6.json")
imag
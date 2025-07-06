import mediapipe as mp
import cv2
import json

# Initialize the pose detection model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode= True)
mp_drawing = mp.solutions.drawing_utils

# Load and prepare image
image = cv2.imread("/Users/onofuka/Desktop/ballet arabesque.jpeg")

# Convert to RGB 
# since OpenCV loads in BGR but Mediapipe expects RGB
# BGR : Blue Green, Red : image processing app
# RGB : image editing and display app

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

# Extract the pose landmark
results = pose.process(image_rgb)


# Extract and structure the landmark data
if results.pose_landmarks:
    keypoints = []
    for i, lm in enumerate(results.pose_landmarks.landmark):
        keypoints.append({
            "id" : i,
            "x" : lm.x,
            "y" : lm.y,
            "z" : lm.z,
            "visibility" : lm.visibility
        })

# Save in Json
with open("keypoints_ballet_image.json", "w") as f:
    json.dump(keypoints, f, indent = 2) # json.dump() method is python's built-in that cleans json data

# Visualize the result
# Mediapipe: draw the landmarks on the image
# dispaly image with Open CV


mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
cv2.imshow("Pose", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



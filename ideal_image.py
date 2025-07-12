import mediapipe as mp
import os
import cv2
import json

# Initialize the pose detection model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode= True)
mp_drawing = mp.solutions.drawing_utils

# File paths
image1_path = "/Users/onofuka/Desktop/ballet arabesque.jpeg"
image2_path = "/Users/onofuka/Desktop/ballet arabesque 2.jpeg"
image3_path = "/Users/onofuka/Desktop/ballet arabesque 3.jpeg"

# # Check if files exist
# for path in [image1_path, image2_path]:
#     if not os.path.exists(path):
#         print(f"Error: File not found: {path}")
#         print("Please check the file path and name.")
#         exit()

# Load and prepare image
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)
image3 = cv2.imread(image3_path)

# # Check if images were loaded successfully
# if image1 is None or image2 is None:
#     print("Error: Could not load one or both images.")
#     exit()


# Convert to RGB 
# since OpenCV loads in BGR but Mediapipe expects RGB
# BGR : Blue Green, Red : image processing app
# RGB : image editing and display app

image_rgb1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB) 
image_rgb2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image_rgb3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

# Extract the pose landmark
results1 = pose.process(image_rgb1)
results2 = pose.process(image_rgb2)
results3 = pose.process(image_rgb3)

# Extract and structure the landmark data
if results1.pose_landmarks and results2.pose_landmarks and results3.pose_landmarks:
    keypoints1 = []
    keypoints2 = []
    keypoints3 = []
    for i, lm in enumerate(results1.pose_landmarks.landmark):
        keypoints1.append({
            "id" : i,
            "x" : lm.x,
            "y" : lm.y,
            "z" : lm.z,
            "visibility" : lm.visibility
        })
    for i, lm in enumerate(results2.pose_landmarks.landmark):
        keypoints2.append({
            "id" : i,
            "x" : lm.x,
            "y" : lm.y,
            "z" : lm.z,
            "visibility" : lm.visibility
        })
    for i, lm in enumerate(results3.pose_landmarks.landmark):
        keypoints3.append({
            "id" : i,
            "x" : lm.x,
            "y" : lm.y,
            "z" : lm.z,
            "visibility" : lm.visibility
        })



# Save in Json
with open("arabesque_ref1.json", "w") as f:
    json.dump(keypoints1, f, indent = 2) # json.dump() method is python's built-in that cleans json data

with open("arabesque_ref2.json", "w") as f:
    json.dump(keypoints2, f, indent = 2) # json.dump() method is python's built-in that cleans json data

with open("arabesque_ref3.json", "w") as f:
    json.dump(keypoints3, f, indent = 2) # json.dump() method is python's built-in that cleans json data


# Visualize the result
# Mediapipe: draw the landmarks on the image
# dispaly image with Open CV


mp_drawing.draw_landmarks(image1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS)
mp_drawing.draw_landmarks(image2, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
mp_drawing.draw_landmarks(image3, results3.pose_landmarks, mp_pose.POSE_CONNECTIONS)

cv2.imshow("Arabesque1", image1)
cv2.imshow("Arabesque2", image2)
cv2.imshow("Arabesque3", image3)

cv2.waitKey(0)
cv2.destroyAllWindows()



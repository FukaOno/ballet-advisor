# Find the best matching pic

import os

# call the class 
from utils import Normalize_Image, LoadJson, Similarity, Feedback

target_pose = "right_arabesque"
ref_dir = f"ref_poses/{target_pose}/"

user_data = LoadJson("user_uploaded_pose.json").load_json()
user_pose = Normalize_Image(user_data).normalize_keypoints()

best_score = float("inf") # Starting w infinity-> lower score = more similar
best_match_filename = None # filename with best matching pose to the user
best_match_pose = None

for filename in os.listdir(ref_dir):
    if filename.endswith(".json"):

        ref_data = LoadJson(os.path.join(ref_dir, filename)).load_json()
        ref_pose = Normalize_Image(ref_data).normalize_keypoints()

        score = Similarity(user_pose, ref_pose).co
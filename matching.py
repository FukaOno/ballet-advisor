# matching.py

from utils import Normalize_Image, LoadJson, Similarity, Feedback
import os

def get_best_match_and_feedback(user_pose, ref_dir):
    best_score = float("inf")  # Lower score = more similar
    best_match_filename = None
    best_match_pose = None

    for filename in os.listdir(ref_dir):
        if filename.endswith(".json"):
            ref_data = LoadJson(os.path.join(ref_dir, filename)).load_json()
            ref_pose = Normalize_Image(ref_data).normalize_keypoints()
            score = Similarity(user_pose, ref_pose).compute_similarity()

            if score < best_score:
                best_score = score
                best_match_filename = filename
                best_match_pose = ref_pose

    feedback = Feedback(user_pose, best_match_filename).generate_feedback()
    return best_match_filename, feedback, best_score


# このファイルを単体で実行したとき用のテストコード（任意）
if __name__ == "__main__":
    user_data = LoadJson("user_uploaded_pose.json").load_json()
    user_pose = Normalize_Image(user_data).normalize_keypoints()
    ref_dir = "ref_poses/right_arabesque/"

    filename, feedback, score = get_best_match_and_feedback(user_pose, ref_dir)
    print("Best Match:", filename)
    print("Score:", score)
    print("Feedback:", feedback)

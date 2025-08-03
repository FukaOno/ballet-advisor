import math
import json

class Normalize_Image:
    def __init__(self, landmarks):
        self.landmarks = landmarks
        
    def normalize_keypoints(self):
        landmarks = self.landmarks
        # Get the list of all x, y coordinates of keypoints
        xs = []
        for lm in landmarks: # each coordinate in each dictionary in json file, 
            xs.append(lm['x']) # add only the x values
        
        ys = []
        for lm in landmarks: 
            ys.append(lm['y'])
        
        # Center of the body= sum of all the x coordinate / number of all the x coordinate
        x_center = sum(xs) / len(xs)
        y_center = sum(ys) / len(ys)

        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        # the size of the entire body
        scale = max(width, height) 


        # coordinates that shows ONLY the pose (independent on the size of image, size of the person)
        normalized = []
        for lm in landmarks:
            # How far from the center of the body
        
            x_length_from_center = lm['x'] - x_center
            y_length_from_center = lm['y'] - y_center
            
            # Get the rate of length from the center
            x_rate = x_length_from_center / scale
            y_rate = y_length_from_center / scale

            normalized.append({
                'x' : x_rate,
                'y': y_rate
            })
        return normalized
    

class Similarity:
    def __init__(self, user_pose, reference_pose):
        self.user_pose = user_pose
        self.reference_pose = reference_pose
    
    def compute_similarity(self, user_pose, reference_pose):
        total_distance = 0
        for u_lm, r_lm in zip(user_pose, reference_pose):
            xd = u_lm['x'] - r_lm['x']
            yd = u_lm['y'] - r_lm['y']

            # dx = x2 - x1
            # dy = y2 - y1
            # d = √((x2 - x1)² + (y2 - y1)²)

            u_distance = math.sqrt(xd ** 2 + yd ** 2)
            total_distance += u_distance
        avg_distance = total_distance / len(user_pose)
        return avg_distance


# load the json file and make it into python dictionary
class LoadJson:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path

    def load_json(self):
        with open(self.json_file_path, "r") as json_file:
            data = json.load(json_file)
        return data

class Feedback:
    def __init__(self,user_pose, best_match_filename):
        self.user_pose = user_pose
        self.best_match_filename= best_match_filename
    
    def generate_feedback(self):
        if self.best_match_filename.endswith("1.json"):
            return ["You are actually perfect!"]
        elif self.best_match_filename.endswith("2.json"):
            return ["Your body is bending too foward! Try to make your body up! It's okay to lower your back leg!"]
        elif self.best_match_filename.endswith("3.json"):
            return ["Your arabesque is beautiful! stunning! You are doing great! Keep up the left side too! "]
        elif self.best_match_filename.endswith("4.json"):
            return ["Your arabesque is beautiful! stunning! Keep up the left side too!"]
        elif self.best_match_filename.endswith("5.json"):
            return ["Your back leg is too low! Try to keep your body straight and lift your back leg higher."]
        elif self.best_match_filename.endswith("6.json"):
            return ["Your posture is perfect! BUT one thing to fix! Try to turn out your back leg to hide your heel! Keep going!"]

# loader = LoadJson("arabesque/1.json")
# print(loader.load_json())
import streamlit as st
import tempfile
import cv2
import mediapipe as mp
import numpy as np
import os
import json
from utils import Normalize_Image, Similarity, Feedback
from matching import get_best_match_and_feedback

st.title("🩰 Ballet Pose Feedback Advisor")

pose_type = st.selectbox("Select Pose Type", ["right_arabesque"])

uploaded_image = st.file_uploader("Upload your ballet pose image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # 保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img_file:
        temp_img_file.write(uploaded_image.read())
        img_path = temp_img_file.name

    # 表示
    st.image(img_path, caption="Uploaded Image", use_column_width=True)

    # Mediapipe で keypoints 抽出
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # ランドマーク抽出
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })

        # 正規化
        user_pose = Normalize_Image(landmarks).normalize_keypoints()

        # 比較 & フィードバック
        ref_dir = f"ref_poses/{pose_type}/"
        best_match_filename, feedback, score = get_best_match_and_feedback(user_pose, ref_dir)

        
        st.markdown("### 🗣️ Feedback:")
        for tip in feedback:
            st.write("- " + tip)
    else:
        st.error("No pose detected. Try another photo with a clearer full-body view.")

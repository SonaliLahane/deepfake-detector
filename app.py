import streamlit as st
from transformers import pipeline
import tempfile
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="DeepFake Detection", layout="centered")
st.title("🎭 Face-Swap DeepFake Detector")
st.write("Upload a video and detect if it contains face-swap based deepfakes.")

model = pipeline("image-classification", model="microsoft/resnet-50")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save the uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    # Use OpenCV to extract middle frame
    cap = cv2.VideoCapture(temp_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    success, frame = cap.read()
    cap.release()

    if success:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        st.image(image, caption="Extracted Frame for Analysis", use_column_width=True)

        st.write("🔍 Running deepfake detection...")
        result = model(image)
        top_result = result[0]

        st.success("✅ Detection Complete!")
        st.markdown(f"**Prediction:** {top_result['label']}")
        st.markdown(f"**Confidence:** {top_result['score'] * 100:.2f}%")
        st.info("⚠️ Note: This is a placeholder model. You can switch to a deepfake-specific model from Hugging Face.")
    else:
        st.error("❌ Failed to extract frame from video.")

import streamlit as st
from moviepy.editor import VideoFileClip
from transformers import pipeline
import tempfile
from PIL import Image
import numpy as np

st.set_page_config(page_title="DeepFake Detection", layout="centered")
st.title("🎭 Face-Swap DeepFake Detector")
st.write("Upload a video and detect if it contains face-swap based deepfakes.")

model = pipeline("image-classification", model="microsoft/resnet-50")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    clip = VideoFileClip(temp_video_path)
    frame_time = clip.duration / 2
    frame = clip.get_frame(frame_time)

    image = Image.fromarray(np.uint8(frame)).convert("RGB")
    st.image(image, caption="Extracted Frame for Analysis", use_column_width=True)

    st.write("🔍 Running deepfake detection...")
    result = model(image)
    top_result = result[0]

    st.success("✅ Detection Complete!")
    st.markdown(f"**Prediction:** {top_result['label']}")
    st.markdown(f"**Confidence:** {top_result['score'] * 100:.2f}%")
    st.info("⚠️ Note: This is a placeholder model. You can switch to a deepfake-specific model from Hugging Face.")
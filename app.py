import streamlit as st
import os
import uuid
import cv2
import numpy as np
from PIL import Image
from src.infer import predict_single_image, load_model

# Set page config
st.set_page_config(page_title="Retinal Vessel Segmentation", layout="centered")

# -------------------------------
# SECTION 1: Title & Description
# -------------------------------
st.markdown("""
# 🧠 Retinal Vessel Segmentation using DeepVessel-Net

Welcome to **DeepVessel-Net**, a deep learning–powered tool that segments blood vessels from **retinal fundus images**. 
This application is designed to support **ophthalmologists, researchers, and AI developers** working on **diabetic retinopathy**, **glaucoma**, and other vision-related diseases.

---

📂 Upload a retinal image and the model will return a **binary vessel segmentation mask**.

""")

# -------------------------------
# SECTION 2: File Upload
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload your image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

# -------------------------------
# SECTION 3: Load Model
# -------------------------------
@st.cache_resource
def load_segmentation_model():
    model_path = "outputs/checkpoints/best_model.pth"
    model = load_model(model_path)
    return model

model = load_segmentation_model()

# -------------------------------
# SECTION 4: Predict on Upload
# -------------------------------
if uploaded_file is not None:
    input_id = str(uuid.uuid4())
    input_path = os.path.join("temp", f"{input_id}_input.jpg")
    os.makedirs("temp", exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show uploaded image
    st.markdown("### 🖼 Uploaded Image")
    st.image(uploaded_file, use_container_width=True)

    # Predict
    segmented = predict_single_image(input_path, model, threshold=0.65)
    output_path = os.path.join("temp", f"{input_id}_segmented.png")
    cv2.imwrite(output_path, segmented)

    st.markdown("### 🧪 Segmented Vessels")
    st.image(segmented, clamp=True, channels="GRAY", use_container_width=True)

    with open(output_path, "rb") as f:
        st.download_button(
            label="📥 Download Result",
            data=f,
            file_name="segmented_retina.png",
            mime="image/png"
        )

# -------------------------------
# SECTION 5: More Info
# -------------------------------
with st.expander("🔬 About DeepVessel-Net"):
    st.markdown("""
DeepVessel-Net is a **U-Net-based deep learning model** trained on the **DRIVE retinal vessel segmentation dataset**.
It segments **thin and thick blood vessels** in high-resolution fundus images and can be used for:

- 🩺 **Early screening** of diabetic retinopathy, glaucoma, and hypertension
- 🧪 **Research on vascular morphology**
- ⚙️ **Feature extraction** for downstream tasks like classification

**Architecture**: U-Net with batch normalization and dropout  
**Input Size**: 512×512 RGB  
**Output**: Binary vessel mask  
**Framework**: PyTorch  
    """)

with st.expander("📊 Model Performance"):
    st.markdown("""
| Metric        | Value   |
|---------------|---------|
| Dice Score    | **0.8105** |
| IoU Score     | **0.6811** |
| Sensitivity   | **0.8282** |
| Specificity   | **0.9797** |

> Threshold used: **0.65** (optimized on test set)
    """)

with st.expander("🚀 How to Use"):
    st.markdown("""
1. Upload a **retinal fundus image** (JPG or PNG).
2. Wait for the **segmentation mask** to appear below.
3. Use the **Download** button to save the result.
4. Use output for research, analysis, or diagnostic support.
    """)

with st.expander("📎 Credits & GitHub"):
    st.markdown("""
- Built by ADITH SREE JEY A S using PyTorch + Streamlit  
- Based on the **DRIVE** dataset    
- GitHub: [🔗 Project Repo](https://github.com/Adith-Sree-Jey/DeepVessel_Net.git)

Feel free to fork, cite, and contribute!
    """)

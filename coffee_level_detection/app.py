import streamlit as st
from PIL import Image
import pandas as pd
import os
from fuzed_models import MaskSegmenterModel, ContourColorModel, DepthMiDaSModel, ZoeDepthModel, fuse_estimates

def get_coffee_level(image, lidar_distance=None):
    """Return a fused coffee level percentage (0-100)."""
    # Initialize models (in practice, these might be singletons to save resources)
    mask_model = MaskSegmenterModel()
    cv_model = ContourColorModel()
    depth_model = DepthMiDaSModel()
    zoeDepth = ZoeDepthModel()

    # Get individual predictions
    
    # Fuse predictions into a single percentage
    estimates = {
        'mask_pred': mask_model.estimate_fill(image),
        'cv_pred': cv_model.estimate_fill(image),
        'depth_pred': depth_model.estimate_fill(image),
        'zoe_pred': zoeDepth.estimate_fill(image)
    }
    fused = fuse_estimates(estimates=estimates)
    return fused

# Data storage
DATA_PATH = "ratings.csv"
IMG_DIR = "coffee_level_detection/img"

# Get list of image files
image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
image_files.sort()

if "img_idx" not in st.session_state:
    st.session_state.img_idx = 0

st.title("Coffee Level Rater & Training Data Generator")

if not image_files:
    st.warning("No images found in the img/ folder.")
else:
    # Navigation
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("Previous") and st.session_state.img_idx > 0:
            st.session_state.img_idx -= 1
    with col3:
        if st.button("Next") and st.session_state.img_idx < len(image_files) - 1:
            st.session_state.img_idx += 1

    current_file = image_files[st.session_state.img_idx]
    image_path = os.path.join(IMG_DIR, current_file)
    image = Image.open(image_path)
    st.image(image, caption=current_file, use_column_width=True)

    # Model prediction
    prediction = get_coffee_level(image_path)
    if prediction is None:
        st.warning("Could not detect coffee or pot in the image.")
        prediction_display = "N/A"
        prediction_value = 0
    else:
        prediction_display = f"{prediction:.1f}%"
        prediction_value = int(prediction)
    st.write(f"**Model Prediction:** {prediction_display}")

    # User rating
    user_rating = st.slider("Your rating of coffee level (%)", 0, 100, prediction_value)
    st.write(f"**Your Rating:** {user_rating}%")

    # Save rating
    if st.button("Save Rating"):
        if not os.path.exists(DATA_PATH):
            df = pd.DataFrame(columns=["filename", "user_rating", "model_prediction"])
        else:
            df = pd.read_csv(DATA_PATH)
        df = pd.concat([
            df,
            pd.DataFrame([{
                "filename": current_file,
                "user_rating": user_rating,
                "model_prediction": prediction
            }])
        ], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)
        st.success("Rating saved!")

    # Show all data
    if os.path.exists(DATA_PATH):
        st.subheader("Collected Training Data")
        st.dataframe(pd.read_csv(DATA_PATH))
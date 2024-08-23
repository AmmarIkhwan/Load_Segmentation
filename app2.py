import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import tempfile

# Load the trained YOLOv8 model for segmentation
model = YOLO('best.pt', task='segment')

# Streamlit UI
st.set_page_config(page_title="YOLOv8 Image Segmentation", layout="wide")
st.title("🖼️ YOLOv8 Image Segmentation")

# Sidebar
st.sidebar.header("Settings")
st.sidebar.write("Configure your segmentation options here.")

# Add an option to show bounding boxes or segmentation masks
show_masks = st.sidebar.checkbox("Show Segmentation Masks", value=True)
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)

st.write("Upload images to perform segmentation using the YOLOv8 model.")
st.write("")

# File uploader to upload images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Temporary directory to save results
temp_dir = tempfile.TemporaryDirectory()

if uploaded_files:
    progress_text = "Processing images..."
    my_bar = st.progress(0)

    for i, uploaded_file in enumerate(uploaded_files):
        # Open the image file
        image = Image.open(uploaded_file)
        
        # Save the uploaded file temporarily
        temp_image_path = os.path.join(temp_dir.name, uploaded_file.name)
        image.save(temp_image_path)

        # Run inference
        results = model(temp_image_path)

        if results:
            for j, result in enumerate(results):
                # Construct save path
                save_path = os.path.join(temp_dir.name, f'result_{os.path.splitext(uploaded_file.name)[0]}_{j}.jpg')

                # Save the results
                result.plot(save=True, filename=save_path)

                # Display the original image and the result side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption='Uploaded Image', use_column_width=True)
                with col2:
                    st.image(save_path, caption=f"Result {j+1} for {uploaded_file.name}", use_column_width=True)

                # Show additional result information if selected
                if show_masks and result.masks is not None:
                    st.write("Segmentation Masks:")
                    st.write(result.masks)

                if show_boxes and result.boxes is not None:
                    st.write("Bounding Boxes:")
                    st.write(result.boxes)

            st.success(f"Results processed for {uploaded_file.name}")
        else:
            st.warning(f"No results found for {uploaded_file.name}.")

        # Update progress bar
        my_bar.progress((i + 1) / len(uploaded_files))

    st.write("All images have been processed.")
else:
    st.info("Please upload one or more images to start the segmentation process.")

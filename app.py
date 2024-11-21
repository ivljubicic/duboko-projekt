import streamlit as st
from PIL import Image
import requests
import base64
import io
import os
from dotenv import load_dotenv
import time  # Corrected import statement

# Load environment variables
load_dotenv()

def load_image(image_file):
    img = Image.open(image_file)
    return img

def encode_image_to_base64(image):
    # Convert PIL Image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_runpod_api(content_image_base64, style_image_base64, image_size):
    runpod_key = os.getenv('RUNPOD_API_KEY')
    endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID')
    runpod_url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"

    headers = {
        "accept": "application/json",
        "authorization": runpod_key,
        "content-type": "application/json"
    }

    input_data = {
        "input": {
            "content_image": content_image_base64,
            "style_image": style_image_base64,
            "image_size": image_size
        }
    }

    try:
        # Start timer
        start_time = time.time()

        # Initial API call to start the job
        response = requests.post(runpod_url, headers=headers, json=input_data, timeout=30).json()

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        if response.get("status") == "COMPLETED":
            st.success(f"Style transfer completed in {elapsed_time:.2f} seconds")
            return response.get("output"), elapsed_time
        else:
            st.error("Style transfer failed")
            return None, None

    except requests.exceptions.RequestException as e:
        st.error(f"Error calling RunPod API: {str(e)}")
        return None, None

def main():
    st.title("Neural Style Transfer")
    st.write("Upload your content and style images to create artistic compositions")

    # Create two columns for image upload
    col1, col2 = st.columns(2)

    content_image = None
    style_image = None

    with col1:
        st.subheader("Content Image")
        content_file = st.file_uploader("Choose your target image", type=["png", "jpg", "jpeg"])
        if content_file is not None:
            content_image = load_image(content_file)
            st.image(content_image, caption="Target Image", use_container_width=True)

    with col2:
        st.subheader("Style Image")
        style_file = st.file_uploader("Choose your style image", type=["png", "jpg", "jpeg"])
        if style_file is not None:
            style_image = load_image(style_file)
            st.image(style_image, caption="Style Image", use_container_width=True)

    # Add image size selector
    st.sidebar.subheader("Settings")
    image_size = st.sidebar.slider("Output Image Size", 128, 1024, 512, 128)
    
    if content_file is not None and style_file is not None:
        if st.button("Generate Styled Image"):
            with st.spinner("Initializing style transfer..."):
                # Convert images to base64
                content_base64 = encode_image_to_base64(content_image)
                style_base64 = encode_image_to_base64(style_image)

                # Call RunPod API with status updates
                result, elapsed_time = call_runpod_api(content_base64, style_base64, image_size)

                if result and 'final_image' in result:
                    # Decode the base64 image
                    output_image_bytes = base64.b64decode(result['final_image'])
                    output_image = Image.open(io.BytesIO(output_image_bytes))
                    
                    # Display the result
                    st.subheader("Generated Image")
                    st.image(output_image, caption="Style Transfer Result", use_container_width=True)
                    
                    # Show elapsed time
                    st.write(f"Time taken for style transfer: {elapsed_time:.2f} seconds")
                    
                    # Add download button
                    buffered = io.BytesIO()
                    output_image.save(buffered, format="PNG")
                    st.download_button(
                        label="Download image",
                        data=buffered.getvalue(),
                        file_name="style_transfer_result.png",
                        mime="image/png"
                    )
                else:
                    st.error("Failed to generate styled image. Please try again.")

if __name__ == "__main__":
    main() 
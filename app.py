import streamlit as st
from PIL import Image
import requests
import base64
import io
import os
from dotenv import load_dotenv
from time import sleep

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
        # Initial API call to start the job
        response = requests.post(runpod_url, headers=headers, json=input_data, timeout=30).json()

        if response.get("status") == "COMPLETED":
            return response.get("output")
        else:
            st.error("Style transfer failed")
            return None
        # job_id = response["id"]

        # # Create a status placeholder
        # status_placeholder = st.empty()
        
        # file_parsed = False
        # while not file_parsed:
        #     status_placeholder.text("Waiting for style transfer to complete...")
        #     waiting_response = requests.get(
        #         f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}", 
        #         headers=headers, 
        #         timeout=10
        #     )
            
        #     status = waiting_response.json()["status"]
        #     status_placeholder.text(f"Status: {status}")
            
        #     if status == "COMPLETED":
        #         file_parsed = True
        #         return waiting_response.json()["output"]
        #     elif status == "FAILED":
        #         st.error("Style transfer failed")
        #         return None
            
        #     sleep(2)

    except requests.exceptions.RequestException as e:
        st.error(f"Error calling RunPod API: {str(e)}")
        return None

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
                result = call_runpod_api(content_base64, style_base64, image_size)

                if result and 'final_image' in result:
                    # Decode the base64 image
                    output_image_bytes = base64.b64decode(result['final_image'])
                    output_image = Image.open(io.BytesIO(output_image_bytes))
                    
                    # Display the result
                    st.subheader("Generated Image")
                    st.image(output_image, caption="Style Transfer Result", use_container_width=True)
                    
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
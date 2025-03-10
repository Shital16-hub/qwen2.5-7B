import os
import time
import subprocess
import requests
import json
import base64
from io import BytesIO
from PIL import Image

# Start the FastAPI application in the background
subprocess.Popen(["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"])

# Wait for the server to start
time.sleep(5)

def handler(event):
    try:
        # Get input data
        input_data = event["input"]
        
        # Check if there's an image in the input
        if "image" not in input_data:
            return {
                "status": "error",
                "message": "No image provided"
            }
        
        # Get the image data
        image_data = input_data["image"]
        
        # Get optional user prompt
        user_prompt = input_data.get("prompt", None)
        
        # Check if the image is a URL or base64
        if image_data.startswith(("http://", "https://")):
            # Download the image
            response = requests.get(image_data)
            if response.status_code != 200:
                return {
                    "status": "error",
                    "message": f"Failed to download image: {response.status_code}"
                }
            image_bytes = BytesIO(response.content)
        else:
            # Assume base64 encoded image
            if "base64," in image_data:
                image_data = image_data.split("base64,")[1]
            image_bytes = BytesIO(base64.b64decode(image_data))
        
        # Prepare the files and data for the FastAPI request
        files = {
            "file": ("image.jpg", image_bytes, "image/jpeg")
        }
        
        data = {}
        if user_prompt:
            data["user_prompt"] = user_prompt
        
        # Send the request to the FastAPI server
        response = requests.post(
            "http://localhost:8000/extract-text",
            files=files,
            data=data
        )
        
        if response.status_code != 200:
            return {
                "status": "error",
                "message": f"API error: {response.text}"
            }
        
        # Return the result
        return {
            "status": "success",
            "output": response.json()["extracted_text"]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
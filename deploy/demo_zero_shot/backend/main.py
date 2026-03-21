"""
FastAPI Backend for Image Caption Demo
Simple API endpoint that accepts image uploads and returns a fixed caption.
Ready for ML model integration in the future.
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
# from inference import build_vision_tower, zero_shot_classify
from inference_onnx import ONNXVisionTower, zero_shot_classify

# Load vision tower and prompt data at startup
# vision_tower = build_vision_tower()
# Load ONNX vision tower at startup
vision_tower = ONNXVisionTower()

# Initialize FastAPI application
app = FastAPI(title="Image Caption API", version="1.0.0")


# Enable CORS middleware to allow frontend requests
# In production, restrict origins to your domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (suitable for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Endpoint to receive an image and return a caption.
    
    Currently returns a fixed response as the ML model is not yet loaded.
    This serves as a placeholder for future ML model integration.
    
    Args:
        image (UploadFile): The image file uploaded via multipart/form-data
        
    Returns:
        dict: JSON response containing the caption
        
    Example:
        POST /predict
        Content-Type: multipart/form-data
        
        Response:
        {
            "caption": "This is a fixed caption from the backend. Model not yet loaded."
        }
    """
    # Convert image to PIL format
    pil_image = Image.open(image.file)
    # Predict caption using zero-shot classification with ONNX vision tower
    captions = zero_shot_classify(pil_image, vision_tower, top_k=1)
    return {
        "caption": captions[0] if captions else "No caption generated."
    }


# Health check endpoint
@app.get("/")
def read_root():
    """
    Health check endpoint.
    """
    return {"message": "Image Caption API is running. Use POST /predict with an image."}


if __name__ == "__main__":
    # Run the application with: python main.py
    # Or use: uvicorn main:app --reload
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

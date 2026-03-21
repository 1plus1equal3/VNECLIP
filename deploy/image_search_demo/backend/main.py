"""
Image Search Demo Backend using FastAPI
Supports both text-based and image-based search using CLIP ONNX Model
"""

import base64
import logging
import sys
import os
from typing import List, Optional, Union
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import numpy as np

# Add mm_rag to path for importing MultimodalRetrievalONNX
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../mm_rag'))

# ==================== Logging Setup ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== FastAPI App Initialization ====================
app = FastAPI(
    title="Image Search Demo API",
    description="A simple API for searching images by text or image",
    version="1.0.0"
)

# ==================== CORS Configuration ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Global Variables ====================
retrieval_client = None

# ==================== Data Models ====================
class SearchResult(BaseModel):
    """Data model for search results"""
    image_url: str
    caption_id: str
    caption: str


class SearchRequest(BaseModel):
    """Data model for search requests"""
    query_text: Optional[str] = None
    query_image: Optional[str] = None  # Base64 encoded image


# ==================== Mock Data ====================
MOCK_RESULTS = [
    SearchResult(
        image_url="https://picsum.photos/300/200?random=1",
        caption_id="IMG_001",
        caption="A red sports car on a sunny road"
    ),
    SearchResult(
        image_url="https://picsum.photos/300/200?random=2",
        caption_id="IMG_002",
        caption="Modern cityscape with tall buildings"
    ),
    SearchResult(
        image_url="https://picsum.photos/300/200?random=3",
        caption_id="IMG_003",
        caption="Mountain landscape during golden hour"
    ),
    SearchResult(
        image_url="https://picsum.photos/300/200?random=4",
        caption_id="IMG_004",
        caption="Person holding a coffee cup outdoors"
    ),
    SearchResult(
        image_url="https://picsum.photos/300/200?random=5",
        caption_id="IMG_005",
        caption="Beautiful sunset over ocean"
    ),
    SearchResult(
        image_url="https://picsum.photos/300/200?random=6",
        caption_id="IMG_006",
        caption="Forest path covered with autumn leaves"
    ),
]

# ==================== Helper Functions ====================
def convert_to_api_url(image_path: str) -> str:
    """
    Convert filesystem image path to API endpoint URL for frontend access.
    
    Converts:
    /root/Project/brick_vidgen/vnclip/deploy/mm_rag/data/uitvic_dataset/...
    to:
    http://localhost:5000/api/image/uitvic_dataset/...
    
    This allows the frontend to fetch images through the backend API endpoint.
    
    Args:
        image_path: Full filesystem path to image
        
    Returns:
        API URL that frontend can use to fetch the image
    """
    
    # Extract relative path after 'data/'
    data_prefix = "/root/Project/brick_vidgen/vnclip/dataset/"
    if image_path.startswith(data_prefix):
        relative_path = image_path[len(data_prefix):]
        return f"http://192.168.0.100:5002/api/image/{relative_path}"
    
    return image_path


# ==================== Search Functions ====================
# TODO: INTEGRATION POINT FOR CLIP ONNX MODEL
# Replace these placeholder functions with actual CLIP model inference:
# 1. Load CLIP ONNX model (vision_tower.onnx and text_tower.onnx)
# 2. Preprocess image/text inputs
# 3. Get embeddings from CLIP
# 4. Compute similarity scores with your image database
# 5. Return top-k results based on similarity scores
# ===============================================================

def process_text_search(text: str) -> List[SearchResult]:
    """
    Perform text-based search using CLIP ONNX model.
    
    Args:
        text: Search query text
        
    Returns:
        List of SearchResult objects ranked by similarity
        
    Raises:
        ValueError: If retrieval client is not initialized or search fails
    """
    if retrieval_client is None:
        raise ValueError("Retrieval client not initialized")
    
    logger.info(f"Processing text search: {text}")
    
    try:
        # Search using CLIP encoded text embeddings
        # ChromaDB returns: {"ids": [[...]], "distances": [[...]], "metadatas": [[...]]}
        results = retrieval_client.search(text, top_k=10)
        
        # Transform results to SearchResult format
        search_results = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        for id, distance, metadata in zip(ids, distances, metadatas):
            search_results.append(SearchResult(
                image_url=convert_to_api_url(metadata.get('image_path', '')),
                caption_id=str(metadata.get('caption_id', id)),
                caption=metadata.get('caption', 'No caption available')
            ))
        
        logger.info(f"Text search returned {len(search_results)} results")
        return search_results
        
    except Exception as e:
        logger.error(f"Text search error: {e}")
        raise ValueError(f"Text search failed: {e}")


def process_image_search(image_base64: str) -> List[SearchResult]:
    """
    Perform image-based search using CLIP ONNX model.
    
    In production, this:
    1. Decodes Base64 image
    2. Converts to PIL Image
    3. Searches using CLIP vision tower ONNX model
    4. Returns ranked results based on similarity
    
    Args:
        image_base64: Base64 encoded image string
        
    Returns:
        List of SearchResult objects ranked by similarity
        
    Raises:
        ValueError: If image decoding or search fails
    """
    if retrieval_client is None:
        raise ValueError("Retrieval client not initialized")
    
    logger.info("Processing image search")
    
    try:
        # Decode Base64 image
        image_data = base64.b64decode(image_base64)
        logger.info(f"Image size: {len(image_data)} bytes")
        
        # Convert to PIL Image
        image = Image.open(BytesIO(image_data)).convert('RGB')
        logger.info(f"Image shape: {image.size}")
        
        # Search using CLIP encoded image embeddings
        # ChromaDB returns: {"ids": [[...]], "distances": [[...]], "metadatas": [[...]]}
        results = retrieval_client.search(image, top_k=10)
        
        # Transform results to SearchResult format
        search_results = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        for id, distance, metadata in zip(ids, distances, metadatas):
            search_results.append(SearchResult(
                image_url=convert_to_api_url(metadata.get('image_path', '')),
                caption_id=str(metadata.get('caption_id', id)),
                caption=metadata.get('caption', 'No caption available')
            ))
        
        logger.info(f"Image search returned {len(search_results)} results")
        return search_results
        
    except Exception as e:
        logger.error(f"Image search error: {e}")
        raise ValueError(f"Image search failed: {e}")


# ==================== API Endpoints ====================
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Image Search Demo API",
        "version": "1.0.0"
    }


@app.post("/api/search", response_model=List[SearchResult], tags=["Search"])
async def search(request: SearchRequest) -> List[SearchResult]:
    """
    Main search endpoint supporting both text and image queries.
    
    Uses CLIP ONNX model for text and image encoding, then performs
    similarity search against the image database.
    
    Args:
        request: SearchRequest containing either query_text or query_image
        
    Returns:
        List of SearchResult objects with image_url, caption_id, and caption
        
    Raises:
        HTTPException: If both or neither query types are provided
    """
    # Validate input
    if request.query_text and request.query_image:
        raise HTTPException(
            status_code=400,
            detail="Cannot use both text and image queries simultaneously. Provide only one."
        )
    
    if not request.query_text and not request.query_image:
        raise HTTPException(
            status_code=400,
            detail="Must provide either query_text or query_image"
        )
    
    try:
        if request.query_text:
            results = process_text_search(request.query_text)
        else:
            results = process_image_search(request.query_image)
        
        logger.info(f"Search completed, returning {len(results)} results")
        return results
        
    except ValueError as e:
        # Fall back to mock data if retrieval client is not initialized
        if retrieval_client is None:
            logger.info("Using mock data (retrieval client not initialized)")
            return MOCK_RESULTS
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/image/{image_path:path}", tags=["Image"])
async def get_image(image_path: str):
    """
    Serve image files from the dataset.
    
    Allows frontend to fetch images via API endpoint.
    
    Args:
        image_path: Relative path to image (e.g., uitvic_dataset/coco_uitvic_train/...)
        
    Returns:
        Image file with appropriate content type
        
    Raises:
        HTTPException: If file not found or path is invalid
    """
    try:
        # Prevent directory traversal attacks
        if ".." in image_path:
            raise HTTPException(status_code=403, detail="Invalid path")
        
        # Construct full path
        base_path = os.path.join(os.path.dirname(__file__), "../../mm_rag/data")
        full_path = os.path.normpath(os.path.join(base_path, image_path))
        
        # Verify path is within base path
        if not full_path.startswith(os.path.normpath(base_path)):
            raise HTTPException(status_code=403, detail="Invalid path")
        
        # Check if file exists
        if not os.path.isfile(full_path):
            logger.warning(f"Image not found: {full_path}")
            raise HTTPException(status_code=404, detail="Image not found")
        
        logger.info(f"Serving image: {image_path}")
        return FileResponse(full_path, media_type="image/jpeg")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ==================== Error Handlers ====================
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unexpected error: {exc}")
    return {
        "detail": "An unexpected error occurred",
        "type": type(exc).__name__
    }


# ==================== Startup Event ====================
@app.on_event("startup")
async def startup_event():
    """Application startup - Initialize retrieval client"""
    global retrieval_client
    
    logger.info("Image Search Demo API starting up...")
    logger.info("CORS enabled for all origins (demo mode)")
    
    try:
        # Import here to avoid import errors if mm_rag dependencies are missing
        from retrieval_onnx import MultimodalRetrievalONNX
        
        logger.info("Initializing MultimodalRetrievalONNX client...")
        
        # Initialize with ONNX models
        retrieval_client = MultimodalRetrievalONNX(
            db_path="../../mm_rag/clip_db",
            collection_name="clip_retrieval",
            vision_tower_onnx="../../weight/vision_tower.onnx",
            text_tower_onnx="../../weight/text_tower.onnx",
            device="cpu",
            use_onnx=True
        )
        
        logger.info("✓ Retrieval client initialized successfully")
        logger.info("Ready to accept requests on /api/search")
        
    except ImportError as e:
        logger.warning(f"Could not import MultimodalRetrievalONNX: {e}")
        logger.warning("Will use mock data for responses")
    except Exception as e:
        logger.error(f"Failed to initialize retrieval client: {e}")
        logger.warning("Will use mock data for responses")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5002,
        log_level="info"
    )

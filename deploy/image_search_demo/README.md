# 🔍 Image Search Demo

A lightweight, fast, and simple web application for searching images using both text queries and image uploads/camera capture. Built with FastAPI and vanilla JavaScript.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Frontend Features](#frontend-features)
- [Integrating CLIP ONNX Model](#integrating-clip-onnx-model)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

---

## 🎯 Project Overview

This demo application provides a simple yet powerful interface for image search with two modalities:

1. **Text-based Search**: Enter descriptive text to find similar images
2. **Image-based Search**: Upload or capture an image from your camera to find visually similar images

The application uses **FastAPI** for the backend and **vanilla JavaScript** for the frontend, with no heavy dependencies. The image processing happens entirely on the client-side using HTML5 Canvas API.

---

## ✨ Features

### Frontend
- ✅ **Mobile-First Design**: Responsive UI optimized for all devices
- ✅ **Camera Capture**: Directly capture images from your device camera
- ✅ **Image Upload**: Select images from your device
- ✅ **Client-Side Processing**: Images are resized to 224×224px using HTML5 Canvas
- ✅ **Real-time Preview**: Preview selected/captured images before search
- ✅ **Error Handling**: User-friendly error messages
- ✅ **Single HTML File**: No build process required

### Backend
- ✅ **FastAPI Framework**: Modern, fast async Python framework
- ✅ **CORS Enabled**: Ready for frontend integration
- ✅ **Both Search Modes**: Text and image search support
- ✅ **Mock Data Ready**: Includes placeholder responses
- ✅ **Comprehensive Logging**: Track all operations
- ✅ **Model Integration Comments**: Clear TODO sections for CLIP ONNX integration

---

## 📁 Project Structure

```
image_search_demo/
├── frontend/
│   └── index.html                 # Single HTML file with styling and JS
├── backend/
│   ├── main.py                    # FastAPI application
│   └── requirements.txt           # Python dependencies
├── start.sh                       # Start both services
├── stop.sh                        # Stop both services
└── README.md                      # This file
```

---

## 📦 Prerequisites

Ensure you have the following installed on your system:

- **Python 3.8+** - Available at https://www.python.org/downloads/
- **pip** - Python package manager (usually included with Python)
- **bash** or **zsh** - For running shell scripts

### Check Installation

```bash
python --version
pip --version
bash --version
```

---

## 🚀 Setup & Installation

### 1. Clone or Navigate to the Project

```bash
cd image_search_demo
```

### 2. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
cd ..
```

Or let the `start.sh` script handle this automatically.

### 3. Make Scripts Executable

```bash
chmod +x start.sh stop.sh
```

---

## 🎮 Running the Application

### Quick Start (Recommended)

```bash
./start.sh
```

This single command will:
- ✅ Validate the project structure
- ✅ Install dependencies if needed
- ✅ Start FastAPI backend on port 5000
- ✅ Start frontend HTTP server on port 5001
- ✅ Log all output to `app.log`
- ✅ Display service URLs and PIDs

### Stopping the Application

```bash
./stop.sh
```

Or press `Ctrl+C` in the terminal running `start.sh`.

### Manual Start (For Development)

**Terminal 1 - Backend:**
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 5000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
python -m http.server 5001
```

Then open your browser to `http://localhost:5001`

---

## 📡 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Health Check
- **URL**: `/`
- **Method**: `GET`
- **Response**:
```json
{
  "status": "ok",
  "service": "Image Search Demo API",
  "version": "1.0.0"
}
```

#### 2. Search Endpoint
- **URL**: `/api/search`
- **Method**: `POST`
- **Content-Type**: `application/json`

**Request Payload (Text Search):**
```json
{
  "query_text": "red car"
}
```

**Request Payload (Image Search):**
```json
{
  "query_image": "iVBORw0KGgoAAAANSUhEUgAAAOE..."
}
```

**Response:**
```json
[
  {
    "image_url": "https://picsum.photos/300/200?random=1",
    "caption_id": "IMG_001",
    "caption": "A red sports car on a sunny road"
  },
  {
    "image_url": "https://picsum.photos/300/200?random=2",
    "caption_id": "IMG_002",
    "caption": "Modern cityscape with tall buildings"
  }
]
```

**Error Responses:**

Missing both queries:
```json
{
  "detail": "Must provide either query_text or query_image"
}
```

Both queries provided:
```json
{
  "detail": "Cannot use both text and image queries simultaneously. Provide only one."
}
```

### Interactive API Documentation
Visit `http://localhost:5000/docs` for the interactive Swagger UI to test endpoints.

---

## 🎨 Frontend Features

### Search Interface
- **Text Input**: Enter any search query (e.g., "sunset", "car", "dog")
- **Image Button**: Upload an image or capture from camera
- **Search Button**: Execute the search

### Image Processing
- Images are automatically resized to **224×224 pixels** on the client-side
- Conversion to **Base64 PNG format** for transmission
- **Optimized payload size** for faster uploads

### Results Display
- Grid layout that adapts to screen size
- Each result card shows:
  - Thumbnail image
  - Caption ID
  - Caption text
- Responsive design for mobile and desktop

### User Feedback
- Loading state indicators
- Error messages for network/validation issues
- Preview of selected/captured images
- Success indicators (green button for selected image)

---

## 🧠 Integrating CLIP ONNX Model

The backend currently uses mock data. Here's how to integrate the CLIP ONNX model:

### Step 1: Prepare ONNX Model Files
Ensure you have:
- `vision_tower.onnx` - Vision encoder
- `text_tower.onnx` - Text encoder
- Place them in a `backend/models/` directory

### Step 2: Install ONNX Runtime
```bash
pip install onnxruntime numpy pillow
```

### Step 3: Modify Backend Code

Update `backend/main.py` following the TODO comments:

**Example implementation structure:**

```python
import onnxruntime as rt
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# Load models at startup
vision_session = rt.InferenceSession("models/vision_tower.onnx")
text_session = rt.InferenceSession("models/text_tower.onnx")

# Preload image embeddings from your database
# image_embeddings_db = load_embeddings("path/to/embeddings.npy")

def process_text_search(text: str) -> List[SearchResult]:
    """
    1. Tokenize text
    2. Get text embedding from CLIP text tower
    3. Compare with precomputed image embeddings
    4. Return top-k results
    """
    # text_tokens = tokenizer(text)
    # text_embedding = text_session.run(None, {"input": text_tokens})[0]
    # similarities = compute_similarity(text_embedding, image_embeddings_db)
    # top_k_indices = get_top_k(similarities, k=6)
    # return format_results(top_k_indices)
    pass

def process_image_search(image_base64: str) -> List[SearchResult]:
    """
    1. Decode Base64 image
    2. Preprocess image (224x224, normalize)
    3. Get image embedding from CLIP vision tower
    4. Compare with precomputed embeddings
    5. Return top-k results
    """
    # image_data = base64.b64decode(image_base64)
    # image = Image.open(BytesIO(image_data))
    # image_tensor = preprocess_image(image)
    # image_embedding = vision_session.run(None, {"input": image_tensor})[0]
    # similarities = compute_similarity(image_embedding, image_embeddings_db)
    # top_k_indices = get_top_k(similarities, k=6)
    # return format_results(top_k_indices)
    pass
```

### Step 4: Precompute Image Embeddings
For efficient search, precompute embeddings for all images in your database:

```python
import numpy as np

def precompute_embeddings(image_paths):
    """Generate and save embeddings for all images"""
    embeddings = []
    for path in image_paths:
        image = Image.open(path)
        image_tensor = preprocess_image(image)
        embedding = vision_session.run(None, {"input": image_tensor})[0]
        embeddings.append(embedding)
    
    np.save("image_embeddings.npy", np.array(embeddings))

# Load at startup
image_embeddings_db = np.load("image_embeddings.npy")
```

### Step 5: Similarity Computation
```python
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(query_embedding, db_embeddings):
    """Compute cosine similarity between query and database embeddings"""
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        db_embeddings
    )[0]
    return similarities

def get_top_k(similarities, k=6):
    """Return indices of top-k most similar items"""
    return np.argsort(similarities)[::-1][:k]
```

---

## 🔧 Troubleshooting

### Port Already in Use
**Error**: `Address already in use`

**Solution**:
```bash
# Kill process on port 5000
lsof -i :5000 | grep LISTEN | awk '{print $2}' | xargs kill -9

# Kill process on port 5001
lsof -i :5001 | grep LISTEN | awk '{print $2}' | xargs kill -9
```

### Backend Not Responding
**Error**: `Failed to connect to http://localhost:5000`

**Solution**:
1. Check if backend is running: `lsof -i :5000`
2. Check logs: `tail -f app.log`
3. Ensure dependencies are installed: `pip install -r backend/requirements.txt`

### CORS Issues
If the frontend gets CORS errors, ensure the backend has:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Image Upload Not Working
1. Check browser console (F12 → Console tab)
2. Ensure image file is < 5MB
3. Verify Base64 encoding in network tab (F12 → Network)

---

## 📚 Next Steps

### Phase 1: Model Integration (Priority)
- [ ] Prepare CLIP ONNX models
- [ ] Implement CLIP text encoder
- [ ] Implement CLIP vision encoder
- [ ] Create image embeddings database
- [ ] Implement similarity search

### Phase 2: Database Integration
- [ ] Set up image embedding storage (e.g., ChromaDB, Qdrant)
- [ ] Implement metadata storage
- [ ] Add image retrieval by ID
- [ ] Implement pagination for large result sets

### Phase 3: Performance & Caching
- [ ] Add result caching
- [ ] Optimize image preprocessing
- [ ] Batch processing for multiple queries
- [ ] Add rate limiting

### Phase 4: Production Deployment
- [ ] Add authentication/authorization
- [ ] Implement request validation
- [ ] Add request/response logging
- [ ] Deploy using Docker
- [ ] Set up CI/CD pipeline
- [ ] Add monitoring and alerts

### Phase 5: Enhanced Frontend
- [ ] Add result pagination
- [ ] Implement search history
- [ ] Add advanced filters
- [ ] Support batch uploads
- [ ] Dark mode theme

---

## 📝 Logging

All logs are saved to `app.log` in the root directory.

**View logs in real-time:**
```bash
tail -f app.log
```

**View last 50 lines:**
```bash
tail -50 app.log
```

---

## 📄 License

This project is provided as-is for demonstration and educational purposes.

---

## 🤝 Support

For issues or questions:
1. Check the `Troubleshooting` section
2. Review `app.log` for detailed error messages
3. Check the FastAPI documentation: https://fastapi.tiangolo.com/
4. Check the browser console (F12) for frontend errors

---

**Happy Searching! 🚀**

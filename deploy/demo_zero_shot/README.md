# 📷 Image Caption Demo Application

A lightweight, fast full-stack web application for capturing images from a device camera and sending them to a backend API for processing. Built with **FastAPI** (backend) and **Vanilla JavaScript + TailwindCSS** (frontend).

Perfect for quick prototyping and ML model integration.

---

## 🎯 Features

- ✅ **Live Camera Feed** — Real-time video stream from device camera (rear camera preferred on mobile)
- ✅ **Image Capture** — Capture frames at exactly **224×224 pixels** (JPEG format)
- ✅ **Mobile-First UI** — Responsive design optimized for touch devices
- ✅ **Error Handling** — Graceful fallbacks for camera permissions, network errors, and API issues
- ✅ **CORS Enabled** — Frontend can safely call backend APIs
- ✅ **Zero Build Tools** — Frontend uses CDN-based TailwindCSS, no build step needed
- ✅ **ML-Ready** — Backend placeholder structure ready for ML model integration

---

## 📁 Project Structure

```
deploy/
├── backend/
│   ├── requirements.txt      # Python dependencies
│   └── main.py              # FastAPI application
├── frontend/
│   └── index.html           # Single HTML file (HTML + JS + CSS via CDN)
└── README.md                # This file
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | HTML5 + Vanilla JavaScript | UI and camera interaction |
| **Frontend Styling** | TailwindCSS (CDN) | Responsive, mobile-first styling |
| **Backend** | FastAPI | REST API server |
| **Backend Server** | Uvicorn | ASGI HTTP server |
| **Image Processing** | Python multipart | File upload handling |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Camera access permissions

### Step 1: Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Dependencies installed:**
- `fastapi` — Modern Python web framework
- `uvicorn` — ASGI server for running FastAPI
- `python-multipart` — Handles multipart/form-data (file uploads)

### Step 2: Start the Backend Server

```bash
cd backend
python main.py
```

Or use uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

The backend API will be available at: `http://localhost:8000`

### Step 3: Start the Frontend Server

Open a new terminal and navigate to the `frontend` folder:

```bash
cd frontend
python -m http.server 8001
```

**With Python's built-in server:**
```
Serving HTTP on 0.0.0.0 port 8001 (http://0.0.0.0:8001/) ...
```

### Step 4: Open in Browser

Navigate to: **`http://localhost:8001`**

1. **Allow camera permissions** when prompted
2. **Tap "Capture Frame"** to capture and send an image
3. **View the caption result** returned from the backend

---

## 📡 API Documentation

### Endpoint: `POST /predict`

**Description:** Upload an image and receive a caption (placeholder implementation).

**Request:**
```
POST http://localhost:8000/predict
Content-Type: multipart/form-data

image: <binary JPEG file, 224×224 pixels>
```

**Response (200 OK):**
```json
{
  "caption": "This is a fixed caption from the backend. Model not yet loaded."
}
```

**Error Responses:**
- `400 Bad Request` — No file provided or invalid file format
- `500 Internal Server Error` — Server error during processing

### Endpoint: `GET /`

**Description:** Health check endpoint.

**Response (200 OK):**
```json
{
  "message": "Image Caption API is running. Use POST /predict with an image."
}
```

---

## 🎨 Frontend Features

### User Interface
- **Video Container** — Displays live camera feed (aspect-fill, rounded corners)
- **Capture Button** — Large touch target (44px minimum height)
- **Loading Indicator** — Animated spinner during API request
- **Result Display** — Shows returned caption from backend
- **Error Display** — Shows user-friendly error messages

### Camera Handling
- Requests **rear-facing camera** (via `facingMode: 'environment'`)
- Gracefully handles camera permission denial
- Shows informative error messages for camera access issues

### Image Processing
- Captures frames from video stream
- Resizes to exactly **224×224 pixels**
- Centers content (maintains aspect ratio with black padding if needed)
- Exports as **JPEG** (ensures RGB channels, no alpha)
- Compresses at 95% quality for efficient transfer

### Error Handling

The frontend gracefully handles:
- ❌ **Camera Permission Denied** — Shows error overlay
- ❌ **Video Not Ready** — Waits and retries
- ❌ **Backend Unreachable** — Shows helpful error message
- ❌ **Invalid API Response** — Validates response structure
- ❌ **Network Errors** — Catches and displays errors

---

## 🔗 CORS Configuration

The backend has CORS (Cross-Origin Resource Sharing) enabled, allowing the frontend to make requests from a different origin.

**Current Settings (Development):**
```python
allow_origins=["*"]        # Allow all origins
allow_credentials=True
allow_methods=["*"]        # Allow all HTTP methods
allow_headers=["*"]        # Allow all headers
```

⚠️ **For Production:** Restrict `allow_origins` to your specific domain:
```python
allow_origins=["https://yourdomain.com", "https://www.yourdomain.com"]
```

---

## 🧠 ML Model Integration Guide

The backend is structured to easily integrate an ML model. Follow these steps:

### 1. Add the Model to `main.py`

```python
# At the top of the file, after imports
from PIL import Image
import io
import numpy as np

# Load your model once at startup (global variable)
# This avoids reloading the model on every request
MODEL = load_your_ml_model()  # Your model loading function
```

### 2. Modify the `/predict` Endpoint

```python
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Process the image with the ML model and return a caption."""
    
    try:
        # Read the uploaded image file
        image_data = await image.read()
        
        # Convert bytes to PIL Image
        img = Image.open(io.BytesIO(image_data))
        
        # Ensure RGB format (3 channels)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Preprocess the image (resize, normalize, etc.)
        img_array = preprocess_image(img)  # Your preprocessing function
        
        # Run the ML model
        caption = MODEL.predict(img_array)  # Your inference function
        
        return {"caption": caption}
    
    except Exception as e:
        return {"error": str(e)}, 500
```

### 3. Add Your ML Dependencies

Update `backend/requirements.txt`:
```
fastapi
uvicorn
python-multipart
torch              # If using PyTorch
transformers       # If using Hugging Face models
pillow             # For image processing
numpy              # For numerical operations
```

Then reinstall:
```bash
pip install -r requirements.txt
```

---

## 🐛 Troubleshooting

### Issue: "Cannot reach the server"
**Solution:**
- Ensure backend is running: `python backend/main.py`
- Check that the backend is listening on `http://localhost:8000`
- In frontend console (F12), verify the API_URL is correct

### Issue: "Camera access denied"
**Solution:**
- Check browser permissions (camera icon in URL bar)
- Grant camera access when prompted
- On mobile, ensure the app has camera permissions in device settings
- Try a different browser or device

### Issue: "Video stream not ready"
**Solution:**
- Wait a moment after page loads before capturing
- Check that your camera is plugged in and not in use by another app
- Try refreshing the page

### Issue: CORS errors in browser console
**Solution:**
- Verify both frontend and backend are running
- Check that backend has CORS enabled (it should by default)
- Ensure the frontend is calling `http://localhost:8000` (not `localhost`, not `127.0.0.1`)

### Issue: Image upload fails but no error message
**Solution:**
- Open browser Developer Tools (F12 → Network tab)
- Check the response from the `/predict` request
- Common issues: backend restarted, image size too large, corrupted image data

---

## 📊 API Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     User Browser                             │
├─────────────────────────────────────────────────────────────┤
│ 1. Load http://localhost:8001                                │
│ 2. Request camera access via navigator.mediaDevices         │
│ 3. Display live video feed                                   │
│ 4. User taps "Capture Frame"                                │
│ 5. JavaScript captures frame → 224×224 JPEG                 │
│ 6. POST /predict with image to http://localhost:8000        │
│                    ↓                                          │
├─────────────────────────────────────────────────────────────┤
│                  FastAPI Backend                             │
├─────────────────────────────────────────────────────────────┤
│ 7. Receive multipart/form-data with image file             │
│ 8. (TODO: Process image with ML model)                      │
│ 9. Return JSON: {"caption": "..."}                          │
│                    ↓                                          │
├─────────────────────────────────────────────────────────────┤
│                  User Browser                                │
├─────────────────────────────────────────────────────────────┤
│ 10. Receive JSON response                                    │
│ 11. Display caption in result container                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 Browser Support

| Browser | Desktop | Mobile |
|---------|---------|--------|
| Chrome | ✅ | ✅ |
| Firefox | ✅ | ✅ |
| Safari | ✅ | ✅ (iOS 13+) |
| Edge | ✅ | ✅ |

**Note:** Camera access requires a secure context (HTTPS) in production. HTTP works for `localhost` development.

---

## 📝 Development Notes

### Image Format Specification
- **Dimensions:** 224×224 pixels (hardcoded in frontend)
- **Format:** JPEG (image/jpeg MIME type)
- **Color Space:** RGB (3 channels, no alpha)
- **Quality:** 95% (trades file size for quality)
- **Encoding:** Via HTML5 Canvas `toBlob()` API

### Backend Image Handling
The current implementation receives the image but does not process it. To handle the image:

```python
import io
from PIL import Image

async def predict(image: UploadFile = File(...)):
    image_data = await image.read()
    img = Image.open(io.BytesIO(image_data))
    # Now you can use 'img' with PIL methods or convert to numpy
```

### Frontend Canvas Resizing Strategy
- Captures from live video at native resolution
- Centers the frame on a 224×224 canvas
- Uses black fill for letterboxing (preserves aspect ratio)
- Ensures no distortion of the image

---

## 🔒 Security Considerations

⚠️ **Current Implementation (Development):**
- CORS allows all origins (`*`)
- No authentication/authorization

🛡️ **For Production:**
1. **Restrict CORS:** Limit to your domain only
2. **Add Authentication:** Use JWT tokens or API keys
3. **Rate Limiting:** Prevent abuse with request throttling
4. **Input Validation:** Validate file size and image dimensions
5. **Error Logging:** Log errors securely without exposing sensitive info
6. **HTTPS:** Always use HTTPS for production deployments

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MDN: MediaDevices.getUserMedia()](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)
- [Canvas API: toBlob()](https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/toBlob)
- [TailwindCSS Documentation](https://tailwindcss.com/docs)

---

## 📄 License

This is a demo application. Feel free to use and modify as needed.

---

## ❓ FAQ

**Q: Can I change the image resolution from 224×224?**  
A: Yes, modify the `width` and `height` attributes in the canvas HTML element and the canvas width/height in JavaScript.

**Q: Why JPEG instead of PNG?**  
A: JPEG automatically ensures 3 RGB channels. PNG can have transparency (alpha channel), which may cause issues with ML models expecting exactly 3 channels.

**Q: How do I test the API without the frontend?**  
A: Use `curl` or Postman:
```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@image.jpg"
```

**Q: Can I deploy this to production?**  
A: Yes! Use a proper ASGI server (Gunicorn + Uvicorn) and a reverse proxy (Nginx). Host the frontend on a static hosting service like Vercel, Netlify, or AWS S3 + CloudFront.

**Q: Why does the video look stretched?**  
A: This may be because the canvas dimensions don't match your camera's aspect ratio. The code centers and letterboxes the image to preserve aspect ratio, but visual scaling depends on your device.

---

**Last Updated:** March 2026  
**Status:** Ready for ML Model Integration

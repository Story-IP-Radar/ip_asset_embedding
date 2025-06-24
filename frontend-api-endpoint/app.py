from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BLIP Embedding Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
processor = None
model = None
device = None

class TextInput(BaseModel):
    text: str

@app.on_event("startup")
async def load_model():
    global processor, model, device
    
    logger.info("Loading BLIP model...")
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Enable GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        logger.info(f"Model loaded successfully on device: {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

def get_text_features(text: str):
    """Extract features from text using BLIP's text encoder"""
    try:
        inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            # Get text features from BLIP model
            text_features = model.get_text_features(**inputs)
            # Normalize the embedding
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        return text_features.cpu().numpy()
    except Exception as e:
        logger.error(f"Error extracting text features: {e}")
        raise e

def get_image_features(image: Image.Image):
    """Extract features from image using BLIP's image encoder"""
    try:
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Get image features from BLIP model
            image_features = model.get_image_features(**inputs)
            # Normalize the embedding
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        return image_features.cpu().numpy()
    except Exception as e:
        logger.error(f"Error extracting image features: {e}")
        raise e

def caption_image(image: Image.Image):
    """Generate caption for image using BLIP"""
    try:
        inputs = processor(image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=30)
            caption = processor.decode(output[0], skip_special_tokens=True)
            
        return caption
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        raise e

@app.post("/embed/text")
async def embed_text(input_data: TextInput):
    """Generate embedding for text"""
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features = get_text_features(input_data.text)
        embedding = features.squeeze().tolist()
        
        return {
            "embedding": embedding,
            "dimension": len(embedding)
        }
    
    except Exception as e:
        logger.error(f"Text embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/image")
async def embed_image(file: UploadFile = File(...)):
    """Generate embedding for image"""
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        features = get_image_features(image)
        embedding = features.squeeze().tolist()
        
        return {
            "embedding": embedding,
            "dimension": len(embedding)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/caption/image")
async def caption_image_endpoint(file: UploadFile = File(...)):
    """Generate caption for image"""
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        caption = caption_image(image)
        
        return {
            "caption": caption
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image captioning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model and processor else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "device": device if device else "unknown"
    }

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "BLIP Embedding Service",
        "version": "1.0.0",
        "endpoints": {
            "/embed/text": "POST - Generate text embeddings",
            "/embed/image": "POST - Generate image embeddings", 
            "/caption/image": "POST - Generate image captions",
            "/health": "GET - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
import uvicorn

app = FastAPI(title="BLIP Embedding Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load BLIP model once at startup
print("Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Enable GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print(f"BLIP model loaded on {device}")

class TextQuery(BaseModel):
    text: str

def get_image_embedding(image: Image.Image) -> list:
    """Extract embedding from image using BLIP's vision encoder"""
    try:
        # Process image
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Get image features from vision encoder
        with torch.no_grad():
            # Use BLIP's vision model to get image embeddings
            vision_outputs = model.vision_model(
                pixel_values=inputs.pixel_values,
                output_hidden_states=True
            )
            
            # Use the pooled output or last hidden state
            # You can experiment with different layers
            image_embedding = vision_outputs.pooler_output
            
            # Normalize the embedding
            image_embedding = torch.nn.functional.normalize(image_embedding, p=2, dim=1)
            
        return image_embedding.cpu().numpy().flatten().tolist()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

def get_text_embedding(text: str) -> list:
    """Extract embedding from text using BLIP's text encoder"""
    try:
        # Process text
        inputs = processor(text=text, return_tensors="pt").to(device)
        
        # Get text features from text encoder
        with torch.no_grad():
            # Use BLIP's text encoder to get text embeddings
            text_outputs = model.text_encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )
            
            # Use pooled output (CLS token)
            text_embedding = text_outputs.pooler_output
            
            # Normalize the embedding
            text_embedding = torch.nn.functional.normalize(text_embedding, p=2, dim=1)
            
        return text_embedding.cpu().numpy().flatten().tolist()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

def get_multimodal_embedding(image: Image.Image, text: str = "") -> list:
    """Get multimodal embedding using both image and text"""
    try:
        # Process both image and text
        if text:
            inputs = processor(images=image, text=text, return_tensors="pt", padding=True).to(device)
        else:
            inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Get multimodal features
            outputs = model.get_image_features(pixel_values=inputs.pixel_values)
            
            # Normalize the embedding
            embedding = torch.nn.functional.normalize(outputs, p=2, dim=1)
            
        return embedding.cpu().numpy().flatten().tolist()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing multimodal input: {str(e)}")

@app.post("/embed/image")
async def embed_image(file: UploadFile = File(...)):
    """Generate embedding for uploaded image"""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Get embedding
        embedding = get_image_embedding(image)
        
        return {
            "embedding": embedding,
            "dimension": len(embedding),
            "model": "blip-image-captioning-base"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/text")
async def embed_text(query: TextQuery):
    """Generate embedding for text"""
    try:
        embedding = get_text_embedding(query.text)
        
        return {
            "embedding": embedding,
            "dimension": len(embedding),
            "model": "blip-image-captioning-base",
            "text": query.text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/multimodal")
async def embed_multimodal(file: UploadFile = File(...), text: str = ""):
    """Generate multimodal embedding for image + text"""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Get multimodal embedding
        embedding = get_multimodal_embedding(image, text)
        
        return {
            "embedding": embedding,
            "dimension": len(embedding),
            "model": "blip-image-captioning-base",
            "text": text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "blip-image-captioning-base",
        "device": device
    }

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "BLIP Embedding Service",
        "model": "Salesforce/blip-image-captioning-base",
        "endpoints": {
            "image": "/embed/image",
            "text": "/embed/text", 
            "multimodal": "/embed/multimodal",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
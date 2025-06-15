import json
from tqdm import tqdm
from PIL import Image
import requests
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

# Load BLIP
print("Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Enable GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Input and output paths
input_path = "assets.ndjson"
output_path = "captioned_assets.ndjson"

# Load already processed record IDs + cached captions
processed_ids = set()
image_cache = {}

if os.path.exists(output_path):
    with open(output_path, 'r') as existing_file:
        for line in existing_file:
            try:
                record = json.loads(line)
                processed_ids.add(record.get("id"))
                image_url = record.get("nftMetadata", {}).get("imageUrl")
                if image_url and "imageCaption" in record:
                    image_cache[image_url] = record["imageCaption"]
            except:
                continue

print(f"Resuming from {len(processed_ids)} already processed records.")
print(f"Loaded {len(image_cache)} cached image captions.")

def caption_image(url):
    if url in image_cache:
        return image_cache[url]
    try:
        response = requests.get(url, stream=True, timeout=10)
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            return f"ERROR: Non-image content type ({content_type})"
        image = Image.open(response.raw).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(output[0], skip_special_tokens=True)
        image_cache[url] = caption
        return caption
    except Exception as e:
        return f"ERROR: {e}"

with open(input_path, 'r') as infile, open(output_path, 'a') as outfile:
    for line in tqdm(infile, desc="Processing images"):
        try:
            record = json.loads(line)
            if record["id"] in processed_ids:
                continue  # Already done

            image_url = record.get("nftMetadata", {}).get("imageUrl")
            if not image_url:
                continue  # Skip entries with no image

            caption = caption_image(image_url)
            if caption.startswith("ERROR:"):
                print(f"[SKIP] {record['id']} -> {caption}")
                continue  # Don't write invalid results

            record["imageCaption"] = caption
            outfile.write(json.dumps(record) + "\n")
            outfile.flush()
        except Exception as e:
            print(f"[ERROR] Failed to process record: {e}")

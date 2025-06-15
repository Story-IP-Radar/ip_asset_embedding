import json

input_path = "assets.ndjson"
captioned_path = "captioned_assets.ndjson"
output_path = "full_assets_with_captions.ndjson"

# Step 1: Load captioned data into a dictionary by ID
captions = {}

with open(captioned_path, 'r') as f:
    for line in f:
        record = json.loads(line)
        record_id = record.get("id")
        caption = record.get("imageCaption", "N/A")
        if record_id:
            captions[record_id] = caption

# Step 2: Merge with original dataset
with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
    for line in infile:
        record = json.loads(line)
        record_id = record.get("id")
        if record_id in captions:
            record["imageCaption"] = captions[record_id]
        else:
            record["imageCaption"] = "N/A"
        outfile.write(json.dumps(record) + "\n")

print(f"Merged output written to: {output_path}")

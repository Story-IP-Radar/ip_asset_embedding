import json

def filter_entries_with_image(input_filename, output_filename):
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                image_url = data.get("nftMetadata", {}).get("imageUrl", "").strip()
                if image_url:
                    outfile.write(json.dumps(data) + '\n')
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line in {input_filename}")

# Process both files
filter_entries_with_image('assets-owner.ndjson', 'assets-owner-reduced.ndjson')
filter_entries_with_image('assets.ndjson', 'assets-reduced.ndjson')

import json

def convert_ndjson_to_jsonl():
    input_file = 'assets.ndjson'
    output_file = 'assets.jsonl'
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            with open(output_file, 'w', encoding='utf-8') as outfile:
                line_count = 0
                for line in infile:
                    line = line.strip()
                    if line:  # Skip empty lines
                        # Parse and reformat to ensure valid JSON
                        json_obj = json.loads(line)
                        outfile.write(json.dumps(json_obj) + '\n')
                        line_count += 1
                
                print(f"Successfully converted {input_file} to {output_file}")
                print(f"Processed {line_count} records")
                
    except Exception as error:
        print(f"Error: {error}")

if __name__ == "__main__":
    convert_ndjson_to_jsonl()
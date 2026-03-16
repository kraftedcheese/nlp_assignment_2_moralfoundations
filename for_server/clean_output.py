import json
import csv
import os
import argparse

# Configuration
TARGET_LABELS = ["Authority", "Purity", "Care", "Thin Morality", "Non-Moral"]

LABEL_MAP = {
    "Authority": ["authority", "hierachy", "submissiveness", "authority/hierarchy"],
    "Purity": ["purity", "sanctity", "degradation", "purity/sanctity"],
    "Care": ["care", "harm", "care/harm", "compassion"],
    "Thin Morality": ["thin morality", "thin", "general morality"],
    "Non-Moral": ["non-moral", "non moral", "none", "neutral", "nm"]
}

def normalize_and_match(raw_label_str, target_name):
    """
    Checks if any variation of the target_name exists in the raw_label_str.
    """
    if not raw_label_str:
        return 0
    
    clean_input = str(raw_label_str).lower()
    variations = LABEL_MAP.get(target_name, [target_name.lower()])
    
    for v in variations:
        if v in clean_input:
            return 1
    return 0

def process_batches(input_dir, output_csv, error_log='malformed_outputs.txt'):
    headers = ['ID', 'text'] + TARGET_LABELS + ['reason', 'confidence']
    
    # Get all json files in the directory and sort them
    json_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} batches. Processing...")

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile, \
         open(error_log, 'w', encoding='utf-8') as errfile:
        
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for filename in json_files:
            file_path = os.path.join(input_dir, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle single object or list of objects
                if isinstance(data, dict):
                    data = [data]

                for item in data:
                    try:
                        row = {
                            'ID': item.get('ID'),
                            'text': item.get('text'),
                            'reason': item.get('reason'),
                            'confidence': item.get('confidence')
                        }

                        raw_labels_field = item.get('labels', '')
                        
                        for label in TARGET_LABELS:
                            row[label] = normalize_and_match(raw_labels_field, label)

                        # Basic validation
                        if row['ID'] is None or row['text'] is None:
                            raise ValueError("Missing ID or Text")

                        writer.writerow(row)

                    except Exception as e:
                        errfile.write(f"FILE: {filename} | ID: {item.get('ID')} | ERR: {str(e)} | DATA: {item}\n")

            except Exception as e:
                errfile.write(f"FILE: {filename} | ERR: {str(e)}\n")

    print(f"Done! Cleaned data is in {output_csv}.")

def main():
    parser = argparse.ArgumentParser(description="Merge JSON batch files into a single normalized CSV.")
    
    parser.add_argument("--input_dir", default="output_batches", 
                        help="Directory containing the JSON batch files (default: output_batches)")
    parser.add_argument("--output_file", default="processed_data_robust.csv", 
                        help="Path for the final CSV output (default: processed_data_robust.csv)")
    parser.add_argument("--error_log", default="malformed_outputs.txt", 
                        help="Path for the error log (default: malformed_outputs.txt)")

    args = parser.parse_args()

    # Ensure input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    process_batches(args.input_dir, args.output_file, args.error_log)

if __name__ == "__main__":
    main()
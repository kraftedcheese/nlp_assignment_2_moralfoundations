import json
import csv
import os

# Configuration
input_dir = 'output_batches' 
output_csv = 'processed_data_robust.csv'
error_log = 'malformed_outputs.txt'
file_count = 175

target_labels = ["Authority", "Purity", "Care", "Thin Morality", "Non-Moral"]

label_map = {
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
    
    variations = label_map.get(target_name, [target_name.lower()])
    
    for v in variations:
        if v in clean_input:
            return 1
    return 0

def process_batches():
    headers = ['ID', 'text'] + target_labels + ['reason', 'confidence']
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile, \
         open(error_log, 'w', encoding='utf-8') as errfile:
        
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for i in range(file_count):
            filename = f'batch_{i}.json'
            file_path = os.path.join(input_dir, filename)

            if not os.path.exists(file_path):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
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
                        
                        for label in target_labels:
                            row[label] = normalize_and_match(raw_labels_field, label)

                        if row['ID'] is None or row['text'] is None:
                            raise ValueError("Missing ID or Text")

                        writer.writerow(row)

                    except Exception as e:
                        errfile.write(f"FILE: {filename} | ID: {item.get('ID')} | ERR: {str(e)} | DATA: {item}\n")

            except Exception as e:
                errfile.write(f"FILE: {filename} | ERR: {str(e)}\n")

    print(f"Done! Cleaned data is in {output_csv}.")

if __name__ == "__main__":
    process_batches()
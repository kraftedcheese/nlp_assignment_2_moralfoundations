import argparse
import re
import json
import os
import numpy as np
import pandas as pd
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm

def get_content(input_val):
    if input_val and Path(input_val).is_file():
        return Path(input_val).read_text(encoding="utf-8").strip()
    return input_val

def extract_json(raw_output):
    """Cleanly extracts JSON from LLM markdown or raw text."""
    try:
        clean_output = re.sub(r'```json\n?|```', '', raw_output).strip()
        return json.loads(clean_output)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}

def classify_text(client, data_input, user_prompt, system_prompt, temperature, model, use_json):
    try:
        if "{TEXT}" in user_prompt:
            full_user_content = user_prompt.replace("{TEXT}", data_input)
        else:
            full_user_content = f"{user_prompt}\n\nINPUT DATA:\n{data_input}" if data_input else user_prompt

        completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_user_content},
            ],
            response_format={"type": "json_object"} if use_json else None
        )

        response = completion.choices[0].message.content.strip()

        if not use_json:
            return response

        return extract_json(response)
    except Exception as e:
        print(f"  Error during API call: {e}")
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Batch process CSV/TSV with LLM annotation")

    # Required Arguments
    parser.add_argument("--input_file", required=True, help="Path to input CSV/TSV")
    parser.add_argument("--text_column", required=True, help="Column name to process")
    parser.add_argument("--user_prompt", required=True, help="User prompt or path to .txt")

    # Optional Defaults
    parser.add_argument("--output_dir", default="output_batches", help="Directory to save batch results") # Added this
    parser.add_argument("--system_prompt", default="You are a helpful assistant.", help="System prompt or path to .txt")
    parser.add_argument("--use_json", action="store_true", help="Whether the output of the LLM will be structured JSON")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="vLLM model name")
    parser.add_argument("--temperature", type=float, default=0.1) 
    parser.add_argument("--sep", default="\t", help="CSV separator (default: tab)")

    args = parser.parse_args()

    # 1. Setup Client
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8080/v1")

    # 2. Resolve Prompts
    user_p = get_content(args.user_prompt)
    system_p = get_content(args.system_prompt)

    # 3. Load Data
    print(f"Loading {args.input_file}...")
    df = pd.read_csv(args.input_file, sep=args.sep)

    print(f"Processing {len(df)} rows. JSON mode: {args.use_json}")

    batch_size = 100
    # Use the argument for output directory and create it
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Processing {len(df)} rows in batches of {batch_size}...")

    for batch_idx, group in df.groupby(np.arange(len(df)) // batch_size):
        # Update file path to use args.output_dir
        batch_filename = os.path.join(args.output_dir, f"batch_{batch_idx}.json")

        # skip batch if exists
        if os.path.exists(batch_filename):
            print(f"Skipping Batch {batch_idx} (already finished).")
            continue
            
        annotations = []
        
        for idx, row in group.iterrows():
            text = str(row[args.text_column])
            result = classify_text(client, text, user_p, system_p, args.temperature, args.model, args.use_json)
            
            if args.use_json:
                if isinstance(result.get("labels"), list):
                    result["labels"] = ", ".join(result["labels"])
                annotations.append(result)
            else:
                annotations.append(result)
                
        batch_df = group.copy()
        if args.use_json:
            res_df = pd.DataFrame(annotations)
            batch_df = pd.concat([batch_df.reset_index(drop=True), res_df], axis=1)
        else:
            batch_df["llm_annotation"] = annotations
        
        batch_df.to_json(batch_filename, orient="records", indent=4, force_ascii=False)
        
        print(f"Saved batch {batch_idx} to {batch_filename}")


if __name__ == "__main__":
    main()
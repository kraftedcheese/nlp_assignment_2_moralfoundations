import argparse
import re
import json
import pandas as pd
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm  # Added for a nice progress bar

def get_content(input_val):
    """Helper to return file content if path exists, else return raw string."""
    if input_val and Path(input_val).exists() and Path(input_val).is_file():
        return Path(input_val).read_text(encoding="utf-8").strip()
    return input_val

def extract_json(raw_output):
    """Cleanly extracts JSON from LLM markdown or raw text."""
    try:
        # Remove markdown code blocks
        clean_output = re.sub(r'```json\s*|```', '', raw_output).strip()
        return json.loads(clean_output)
    except json.JSONDecodeError:
        # Fallback: Regex to find the first '{' and last '}'
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"raw_output": raw_output, "error": "json_decode_failed"}

def classify_text(client, data_input, user_prompt, system_prompt, temperature, model, use_json):
    try:
        full_user_content = f"{user_prompt}\n\nINPUT DATA:\n{data_input}" if data_input else user_prompt

        completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_user_content},
            ]
        )

        response = completion.choices[0].message.content.strip()
        
        if not use_json:
            return response
        return extract_json(response)
    except Exception as e:
        print(f"  Error during API call: {e}")
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Batch process CSV/TSV with LM Studio annotation")

    parser.add_argument("--input_file", required=True, help="Path to input CSV/TSV")
    parser.add_argument("--output_file", required=True, help="Path to save results")
    parser.add_argument("--text_column", required=True, help="Column name to process")
    parser.add_argument("--user_prompt", required=True, help="User prompt or path to .txt")
    parser.add_argument("--system_prompt", default="You are a helpful assistant.", help="System prompt or path to .txt")
    parser.add_argument("--use_json", action="store_true", help="Expect structured JSON output")
    parser.add_argument("--model", default="qwen/qwen-3.5-9b", help="Model name")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--sep", default=",", help="CSV separator (default: comma)")

    args = parser.parse_args()

    # 1. Setup Client for LM Studio
    client = OpenAI(api_key="lm-studio", base_url="http://localhost:1234/v1")

    # 2. Resolve Prompts
    user_p = get_content(args.user_prompt)
    system_p = get_content(args.system_prompt)

    # 3. Load Data
    print(f"Loading {args.input_file}...")
    # Using engine='python' helps with various encoding/separator edge cases
    df = pd.read_csv(args.input_file, sep=args.sep)
    
    # 4. Process Rows
    annotations = []
    print(f"Starting inference on {len(df)} rows...")
    
    # Added tqdm for better visibility during long runs
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row[args.text_column])
        
        result = classify_text(client, text, user_p, system_p, args.temperature, args.model, args.use_json)
        
        if args.use_json:
            annotations.append(json.dumps(result, ensure_ascii=False))
        else:
            annotations.append(result)

    # 5. Save Results
    df["llm_annotation"] = annotations
    df.to_csv(args.output_file, sep=args.sep, index=False)
    print(f"Successfully saved results to {args.output_file}")

if __name__ == "__main__":
    main()
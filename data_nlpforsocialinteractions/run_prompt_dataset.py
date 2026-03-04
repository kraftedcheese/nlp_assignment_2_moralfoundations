import argparse
import re
import json
import pandas as pd
from openai import OpenAI
from pathlib import Path


# Helper to handle file vs string inputs
def get_content(input_val):
    if input_val and Path(input_val).is_file():
        return Path(input_val).read_text(encoding="utf-8").strip()
    return input_val


def extract_json(raw_output):
    """Cleanly extracts JSON from LLM markdown or raw text."""
    try:
        # Remove markdown code blocks if present
        clean_output = re.sub(r'```json\n?|```', '', raw_output).strip()
        return json.loads(clean_output)
    except json.JSONDecodeError:
        # Fallback: Regex to find the first '{' and last '}'
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def classify_text(client, data_input, user_prompt, system_prompt, temperature, model, use_json):
    try:
        # Combine instructions with specific data
        full_user_content = f"{user_prompt}\n\nINPUT DATA:\n{data_input}" if data_input else user_prompt

        completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_user_content},
            ]
        )

        response = completion.choices[0].message.content
        if not use_json:
            return {response}
        return extract_json(response)
    except Exception as e:
        print(f"  Error during API call: {e}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Batch process CSV/TSV with LLM annotation")

    # Required Arguments
    parser.add_argument("--input_file", required=True, help="Path to input CSV/TSV")
    parser.add_argument("--output_file", required=True, help="Path to save results")
    parser.add_argument("--text_column", required=True, help="Column name to process")
    parser.add_argument("--user_prompt", required=True, help="User prompt or path to .txt")

    # Optional Defaults
    parser.add_argument("--system_prompt", default="You are a helpful assistant.", help="System prompt or path to .txt")
    parser.add_argument("--use_json", action="store_true", help="Whether the output of the LLM will be structured JSON")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="vLLM model name")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sep", default="\t", help="CSV separator (default: tab)")

    args = parser.parse_args()

    # 1. Setup Client
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8080/v1")

    # 2. Resolve Prompts (File or String)
    user_p = get_content(args.user_prompt)
    system_p = get_content(args.system_prompt)
    print("Using User Prompt:\n", user_p)
    print("Using System Prompt:\n", system_p)

    # 3. Load Data
    print(f"Loading {args.input_file}...")
    df = pd.read_csv(args.input_file, sep=args.sep)
    df = df.head(100)
    print("output structure will be JSON:", args.use_json)
    # 4. Process Rows
    annotations = []
    for idx, row in df.iterrows():
        print(f"Processing row {idx + 1}/{len(df)}...")
        text = str(row[args.text_column])

        result = classify_text(client, text, user_p, system_p, args.temperature, args.model, args.use_json)
        if not args.use_json:
            annotations.append(result)
        # print("  Result:", result)
        else:
            annotations.append(json.dumps(result, ensure_ascii=False))

    # 5. Save Results
    df["llm_annotation"] = annotations
    df.to_csv(args.output_file, sep="\t", index=False)
    print(f"Done! Saved to {args.output_file}")


if __name__ == "__main__":
    main()

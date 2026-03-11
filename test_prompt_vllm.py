import argparse
from pathlib import Path
from openai import OpenAI

def get_content(input_val):
    """Helper to return file content if path exists, else return raw string."""
    if input_val and Path(input_val).exists() and Path(input_val).is_file():
        return Path(input_val).read_text(encoding="utf-8").strip()
    return input_val

def main():
    parser = argparse.ArgumentParser(description="Run LLM inference via LM Studio / OpenAI API")

    parser.add_argument("--system", type=str, 
                        default="You are a helpful assistant.",
                        help="System prompt string or path to .txt")
    parser.add_argument("--user", type=str, required=True,
                        help="User prompt template string or path to .txt")
    parser.add_argument("--input", type=str, default="", 
                        help="The data/text to process (string or path to .txt)")
    parser.add_argument("--model", type=str, default="qwen3.5-9b-uncensored-hauhaucs-aggressive",
                        help="Model name (Note: LM Studio uses the currently loaded model regardless of this string)")
    parser.add_argument("--temp", type=float, default=0.6,
                        help="Temperature: higher for randomness, lower for determinism")

    args = parser.parse_args()

    # Resolve content from strings or files
    system_content = get_content(args.system)
    user_template = get_content(args.user)
    data_input = get_content(args.input)

    # Combine user prompt and data input if both exist
    if data_input:
        full_user_content = f"{user_template}\n\n### INPUT DATA:\n{data_input}"
    else:
        full_user_content = user_template

    # Initialize Client for LM Studio
    # LM Studio default port is 1234. api_key is required by the library but ignored by local server.
    client = OpenAI(
        api_key="lm-studio",
        base_url="http://localhost:1234/v1",
    )

    print(f"--- Sending Request to Local Model ({args.model}) ---")

    try:
        chat_response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": full_user_content},
            ],
            temperature=args.temp,
        )

        content = chat_response.choices[0].message.content
        print("\n--- Response ---\n")
        print(content)
        
    except Exception as e:
        print(f"\nERROR: Could not connect to LM Studio. \nEnsure the Local Server is running at http://localhost:1234\nDetails: {e}")

if __name__ == "__main__":
    main()
import argparse
from pathlib import Path
from openai import OpenAI


def get_content(input_val):
    """Helper to return file content if path exists, else return raw string."""
    if input_val and Path(input_val).is_file():
        return Path(input_val).read_text(encoding="utf-8").strip()
    return input_val


def main():
    parser = argparse.ArgumentParser(description="Run LLM inference via vLLM/OpenAI API")

    parser.add_argument("--system", type=str, help="System prompt string or path to .txt",
                        default="You are a helpful assistant.")
    parser.add_argument("--user", type=str, help="User prompt template string or path to .txt", required=True)
    parser.add_argument("--input", type=str, help="The data/text to process (string or path to .txt)", default="")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--temp", type=float, default=0.6,
                        help="for more randomness in output, increase temp, for more deterministic lower it")

    args = parser.parse_args()

    # Resolve content from strings or files
    system_content = get_content(args.system)
    user_template = get_content(args.user)
    data_input = get_content(args.input)

    # Combine user prompt and data input if both exist
    full_user_content = f"{user_template}\n\nINPUT DATA:\n{data_input}" if data_input else user_template

    # Initialize Client
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8080/v1",
    )

    print(f"--- Sending Request to {args.model} ---")

    chat_response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": full_user_content},
        ],
        temperature=args.temp,
    )

    content = chat_response.choices[0].message.content
    print("\nResponse:\n", content)


if __name__ == "__main__":
    main()

#! Python 3.8.10
import requests
import json
import argparse

'''
This program is for accessing google colab ollam server using ngrok exposed public url
'''

# OLLAMA API DOCS: https://github.com/ollama/ollama/blob/main/docs/api.md
# Run on colab refrence: https://github.com/marcogreiveldinger/videos/blob/main/ollama-ai/run-on-colab/ollama-ai-colab.ipynb

# Example URL to make a GET request
NGROK_EXPOSED_URL = "<NGROK_EXPOSED_URL>"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
LOCAL_LLM = "mistral:instruct"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Define command-line arguments with default values
    parser.add_argument(
        "--model", type=str, default="mistral:instruct", help="model name eg. mistral"
    )
    parser.add_argument("--url", type=str, help="url of exposed api")
    parser.add_argument(
        "--forever", action="store_true", help="run only one time or in loop"
    )
    parser.add_argument(
        "--stream", action="store_false", help="output from url will be streamed"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Convert parsed arguments into a dictionary
    args_dict = vars(args)

    # Filter out arguments with default values
    args_dict = {k: v for k, v in args_dict.items() if v is not None}

    return args_dict


def ollama_generate_from_url(
    user_query: str, url: str = OLLAMA_URL, model=LOCAL_LLM, stream=True
):
    print("*" * 30 + " Configs " + "*" * 30)
    print(f"{user_query = }")
    print(f"{url = }")
    print(f"{model = }")
    print(f"{stream = }")
    print("*" * 30 + "  " + "*" * 30)

    payload = {
        "model": model,  # Your model name
        "prompt": user_query,  # prompt for the model
    }

    try:
        # Making a GET request with stream=True
        response = requests.post(url, json=payload, stream=stream)

        # Iterating over the response content line by line
        for line in response.iter_lines():
            if line:
                # Process each line of the streaming response
                response_dict = json.loads(line.decode("utf-8"))
                print(response_dict["response"], end="")  # Decode bytes to string
    except requests.RequestException as e:
        print("Error occurred:", e)


def start_ollama_server(
    model: str = LOCAL_LLM, forever=False, url: str = OLLAMA_URL, stream=True
):

    user_query = input(f"Ask ollama: ")
    ollama_generate_from_url(model=model, url=url, user_query=user_query, stream=stream)

    while forever:
        user_query = input(f"Ask ollama: ")
        ollama_generate_from_url(
            model=model, url=url, user_query=user_query, stream=stream
        )


def main():
    args = parse_arguments()
    start_ollama_server(**args)


if __name__ == "__main__":
    main()

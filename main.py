from gpt4all import GPT4All
from pathlib import Path
import time

def main():
    model_path = Path(__file__).parent / "ggml-nomic-ai-gpt4all-falcon-Q4_1.gguf"
    model = GPT4All(str(model_path), allow_download=False)

    print("чат")

    with model.chat_session() as session:
        while True:
            user_input = input("я: ")
            if user_input.strip().lower() in ["exit", "quit", "выход"]:
                print("завершение работы")
                break

            start_time = time.time()
            response = model.generate(
                user_input,
                max_tokens=100,
                temp=0.5,
            )
            end_time = time.time()

            elapsed = end_time - start_time
            length = len(response.split())

            print("модель:", response)
            print(f"время генерации: {elapsed:.2f} сек")
            print(f"длина ответа: {length} слов\n")


if __name__ == "__main__":
    main()

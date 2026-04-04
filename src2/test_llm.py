from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent.parent / ".env")

from llm import LLMClient

client = LLMClient()

response = client.complete(
    messages=[{"role": "user", "content": "What is Stardew Valley?"}],
    system="You are a helpful Stardew Valley assistant."
)

print("Answer:", response.answer)
print("Reasoning:", response.reasoning)
print("Tokens:", response.total_tokens)
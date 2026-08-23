import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

groq_key = os.getenv("GROQ_KEY_1")

if not groq_key:
    raise ValueError("GROQ_KEY_1 is missing in .env")

response = completion(
    model="groq/openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "Give one short marketing suggestion for a customer segment."
        }
    ],
    api_key=groq_key
)

print("\nLLM Response:")
print(response.choices[0].message.content)
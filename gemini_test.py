import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

gemini_key = os.getenv("GEMINI_KEY_1")

if not gemini_key:
    raise ValueError("GEMINI_KEY_1 is missing in .env")

response = completion(
    model="gemini/gemini-2.5-flash",
    messages=[
        {
            "role": "user",
            "content": "Give one short marketing suggestion for a customer segment."
        }
    ],
    api_key=gemini_key
)

print("\nGemini LLM Response:")
print(response.choices[0].message.content)
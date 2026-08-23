import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

GROQ_KEY = os.getenv("GROQ_KEY_1")
GEMINI_KEY = os.getenv("GEMINI_KEY_1")


def generate_marketing_insight(prompt):

    # Try Groq first
    try:
        response = completion(
            model="groq/openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            api_key=GROQ_KEY
        )

        return response.choices[0].message.content

    except Exception as groq_error:
        print("Groq failed. Trying Gemini...")

        # Gemini fallback
        try:
            response = completion(
                model="gemini/gemini-2.5-flash",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                api_key=GEMINI_KEY
            )

            return response.choices[0].message.content

        except Exception as gemini_error:
            return "Both Groq and Gemini failed."


if __name__ == "__main__":

    prompt = """
    Give a short personalized marketing recommendation
    for a high-value customer segment.
    """

    result = generate_marketing_insight(prompt)

    print("\nMarketing Insight:")
    print(result)
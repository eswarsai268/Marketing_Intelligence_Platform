import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


# --------------------------------------------------
# LLM FALLBACK ROUTER
# --------------------------------------------------

def _call_llm_with_fallback(messages):

    groq_models = [
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-20b"
    ]

    groq_keys = [
        os.getenv("GROQ_KEY_1"),
        os.getenv("GROQ_KEY_2"),
        os.getenv("GROQ_KEY_3"),
        os.getenv("GROQ_KEY_4")
    ]

    # Try Groq keys
    for key in groq_keys:

        if not key:
            continue

        for model in groq_models:

            try:
                client = Groq(api_key=key)

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7
                )

                return response.choices[0].message.content

            except Exception as e:
                print(f"Groq failed with {model}: {e}")
                continue

    # If all Groq keys fail
    raise RuntimeError("All configured LLM providers failed.")


# --------------------------------------------------
# CHAT HISTORY PRUNER
# --------------------------------------------------

def prune_chat_history(chat_history, max_messages=6):

    if len(chat_history) <= max_messages:
        return chat_history

    system_message = chat_history[0]
    remaining = chat_history[1:]

    # Keep the most recent messages
    remaining = remaining[-(max_messages - 1):]

    return [system_message] + remaining


# --------------------------------------------------
# CAMPAIGN GENERATOR
# --------------------------------------------------

def generate_campaign(cluster_name: str, top_category: str):

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert e-commerce marketing copywriter. "
                "Create concise, personalized and engaging marketing campaigns."
            )
        },
        {
            "role": "user",
            "content": (
                f"Write a short, engaging marketing email for a customer "
                f"in the {cluster_name} segment who mostly buys "
                f"{top_category} products. Include a subject line."
            )
        }
    ]

    generated_text = _call_llm_with_fallback(messages)

    messages.append(
        {
            "role": "assistant",
            "content": generated_text
        }
    )

    return generated_text, messages


# --------------------------------------------------
# CHAT REFINEMENT
# --------------------------------------------------

def chat_refinement(chat_history: list, new_user_prompt: str):

    chat_history.append(
        {
            "role": "user",
            "content": new_user_prompt
        }
    )

    # Prune before sending to LLM
    chat_history = prune_chat_history(chat_history)

    generated_text = _call_llm_with_fallback(chat_history)

    chat_history.append(
        {
            "role": "assistant",
            "content": generated_text
        }
    )

    return generated_text, chat_history

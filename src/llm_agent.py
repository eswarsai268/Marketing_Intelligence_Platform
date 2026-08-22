import os
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai

load_dotenv()

# --------------------------------------------------
# LLM FALLBACK ROUTER (The "Traffic-Aware" Engine)
# --------------------------------------------------

def _call_llm_with_fallback(messages):
    """
    Tries Groq models first. If a model hits a traffic error, it switches models.
    If Groq fails entirely, it repeats the exact same logic for Gemini.
    """
    
    # 1. Configuration
    groq_models = [
        "llama3-70b-8192",       # Primary Groq
        "mixtral-8x7b-32768",    # Secondary Groq
        "gemma2-9b-it"           # Tertiary Groq
    ]
    groq_keys = [os.getenv(f"GROQ_KEY_{i}") for i in range(1, 5)]

    gemini_models = [
        "gemini-3.7-flash",      # Primary Gemini (Blazing fast)
        "gemini-3.5-flash-lite",   # Secondary Gemini (Heavy reasoning)
        "gemini-2.5-flash"         # Tertiary Gemini
    ]
    gemini_keys = [os.getenv(f"GEMINI_KEY_{i}") for i in range(1, 5)]

    # --- PHASE 1: GEMINI EXECUTION ---
    # Gemini requires us to separate the system prompt from the user/assistant messages
    gemini_messages = []
    system_instruction = None
    
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        else:
            # Map OpenAI's 'assistant' to Gemini's 'model'
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_messages.append({"role": role, "parts": [msg["content"]]})

    for model_name in gemini_models:
        for key in gemini_keys:
            if not key:
                continue
            try:
                genai.configure(api_key=key)
                llm = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction
                )
                response = llm.generate_content(gemini_messages)
                return response.text
                
            except Exception as e:
                error_msg = str(e).lower()
                # TRAFFIC LOGIC FOR GEMINI
                if "429" in error_msg or "quota" in error_msg or "rate limit" in error_msg:
                    print(f"Gemini Traffic Error on {model_name}. Switching model entirely.")
                    break
                
                print(f"Gemini Auth/Connection Error on key. Trying next key.")
                continue

    # --- PHASE 2: GROQ EXECUTION ---
    for model in groq_models:
        for key in groq_keys:
            if not key:
                continue
            try:
                client = Groq(api_key=key)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=800
                )
                return response.choices[0].message.content
                
            except Exception as e:
                error_msg = str(e).lower()
                # TRAFFIC LOGIC: If Rate Limit/429, kill the key loop and switch model entirely.
                if "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
                    print(f"Groq Traffic Error on {model}. Switching model entirely.")
                    break # Breaks out of the key loop, continuing to the next model
                
                # Otherwise, it's an auth/connection error. Try the next key.
                print(f"Groq Auth/Connection Error on key. Trying next key.")
                continue

    # --- PHASE 3: TOTAL FAILURE ---
    raise RuntimeError("CRITICAL ERROR: All Groq and Gemini models/keys exhausted.")


# --------------------------------------------------
# CHAT HISTORY PRUNER
# --------------------------------------------------

def prune_chat_history(chat_history, max_messages=8):
    """Keeps the system prompt but prunes old messages to save tokens."""
    if len(chat_history) <= max_messages:
        return chat_history
        
    system_message = chat_history[0]
    remaining = chat_history[1:]
    
    # Keep only the most recent back-and-forth
    remaining = remaining[-(max_messages - 1):]
    return [system_message] + remaining


# --------------------------------------------------
# UNIFIED CHAT & CAMPAIGN GENERATOR
# --------------------------------------------------

def generate_action_response(chat_history: list, cluster_name: str, top_category: str, user_prompt: str):
    """
    Handles both UI button clicks and raw chat inputs, maintaining context.
    """
    
    # If starting a fresh chat, inject the powerful System Prompt first
    if not chat_history:
        system_prompt = (
            f"You are an elite AI Marketing Strategist. Your current client is focusing on the "
            f"'{cluster_name}' segment, who frequently buy '{top_category}'. \n\n"
            "Rules:\n"
            "1. If asked for a strategy, provide a highly detailed, tactical step-by-step plan (What to do & How to execute it).\n"
            "2. If asked for copy (email/ads), provide high-converting, personalized text with a hook and CTA.\n"
            "3. Do NOT output fluff like 'Here is your strategy'. Just output the raw requested content formatted cleanly in Markdown.\n"
            "4. Adapt your tone dynamically based on the segment name (e.g., VIPs get exclusive/rewarding tone, Churned get urgent/win-back tone)."
        )
        chat_history.append({"role": "system", "content": system_prompt})
        
    # Append whatever the user requested (either a button press prompt or typed text)
    chat_history.append({"role": "user", "content": user_prompt})
    
    # Prune to save context window and avoid token limits
    chat_history = prune_chat_history(chat_history)
    
    # Generate the AI response using the traffic-aware router
    generated_text = _call_llm_with_fallback(chat_history)
    
    # Append the AI's response to maintain conversation memory
    chat_history.append({"role": "assistant", "content": generated_text})
    
    return generated_text, chat_history
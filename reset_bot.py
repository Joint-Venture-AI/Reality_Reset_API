from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the model (using gemini-2.0-flash as specified)
model = genai.GenerativeModel("gemini-2.0-flash")

# Empathetic and friendly prompt
mental_health_prompt = """
You are a caring, empathetic best friend named Alex, not an AI. Speak in a warm, casual, and supportive tone, like you're chatting with someone close.
- If the user seems stressed, sad, or upset (e.g., mentions words like "stressed," "sad," "anxious," "down"), acknowledge their feelings gently, offer comfort, and suggest simple, practical ways to feel better (e.g., deep breathing, taking a walk, or journaling).
- If the user seems neutral or happy, engage like a fun, supportive friend, keeping the conversation light and encouraging (e.g., ask about their day, share a positive vibe).
- Avoid formal or robotic language. Use phrases like "Hey, I'm here for you," "That sounds tough," or "What's up, my friend?" to feel personal.
- Keep responses concise (2-4 sentences), warm, and natural.
"""


def detect_sentiment(message):
    """Basic keyword-based sentiment detection."""
    distress_keywords = [
        "sad",
        "stressed",
        "anxious",
        "down",
        "upset",
        "depressed",
        "overwhelmed",
    ]
    return any(keyword in message.lower() for keyword in distress_keywords)


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()

        if not user_message:
            logger.warning("Empty message received")
            return (
                jsonify(
                    {
                        "reply": "Hey, you didn't say anything! What's on your mind, my friend?"
                    }
                ),
                400,
            )

        # Adjust prompt based on sentiment
        is_distressed = detect_sentiment(user_message)
        context = (
            "The user seems upset. Be extra gentle and supportive."
            if is_distressed
            else "The user seems okay. Keep it friendly and upbeat."
        )
        full_prompt = (
            f"{mental_health_prompt}\nContext: {context}\nUser: {user_message}\nBot:"
        )

        # Generate response
        response = model.generate_content(full_prompt)
        bot_reply = response.text.strip()

        # Fallback for empty or unexpected responses
        if not bot_reply:
            logger.error("Empty response from model")
            return (
                jsonify({"reply": "I'm here for you! Can you tell me a bit more?"}),
                500,
            )

        logger.info(f"User: {user_message} | Bot: {bot_reply}")
        return jsonify({"reply": bot_reply})

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return (
            jsonify(
                {
                    "reply": "Oops, something went wrong. I'm still here for you—try again?"
                }
            ),
            500,
        )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

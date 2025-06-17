from flask import Flask, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
import os
import logging
from datetime import datetime
import json

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key-here")

# CORS setup - Add your frontend domains here
CORS(
    app,
    supports_credentials=True,
    origins=["http://127.0.0.1:5500", "http://localhost:5500", "http://localhost:3000"],
)

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chatbot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Gemini setup with error handling
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        "gemini-2.0-flash-exp",
        generation_config={
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        },
    )
    logger.info("Gemini AI model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini AI: {str(e)}")
    model = None

# Enhanced mental health prompt
mental_health_prompt = """
You are Alex, a warm, empathetic, and supportive friend. You're having a genuine conversation with someone who might need emotional support.

IMPORTANT GUIDELINES:
- Always respond as a caring friend, NOT as an AI or therapist
- Use natural, conversational language with warmth and empathy
- Keep responses concise (2-4 sentences) but meaningful
- If someone seems distressed, acknowledge their feelings and offer gentle support
- Suggest practical coping strategies when appropriate (deep breathing, taking walks, journaling, talking to loved ones)
- If someone mentions serious mental health concerns or self-harm, gently encourage them to speak with a mental health professional
- Stay positive and supportive while being authentic
- Ask follow-up questions to show you care and want to understand

TONE EXAMPLES:
- "Hey, I hear you. That sounds really tough."
- "I'm here for you, friend. Want to talk about what's going on?"
- "That's a lot to handle. How are you taking care of yourself right now?"
- "I'm glad you reached out. You don't have to go through this alone."

Remember: Be human, be kind, be present.
"""


def detect_sentiment_and_keywords(message):
    """Enhanced sentiment detection with multiple categories"""
    message_lower = message.lower()

    # Crisis keywords - need immediate attention
    crisis_keywords = [
        "kill myself",
        "end it all",
        "suicide",
        "hurt myself",
        "self harm",
        "want to die",
        "better off dead",
        "can't go on",
    ]

    # High distress keywords
    high_distress_keywords = [
        "depressed",
        "hopeless",
        "overwhelmed",
        "panic",
        "breakdown",
        "can't cope",
        "falling apart",
        "desperate",
    ]

    # Moderate distress keywords
    moderate_distress_keywords = [
        "sad",
        "stressed",
        "anxious",
        "down",
        "upset",
        "worried",
        "tired",
        "exhausted",
        "lonely",
        "frustrated",
    ]

    # Positive keywords
    positive_keywords = [
        "happy",
        "good",
        "great",
        "excited",
        "better",
        "improving",
        "grateful",
        "thankful",
        "optimistic",
        "hopeful",
    ]

    # Check for crisis indicators
    if any(keyword in message_lower for keyword in crisis_keywords):
        return "crisis"

    # Check for high distress
    if any(keyword in message_lower for keyword in high_distress_keywords):
        return "high_distress"

    # Check for moderate distress
    if any(keyword in message_lower for keyword in moderate_distress_keywords):
        return "moderate_distress"

    # Check for positive sentiment
    if any(keyword in message_lower for keyword in positive_keywords):
        return "positive"

    return "neutral"


def get_context_for_sentiment(sentiment_level):
    """Get appropriate context based on sentiment analysis"""
    contexts = {
        "crisis": "URGENT: The user may be in crisis. Respond with immediate care and gently suggest professional help. Be extremely supportive.",
        "high_distress": "The user is experiencing significant distress. Be very gentle, validating, and supportive. Acknowledge their pain.",
        "moderate_distress": "The user seems upset or stressed. Offer comfort and practical suggestions in a caring way.",
        "positive": "The user seems to be doing well! Match their positive energy while staying supportive.",
        "neutral": "Have a normal, friendly conversation. Be warm and check in on how they're doing.",
    }
    return contexts.get(sentiment_level, contexts["neutral"])


def manage_conversation_history(history, max_messages=20):
    """Keep conversation history manageable while preserving important context"""
    if len(history) > max_messages:
        # Keep the first few messages for context and the most recent ones
        return history[:4] + history[-(max_messages - 4) :]
    return history


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "service": "Mental Health Chatbot API",
            "timestamp": datetime.now().isoformat(),
            "gemini_available": model is not None,
        }
    )


@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Check if Gemini is available
        if model is None:
            return (
                jsonify(
                    {
                        "reply": "I'm sorry, I'm having some technical difficulties right now. Please try again later or reach out to someone you trust if you need immediate support."
                    }
                ),
                500,
            )

        data = request.get_json()
        if not data:
            return (
                jsonify(
                    {
                        "reply": "Hey there! I didn't get your message. What's on your mind?"
                    }
                ),
                400,
            )

        user_message = data.get("message", "").strip()
        if not user_message:
            return (
                jsonify(
                    {"reply": "I'm here to listen! What would you like to talk about?"}
                ),
                400,
            )

        # Get or initialize conversation history
        history = session.get("chat_history", [])

        # Analyze sentiment
        sentiment_level = detect_sentiment_and_keywords(user_message)
        context = get_context_for_sentiment(sentiment_level)

        # Log the interaction (without storing sensitive data)
        logger.info(
            f"Chat interaction - Sentiment: {sentiment_level}, Message length: {len(user_message)}"
        )

        # Manage conversation history
        history = manage_conversation_history(history)

        # Create the full prompt with context
        full_prompt = f"{mental_health_prompt}\n\nContext: {context}\n\nUser message: {user_message}"

        # Generate response
        if history:
            # Continue existing conversation
            chat = model.start_chat(history=history)
            response = chat.send_message(full_prompt)
        else:
            # Start new conversation
            response = model.generate_content(full_prompt)

        bot_reply = response.text.strip()

        if not bot_reply:
            bot_reply = (
                "I'm here for you. Can you tell me a bit more about what's going on?"
            )

        # Update conversation history
        history.append({"role": "user", "parts": [user_message]})
        history.append({"role": "model", "parts": [bot_reply]})
        session["chat_history"] = history

        # Store last interaction timestamp
        session["last_interaction"] = datetime.now().isoformat()

        return jsonify(
            {
                "reply": bot_reply,
                "sentiment_detected": sentiment_level,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        return (
            jsonify(
                {
                    "reply": "I'm sorry, something went wrong on my end. I'm still here for you though - please try sending your message again, or reach out to someone you trust if you need immediate support."
                }
            ),
            500,
        )


@app.route("/clear", methods=["POST"])
def clear_history():
    """Clear conversation history"""
    try:
        session.pop("chat_history", None)
        session.pop("last_interaction", None)
        logger.info("Chat history cleared")
        return jsonify(
            {
                "message": "Chat history cleared successfully",
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        return jsonify({"error": "Failed to clear history"}), 500


@app.route("/session-info", methods=["GET"])
def session_info():
    """Get session information"""
    try:
        history_length = len(session.get("chat_history", []))
        last_interaction = session.get("last_interaction")

        return jsonify(
            {
                "session_active": history_length > 0,
                "message_count": history_length
                // 2,  # Divide by 2 since we store both user and bot messages
                "last_interaction": last_interaction,
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        logger.error(f"Session info error: {str(e)}")
        return jsonify({"error": "Failed to get session info"}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Check for required environment variables
    required_vars = ["GEMINI_API_KEY", "FLASK_SECRET_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        print("Please set the following environment variables:")
        for var in missing_vars:
            print(f"  {var}")
        exit(1)

    logger.info("Starting Mental Health Chatbot API...")
    app.run(debug=True, host="0.0.0.0", port=5000)

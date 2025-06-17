from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")


def get_mood_context(mood_score):
    """Get mood context based on score (1-10)"""
    if mood_score <= 3:
        return "very low, struggling, depressed"
    elif mood_score <= 5:
        return "low, feeling down, unmotivated"
    elif mood_score <= 7:
        return "neutral to slightly positive, okay"
    elif mood_score <= 9:
        return "good, positive, motivated"
    else:
        return "excellent, very happy, energetic"


@app.route("/motivational-quotes", methods=["POST"])
def get_motivational_quotes():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        mood_score = data.get("mood_score")
        mood_description = data.get("mood_description", "")

        if mood_score is None:
            return jsonify({"error": "mood_score is required"}), 400

        if not isinstance(mood_score, (int, float)) or not (1 <= mood_score <= 10):
            return (
                jsonify({"error": "mood_score must be a number between 1 and 10"}),
                400,
            )

        mood_context = get_mood_context(mood_score)

        prompt = f"""
        Provide exactly 2-3 motivational quotes for someone with this mood:

        Mood Score: {mood_score}/10
        Mood Description: {mood_description}

        Requirements:
        - Each quote should be on its own line
        - Format: "Quote text" (Author Name) or "Quote text" if author unknown
        - No explanatory text or introductions
        - Choose quotes specifically helpful for this mood level
        - Start each line with just the quote, no numbering

        Example format:
        "The way to get started is to quit talking and begin doing." (Walt Disney)
        "Don't watch the clock; do what it does. Keep going." (Sam Levenson)
        """

        response = model.generate_content(prompt)

        if not response.text:
            return jsonify({"error": "Failed to generate quotes"}), 500

        quotes_text = response.text.strip()
        quotes_lines = [
            line.strip() for line in quotes_text.split("\n") if line.strip()
        ]

        quotes = []
        for line in quotes_lines:
            if any(
                phrase in line.lower()
                for phrase in [
                    "here are",
                    "based on",
                    "tailored for",
                    "mood of",
                    "feeling",
                    "quotes for",
                    "suggestions:",
                    "consider these",
                ]
            ):
                continue

            cleaned_quote = line
            if line and (
                line[0].isdigit() or line.startswith("-") or line.startswith("•")
            ):
                for i, char in enumerate(line):
                    if char in [".", ")", "-", " "] and i < 5:
                        cleaned_quote = line[i + 1 :].strip()
                        break

            if cleaned_quote and ('"' in cleaned_quote or len(cleaned_quote) > 20):
                quotes.append(cleaned_quote)

        if len(quotes) < 2:
            quote_pattern = r'"([^"]+)"[^"]*\([^)]+\)|"([^"]+)"'
            matches = re.findall(quote_pattern, quotes_text)
            for match in matches:
                quote_text = match[0] if match[0] else match[1]
                if quote_text:
                    author_match = re.search(
                        rf'"{re.escape(quote_text)}"[^"]*\(([^)]+)\)', quotes_text
                    )
                    if author_match:
                        quotes.append(f'"{quote_text}" ({author_match.group(1)})')
                    else:
                        quotes.append(f'"{quote_text}"')

        quotes = quotes[:3]
        if len(quotes) < 2:
            fallback_quotes = [
                q.strip()
                for q in quotes_text.split("\n")
                if q.strip() and len(q.strip()) > 20
            ]
            quotes = (
                fallback_quotes[:3]
                if fallback_quotes
                else ["Unable to generate appropriate quotes at this time."]
            )

        return jsonify(
            {
                "mood_score": mood_score,
                "mood_context": mood_context,
                "mood_description": mood_description,
                "motivational_quotes": quotes,
                "total_quotes": len(quotes),
            }
        )

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(
        {"status": "API is running", "message": "Mood-based motivational quotes API"}
    )


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "message": "Mood-Based Motivational Quotes API",
            "usage": {
                "endpoint": "/motivational-quotes",
                "method": "POST",
                "required_fields": {"mood_score": "Number between 1-10 (required)"},
                "optional_fields": {
                    "mood_description": "String describing how you feel"
                },
                "example_request": {
                    "mood_score": 4,
                    "mood_description": "Feeling stressed about work and unmotivated",
                },
            },
        }
    )


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY not found in environment variables")
        print("Please create a .env file with your Gemini API key")
    app.run(debug=True, host="0.0.0.0", port=5000)

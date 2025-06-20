from flask import Flask, request, render_template_string
import os
import sqlite3
from datetime import datetime
from openai import OpenAI
import logging
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(__name__)

# Initialize OpenAI client with API key
openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key:
    logger.error("OPENAI_API_KEY environment variable not set")
    raise ValueError("OpenAI API key is required")

client = OpenAI(api_key=openai_api_key)

# Database setup (SQLite file-based)
DB_PATH = 'search_history.db'

def init_database():
    """Initialize the database with required tables."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''CREATE TABLE IF NOT EXISTS searches (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       topic TEXT NOT NULL,
                       result TEXT NOT NULL,
                       created_at TIMESTAMP NOT NULL
                   )'''
            )
            conn.commit()
            logger.info("Database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
        raise

@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        yield conn
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def validate_topic(topic):
    """Validate and sanitize the topic input."""
    if not topic or not topic.strip():
        return None
    
    topic = topic.strip()
    
    # Basic length validation
    if len(topic) > 200:
        return None
    
    # Remove potentially problematic characters
    # Allow letters, numbers, spaces, and basic punctuation
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-")
    if not all(char in allowed_chars for char in topic):
        return None
    
    return topic

def get_explanation(topic):
    """Get explanation from OpenAI API."""
    prompt = f"""
Explain {topic} in 3 levels:
1. Like I'm 5 (simple, basic concepts)
2. Like I'm 15 (more detail, some technical terms)
3. Like I'm 30 (comprehensive, technical depth)

Keep each explanation concise but informative.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a friendly and knowledgeable explainer bot. Provide clear, accurate explanations at different complexity levels."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise

def save_search(topic, result):
    """Save search to database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO searches (topic, result, created_at) VALUES (?, ?, ?)",
                (topic, result, datetime.utcnow())
            )
            conn.commit()
            logger.info(f"Saved search for topic: {topic}")
    except sqlite3.Error as e:
        logger.error(f"Error saving search: {e}")
        raise

def get_recent_history(limit=5):
    """Get recent search history."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT topic, result FROM searches ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            return [{'topic': row['topic'], 'result': row['result']} for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Error fetching history: {e}")
        return []

# HTML template with improved styling and error handling
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Explainr</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            padding: 40px; 
            max-width: 800px; 
            margin: 0 auto;
            line-height: 1.6;
            color: #333;
        }
        .container { background: white; }
        input[type=text] { 
            width: 100%; 
            max-width: 400px;
            padding: 12px; 
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 16px;
            margin-bottom: 15px;
        }
        input[type=text]:focus {
            outline: none;
            border-color: #4F46E5;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }
        button { 
            padding: 12px 24px; 
            background-color: #4F46E5; 
            color: white; 
            border: none; 
            border-radius: 8px; 
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        button:hover { background-color: #3730A3; }
        button:disabled { 
            background-color: #9CA3AF; 
            cursor: not-allowed; 
        }
        .result { 
            margin-top: 30px; 
            white-space: pre-wrap; 
            background: #f8fafc; 
            padding: 24px; 
            border-radius: 12px; 
            border-left: 4px solid #4F46E5;
        }
        .error {
            margin-top: 30px;
            background: #fef2f2;
            color: #dc2626;
            padding: 16px;
            border-radius: 8px;
            border-left: 4px solid #dc2626;
        }
        .history { 
            margin-top: 50px; 
            border-top: 1px solid #e5e7eb;
            padding-top: 30px;
        }
        .history-item { 
            margin-bottom: 20px; 
            background: #f9fafb;
            padding: 16px;
            border-radius: 8px;
        }
        .history-item strong { 
            display: block; 
            color: #1f2937;
            margin-bottom: 8px;
        }
        .history-result {
            white-space: pre-wrap; 
            background: white; 
            padding: 12px; 
            border-radius: 6px;
            font-size: 14px;
            border: 1px solid #e5e7eb;
        }
        .loading {
            display: none;
            margin-left: 10px;
            color: #6b7280;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #374151;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Explainr</h1>
        <p>Get any topic explained at three different levels of complexity!</p>
        
        <form method="post" onsubmit="showLoading()">
            <div class="form-group">
                <label for="topic">What do you want explained?</label>
                <input type="text" id="topic" name="topic" required 
                       placeholder="e.g., quantum physics, blockchain, photosynthesis..."
                       maxlength="200">
            </div>
            <button type="submit" id="submitBtn">Explain</button>
            <span class="loading" id="loading">Generating explanation...</span>
        </form>
        
        {% if error %}
        <div class="error">
            <strong>Error:</strong> {{ error }}
        </div>
        {% endif %}
        
        {% if result %}
        <div class="result">
            <strong>📚 Explained in 3 levels:</strong><br><br>
            {{ result }}
        </div>
        {% endif %}
        
        {% if history %}
        <div class="history">
            <h2>🔄 Recent Topics</h2>
            {% for item in history %}
            <div class="history-item">
                <strong>{{ item['topic'] }}</strong>
                <div class="history-result">{{ item['result'] }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </div>

    <script>
        function showLoading() {
            document.getElementById('submitBtn').disabled = true;
            document.getElementById('loading').style.display = 'inline';
        }
    </script>
</body>
</html>
'''

@app.route("/", methods=["GET", "POST"])
def explain():
    result = None
    error = None
    
    if request.method == "POST":
        topic = request.form.get('topic', '').strip()
        
        # Validate input
        validated_topic = validate_topic(topic)
        if not validated_topic:
            error = "Please enter a valid topic (letters, numbers, and basic punctuation only, max 200 characters)."
        else:
            try:
                # Get explanation from OpenAI
                result = get_explanation(validated_topic)
                
                # Save to database
                save_search(validated_topic, result)
                
                logger.info(f"Successfully processed topic: {validated_topic}")
                
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                error = "Sorry, there was an error generating the explanation. Please try again."
    
    # Fetch recent history
    history = get_recent_history(5)
    
    return render_template_string(HTML_TEMPLATE, result=result, history=history, error=error)

@app.errorhandler(404)
def not_found(error):
    return "<h1>404 - Page Not Found</h1><p>The page you're looking for doesn't exist.</p>", 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return "<h1>500 - Internal Server Error</h1><p>Something went wrong. Please try again later.</p>", 500

# Initialize database on startup
init_database()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Explainr app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)

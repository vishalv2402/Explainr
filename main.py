from flask import Flask, request, render_template_string, jsonify, send_from_directory, session
import openai
import os
import logging
import re
from typing import Optional, Dict, List, Tuple
from functools import wraps
import time
import sqlite3
from datetime import datetime
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configuration
class Config:
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    MAX_TOPIC_LENGTH = 200
    RATE_LIMIT_REQUESTS = 10  # requests per minute
    OPENAI_MODEL = "gpt-3.5-turbo"
    MAX_RETRIES = 3
    TIMEOUT = 30

config = Config()

# Initialize OpenAI
if config.OPENAI_API_KEY and config.OPENAI_API_KEY != 'your-api-key-here':
    openai.api_key = config.OPENAI_API_KEY
else:
    logger.error("OpenAI API key not properly configured")

# Rate limiting storage (in production, use Redis)
request_timestamps = []

def init_db():
    """Initialize the search history database"""
    conn = sqlite3.connect('search_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_search_to_history(topic: str):
    """Add a search topic to the history"""
    conn = sqlite3.connect('search_history.db')
    cursor = conn.cursor()

    # Remove any existing entry of the same topic
    cursor.execute('DELETE FROM search_history WHERE topic = ?', (topic,))

    # Add the new search
    cursor.execute('INSERT INTO search_history (topic) VALUES (?)', (topic,))

    # Keep only the last 10 searches
    cursor.execute('''
        DELETE FROM search_history 
        WHERE id NOT IN (
            SELECT id FROM search_history 
            ORDER BY timestamp DESC 
            LIMIT 10
        )
    ''')

    conn.commit()
    conn.close()

def get_recent_searches():
    """Get the last 10 search topics"""
    try:
        conn = sqlite3.connect('search_history.db')
        cursor = conn.cursor()
        cursor.execute('SELECT topic FROM search_history ORDER BY timestamp DESC LIMIT 10')
        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        return results
    except sqlite3.Error:
        return []

# Initialize database on startup
init_db()

def rate_limit(max_requests: int = 10):
    """Simple rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            now = time.time()
            # Clean old timestamps
            global request_timestamps
            request_timestamps = [ts for ts in request_timestamps if now - ts < 60]

            if len(request_timestamps) >= max_requests:
                return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

            request_timestamps.append(now)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def sanitize_input(text: str) -> str:
    """Sanitize user input"""
    if not text:
        return ""

    # Remove potentially harmful characters
    text = re.sub(r'[<>"\']', '', text)
    text = text.strip()

    # Limit length
    if len(text) > config.MAX_TOPIC_LENGTH:
        text = text[:config.MAX_TOPIC_LENGTH]

    return text

def make_openai_request(messages: List[Dict], max_retries: int = 3) -> Optional[str]:
    """Make OpenAI API request with error handling and retries"""
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == 'your-api-key-here':
        return "OpenAI API key not configured. Please set the OPENAI_API_KEY environment variable."

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                max_tokens=1500,
                temperature=0.7,
                timeout=config.TIMEOUT
            )
            return response['choices'][0]['message']['content'].strip()

        except openai.error.RateLimitError:
            logger.warning(f"Rate limit hit, attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return "Service temporarily unavailable due to high demand. Please try again later."

        except openai.error.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            return f"API Error: Unable to process request. Please try again."

        except openai.error.InvalidRequestError as e:
            logger.error(f"Invalid request: {e}")
            return "Invalid request. Please check your input and try again."

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return "An unexpected error occurred. Please try again."

    return "Request failed after multiple attempts."

def generate_explanation(topic: str, explanation_type: str) -> Tuple[Optional[str], List[str], List[str]]:
    """Generate explanation with follow-up questions and related topics"""

    if explanation_type == "simple":
        main_prompt = f"""
        Explain "{topic}" for a 5-year-old child in 2-3 short paragraphs. Use:
        - Very simple words and short sentences
        - One fun example they can relate to
        - Keep it brief and engaging
        """
    elif explanation_type == "teen":
        main_prompt = f"""
        Explain "{topic}" for a 15-year-old student in 2-3 paragraphs. Use:
        - Clear explanations with some technical terms
        - One real-world example
        - Why it matters to them
        - Keep it concise and informative
        """
    else:  # adult
        main_prompt = f"""
        Explain "{topic}" for an adult in 2-3 paragraphs. Include:
        - Technical terminology and precise language
        - Key applications and implications
        - Keep it comprehensive but concise
        """

    messages = [
        {"role": "system", "content": "You are an expert educator who explains complex topics clearly at different levels. Always use the exact format requested with clear level headings."},
        {"role": "user", "content": main_prompt}
    ]

    result = make_openai_request(messages)

    # Clean up the result formatting
    if result and not result.startswith(("Error:", "API Error:", "Service temporarily")):
        # Remove ** markdown formatting
        result = re.sub(r'\*\*(.*?)\*\*', r'\1', result)
        result = re.sub(r'\*(.*?)\*', r'\1', result)

        # Split into lines and clean each one
        lines = result.split('\n')
        cleaned_lines = []

        for line in lines:
            # Strip whitespace
            stripped = line.strip()
            if stripped:  # Only keep non-empty lines
                cleaned_lines.append(stripped)

        # Join with single newlines between paragraphs
        result = '\n\n'.join(cleaned_lines)

        # Final cleanup - ensure no more than double newlines and clean start/end
        result = re.sub(r'\n{3,}', '\n\n', result)  # Max 2 consecutive newlines
        result = result.strip()

    # Generate follow-up questions
    followup_messages = [
        {"role": "system", "content": "Generate exactly 3 thoughtful follow-up questions. Return only the questions, one per line, without numbering."},
        {"role": "user", "content": f"Generate 3 follow-up questions someone might ask after learning about {topic}."}
    ]

    followup_response = make_openai_request(followup_messages)
    followup_questions = []
    if followup_response and not followup_response.startswith(("Error:", "API Error:", "Service temporarily")):
        followup_questions = [q.strip() for q in followup_response.split('\n') if q.strip()][:3]

    # Generate related topics
    related_messages = [
        {"role": "system", "content": "Generate exactly 5 related topics. Return only the topic names, one per line."},
        {"role": "user", "content": f"List 5 topics closely related to {topic} that would be interesting to explore."}
    ]

    related_response = make_openai_request(related_messages)
    related_topics = []
    if related_response and not related_response.startswith(("Error:", "API Error:", "Service temporarily")):
        related_topics = [t.strip() for t in related_response.split('\n') if t.strip()][:5]

    return result, followup_questions, related_topics

def generate_new_suggested_questions(topic: str, asked_questions: List[str], explanation_type: str) -> List[str]:
    """Generate new suggested questions excluding already asked ones"""
    # Build context of already asked questions
    asked_context = ""
    if asked_questions:
        asked_context = f"\n\nAvoid these questions that have already been asked:\n" + "\n".join(f"- {q}" for q in asked_questions)
    
    messages = [
        {"role": "system", "content": f"Generate exactly 3 new follow-up questions about {topic} at a {explanation_type} level. Return only the questions, one per line, without numbering."},
        {"role": "user", "content": f"Generate 3 NEW follow-up questions someone might ask after learning about {topic}.{asked_context}"}
    ]
    
    response = make_openai_request(messages)
    new_questions = []
    if response and not response.startswith(("Error:", "API Error:", "Service temporarily")):
        new_questions = [q.strip() for q in response.split('\n') if q.strip()][:3]
    
    return new_questions

# Enhanced HTML template with modern premium design
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Explainr - Master Any Concept</title>
    <meta name="description" content="Transform complex ideas into clear understanding with AI-powered explanations">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        * { 
            margin: 0; 
            padding: 0; 
            box-sizing: border-box; 
        }

        body { 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #faf9f7 0%, #f5f3f0 100%);
            min-height: 100vh;
            color: #2c2c2c;
            line-height: 1.7;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 32px 24px;
            text-align: left;
        }

        .layout {
            display: flex;
            gap: 32px;
            align-items: start;
            min-height: 100vh;
            width: 100%;
        }

        .sidebar {
            position: sticky;
            top: 32px;
            width: 300px;
            flex-shrink: 0;
        }

        .main-content {
            flex: 1;
            min-width: 0;
            overflow-wrap: break-word;
            text-align: left;
        }

        .header {
            text-align: center;
            margin-bottom: 24px;
            padding-top: 40px;
        }

        .logo {
            display: inline-flex;
            align-items: center;
            gap: 12px;
            margin-bottom: -8px;
        }

        .logo-icon {
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        h1 { 
            font-size: 2.5rem;
            font-weight: 700;
            color: #1a1a1a;
            margin: 0;
            letter-spacing: -0.02em;
        }

        .tagline {
            font-size: 1.125rem;
            color: #666;
            font-weight: 400;
            margin-top: -40px;
        }

        .card {
            background: white;
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
            border: 1px solid #f0f0f0;
            margin-bottom: 24px;
        }

        .input-group {
            margin-bottom: 24px;
        }

        label { 
            display: block;
            font-weight: 500; 
            color: #2c2c2c; 
            margin-bottom: 8px; 
            font-size: 0.925rem;
        }

        input[type=text] { 
            width: 100%; 
            padding: 16px 20px; 
            border: 2px solid #e8e8e8;
            border-radius: 12px;
            font-size: 16px;
            font-family: inherit;
            transition: all 0.2s ease;
            background: #fafafa;
        }

        input[type=text]:focus {
            outline: none;
            border-color: #1a1a1a;
            background: white;
            box-shadow: 0 0 0 4px rgba(26, 26, 26, 0.08);
        }

        .style-selector {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 24px;
        }

        .style-option {
            position: relative;
        }

        .style-option input[type=radio] {
            position: absolute;
            opacity: 0;
            pointer-events: none;
        }

        .style-option label {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 16px 20px;
            border: 2px solid #e8e8e8;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.2s ease;
            font-weight: 500;
            background: #fafafa;
            margin: 0;
        }

        .style-option input[type=radio]:checked + label {
            border-color: #1a1a1a;
            background: #1a1a1a;
            color: white;
        }

        .submit-btn { 
            width: 100%;
            padding: 16px 24px; 
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            color: white; 
            border: none; 
            border-radius: 12px; 
            font-size: 16px;
            font-weight: 600;
            font-family: inherit;
            cursor: pointer; 
            transition: all 0.2s ease;
            position: relative;
            overflow: hidden;
        }

        .submit-btn:hover:not(:disabled) {
            transform: translateY(-1px);
            box-shadow: 0 8px 24px rgba(26, 26, 26, 0.2);
        }

        .submit-btn:disabled {
            opacity: 0.7;
            cursor: not-allowed;
            transform: none;
        }

        .loading {
            display: none;
            text-align: center;
            margin: 24px 0;
            color: #666;
        }

        .spinner {
            border: 2px solid #f0f0f0;
            border-top: 2px solid #1a1a1a;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
            margin: 0 auto 12px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Fixed result section with stronger left alignment */
        .result { 
            white-space: pre-wrap; 
            background: white;
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
            border: 1px solid #f0f0f0;
            position: relative;
            margin-bottom: 24px;
            line-height: 1.8;
            text-align: left !important;
        }

        .result-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #1a1a1a;
            text-align: left !important;
            margin-bottom: 16px;
            display: block;
            width: 100%;
            padding: 0;
            margin-left: 0;
            margin-right: 0;
        }

        /* Stronger left alignment rules for all result content */
        .result,
        .result *,
        .result p,
        .result div,
        .result span,
        .result-content,
        .result-content * {
            text-align: left !important;
            margin-left: 0 !important;
            margin-right: auto !important;
        }

        .result-content {
            margin: 0 !important;
            word-wrap: break-word;
            overflow-wrap: break-word;
            text-align: left !important;
            display: block;
            width: 100%;
        }

        .error {
            background: #fef2f2;
            border-color: #fed7d7;
            color: #c53030;
        }

        .topics-section { 
            background: white;
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
            border: 1px solid #f0f0f0;
            margin-bottom: 24px;
        }

        .section-title { 
            font-size: 1.125rem;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .topics-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }

        .topic-tag { 
            background: #f5f5f5;
            color: #2c2c2c; 
            padding: 8px 16px; 
            border-radius: 24px; 
            font-size: 14px; 
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            border: 1px solid #e8e8e8;
        }

        .topic-tag:hover { 
            background: #1a1a1a;
            color: white;
            transform: translateY(-1px);
        }

        .clear-btn:hover {
            background: #e8e8e8 !important;
            color: #333 !important;
            border-color: #d0d0d0 !important;
        }

        .question-item { 
            display: block; 
            background: #fafafa;
            padding: 16px 20px; 
            margin: 8px 0; 
            border-radius: 12px; 
            color: #2c2c2c; 
            border: 1px solid #e8e8e8;
            transition: all 0.2s ease;
            cursor: pointer;
            font-weight: 400;
        }

        .question-item:hover { 
            background: white;
            border-color: #1a1a1a;
            transform: translateX(4px);
        }

        .popular-topics {
            margin-bottom: 32px;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .card, .result, .topics-section {
            animation: fadeIn 0.4s ease-out;
        }

        @media (max-width: 768px) {
            .container { padding: 20px 16px; }
            h1 { font-size: 2rem; }
            .card, .topics-section { padding: 24px; }
            .style-selector { grid-template-columns: 1fr; }
            .result-header { flex-direction: column; gap: 12px; align-items: stretch; }
            .layout { 
                flex-direction: column;
                gap: 24px; 
            }
            .sidebar {
                position: static;
                width: 100%;
                order: 2;
            }
            .main-content {
                order: 1;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">
                <div class="logo-icon">
                    <img src="/logo.png" alt="Explainr Logo" style="width: 419px; height: auto; max-height: 210px; object-fit: contain;">
                </div>
            </div>
            <p class="tagline">Transform complex concepts into crystal-clear understanding!</p>
        </div>

        <div class="layout">
            <div class="sidebar">
                <div class="topics-section popular-topics">
                    <div class="section-title">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="5"/>
                            <line x1="12" y1="1" x2="12" y2="3"/>
                            <line x1="12" y1="21" x2="12" y2="23"/>
                            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                            <line x1="1" y1="12" x2="3" y2="12"/>
                            <line x1="21" y1="12" x2="23" y2="12"/>
                            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
                        </svg>
                        Popular Topics
                    </div>
                    <div class="topics-grid">
                        <span class="topic-tag" onclick="fillTopic(&quot;Quantum Computing&quot;)">Quantum Computing</span>
                        <span class="topic-tag" onclick="fillTopic(&quot;Machine Learning&quot;)">Machine Learning</span>
                        <span class="topic-tag" onclick="fillTopic(&quot;Blockchain&quot;)">Blockchain</span>
                        <span class="topic-tag" onclick="fillTopic(&quot;Climate Change&quot;)">Climate Change</span>
                        <span class="topic-tag" onclick="fillTopic(&quot;DNA&quot;)">DNA</span>
                        <span class="topic-tag" onclick="fillTopic(&quot;Black Holes&quot;)">Black Holes</span>
                        <span class="topic-tag" onclick="fillTopic(&quot;Cryptocurrency&quot;)">Cryptocurrency</span>
                        <span class="topic-tag" onclick="fillTopic(&quot;Stock Market&quot;)">Stock Market</span>
                    </div>
                </div>

                <div class="topics-section">
                    <div class="section-title">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="3"/>
                            <path d="M12 1v6m0 6v6"/>
                            <path d="M1 12h6m6 0h6"/>
                        </svg>
                        Recent Searches
                    </div>
                    <div class="topics-grid">
                        <span class="topic-tag" onclick="fillTopic(&quot;Neural Networks&quot;)">Neural Networks</span>
                        <span class="topic-tag" onclick="fillTopic(&quot;Photosynthesis&quot;)">Photosynthesis</span>
                        <span class="topic-tag" onclick="fillTopic(&quot;Relativity&quot;)">Relativity</span>
                    </div>
                </div>
            </div>

            <div class="main-content">
                <div class="card">
                    <form method="post" id="explainForm" onsubmit="showLoading()">
                        <div class="input-group">
                            <label for="topic">What would you like to be explained?</label>
                            <input type="text" id="topic" name="topic" placeholder="Enter any topic..." required maxlength="200">
                        </div>

                        <div class="input-group">
                            <label>Choose explanation style</label>
                            <div class="style-selector">
                                <div class="style-option">
                                    <input type="radio" id="simple" name="explanation_type" value="simple" checked>
                                    <label for="simple">
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                            <circle cx="12" cy="12" r="10"/>
                                            <circle cx="12" cy="12" r="4"/>
                                            <line x1="21.17" y1="8" x2="12" y2="8"/>
                                            <line x1="3.95" y1="6.06" x2="8.54" y2="14"/>
                                            <line x1="10.88" y1="21.94" x2="15.46" y2="14"/>
                                        </svg>
                                        Twinkle (Like you're 5!)
                                    </label>
                                </div>
                                <div class="style-option">
                                    <input type="radio" id="teen" name="explanation_type" value="teen">
                                    <label for="teen">
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                            <path d="m22 10-6-6-4 4-4-4-6 6v12h20z"/>
                                            <path d="M6 12h16"/>
                                            <path d="m16 16 2 2 4-4"/>
                                        </svg>
                                        Shine (Like you're 15!)
                                    </label>
                                </div>
                                <div class="style-option">
                                    <input type="radio" id="adult" name="explanation_type" value="adult">
                                    <label for="adult">
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                            <circle cx="12" cy="12" r="10"/>
                                            <circle cx="12" cy="12" r="6"/>
                                            <circle cx="12" cy="12" r="2"/>
                                        </svg>
                                        Radiate (Like you're 30!)
                                    </label>
                                </div>
                            </div>
                        </div>

                        <div style="display: flex; gap: 12px;">
                            <button type="submit" class="submit-btn" id="submitBtn" style="flex: 1;">Generate Explanation</button>
                            {% if result %}
                            <button onclick="clearMainResult()" class="submit-btn" style="flex: 1; background: #f5f5f5; color: #666; border: 1px solid #e8e8e8;">Clear</button>
                            {% endif %}
                        </div>
                    </form>

                    <div class="loading" id="loading">
                        <div class="spinner"></div>
                        <p>Crafting your explanation...</p>
                    </div>
                </div>

                {% if result %}
                <div class="result" id="mainResult">
                    <div class="result-title">{{ current_topic if current_topic else 'Topic' }}</div>
                    <div class="result-content">{{ result }}</div>
                </div>
                {% endif %}

                {% if followup_questions or followup_conversation %}
                <div class="topics-section" id="followupSection">
                    <div class="section-title" style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <circle cx="12" cy="12" r="10"/>
                                <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                                <point cx="12" cy="17"/>
                            </svg>
                            Follow-up Questions
                        </div>
                        {% if followup_conversation and followup_conversation|length > 0 %}
                        <button onclick="clearFollowupConversation()" class="clear-btn" style="padding: 6px 12px; background: #f5f5f5; color: #666; border: 1px solid #e8e8e8; border-radius: 6px; font-size: 12px; cursor: pointer; transition: all 0.2s ease;">
                            Clear Conversation
                        </button>
                        {% endif %}
                    </div>

                    <!-- Conversation History -->
                    {% if followup_conversation and followup_conversation|length > 0 %}
                    <div class="conversation-history" id="conversationHistory" style="margin-bottom: 24px;">
                        {% for exchange in followup_conversation %}
                        <div class="conversation-item" style="margin-bottom: 20px; border-left: 3px solid #1a1a1a; padding-left: 16px; background: #f9f9f9; border-radius: 8px; padding: 16px;">
                            <div class="conversation-question" style="font-weight: 600; color: #1a1a1a; margin-bottom: 8px;">
                                Q: {{ exchange.question }}
                            </div>
                            <div class="conversation-answer" style="color: #2c2c2c; line-height: 1.6; white-space: pre-wrap;">
                                {{ exchange.answer }}
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                    {% endif %}

                    <!-- Follow-up Question Input -->
                    {% if result %}
                    <form method="post" id="followupForm" style="margin-bottom: 20px;">
                        <input type="hidden" name="original_topic" value="{{ request.form.get('topic', '') if request.form.get('topic') else request.form.get('original_topic', '') }}">
                        <input type="hidden" name="explanation_type" value="{{ request.form.get('explanation_type', 'simple') }}">
                        <input type="hidden" name="original_result" value="{{ result | e if result else '' }}">
                        
                        <div style="display: flex; gap: 12px; align-items: stretch;">
                            <input type="text" name="followup_question" placeholder="Ask a follow-up question..." 
                                   style="flex: 1; padding: 12px 16px; border: 2px solid #e8e8e8; border-radius: 8px; font-size: 14px;" required>
                            <button type="submit" style="padding: 12px 20px; background: #1a1a1a; color: white; border: none; border-radius: 8px; font-weight: 500; cursor: pointer; white-space: nowrap;">
                                Ask
                            </button>
                        </div>
                    </form>
                    {% endif %}

                    <!-- Suggested Follow-up Questions -->
                    {% if followup_questions %}
                    <div style="margin-top: 16px;">
                        <div style="font-size: 0.9rem; color: #666; margin-bottom: 12px;">Suggested questions:</div>
                        {% for question in followup_questions %}
                        <span class="question-item" onclick="fillFollowup(&quot;{{ question | e | replace('&quot;', '&amp;quot;') }}&quot;)">{{ question }}</span>
                        {% endfor %}
                    </div>
                    {% endif %}
                </div>
                {% endif %}

                {% if related_topics %}
                <div class="topics-section">
                    <div class="section-title">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
                            <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
                        </svg>
                        Related Topics
                    </div>
                    <div class="topics-grid">
                        {% for topic in related_topics %}
                        <span class="topic-tag" onclick="fillTopic(&quot;{{ topic | e | replace('&quot;', '&amp;quot;') }}&quot;)">{{ topic }}</span>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
            </div>
        </div>
    </div>

    <script>
        function fillTopic(topic) {
            document.getElementById('topic').value = topic;
            document.getElementById('topic').focus();
            document.getElementById('explainForm').scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        function fillFollowup(question) {
            const followupInput = document.querySelector('input[name="followup_question"]');
            if (followupInput) {
                followupInput.value = question;
                followupInput.focus();
            }
        }

        function clearMainResult() {
            const mainResult = document.getElementById('mainResult');
            if (mainResult) {
                mainResult.style.opacity = '0';
                mainResult.style.transform = 'translateY(-10px)';
                setTimeout(() => {
                    mainResult.remove();
                    // Clear the form to reset state
                    window.location.href = '/';
                }, 200);
            }
        }

        function clearFollowupConversation() {
            if (confirm('Are you sure you want to clear the conversation history?')) {
                // Clear conversation history from the current session
                fetch(window.location.pathname, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: 'clear_conversation=true'
                }).then(response => {
                    if (response.ok) {
                        // Remove conversation history from DOM
                        const conversationHistory = document.getElementById('conversationHistory');
                        const clearButton = document.querySelector('.clear-btn');
                        if (conversationHistory) {
                            conversationHistory.style.opacity = '0';
                            conversationHistory.style.transform = 'translateY(-10px)';
                            setTimeout(() => {
                                conversationHistory.remove();
                                if (clearButton && clearButton.textContent.includes('Clear Conversation')) {
                                    clearButton.remove();
                                }
                            }, 200);
                        }
                    }
                });
            }
        }

        function showLoading() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('submitBtn').disabled = true;
            document.getElementById('submitBtn').textContent = 'Generating...';
        }

        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('explainForm').addEventListener('submit', function(e) {
                const topic = document.getElementById('topic').value.trim();
                if (!topic) {
                    e.preventDefault();
                    alert('Please enter a topic to explain.');
                    return false;
                }
                if (topic.length > 200) {
                    e.preventDefault();
                    alert('Topic is too long. Please keep it under 200 characters.');
                    return false;
                }

                showLoading();

                setTimeout(function() {
                    document.getElementById('topic').value = '';
                }, 100);
            });

            // Handle follow-up form submission
            const followupForm = document.getElementById('followupForm');
            if (followupForm) {
                followupForm.addEventListener('submit', function(e) {
                    const followupInput = document.querySelector('input[name="followup_question"]');
                    if (followupInput && followupInput.value.trim() === '') {
                        e.preventDefault();
                        alert('Please enter a follow-up question.');
                        return false;
                    }
                    
                    // Clear the follow-up input after a short delay to allow form submission
                    setTimeout(function() {
                        if (followupInput) {
                            followupInput.value = '';
                        }
                    }, 100);
                });
            }
        });
    </script>
</body>
</html>
'''

@app.route("/", methods=["GET", "POST"])
@rate_limit(config.RATE_LIMIT_REQUESTS)
def explain():
    result = None
    followup_questions = []
    related_topics = []
    followup_response = None
    followup_conversation = []
    recent_searches = get_recent_searches()

    if request.method == "POST":
        # Check if this is a request to clear conversation
        if 'clear_conversation' in request.form:
            # Clear all conversation sessions
            keys_to_remove = [key for key in session.keys() if key.startswith('conversation_')]
            for key in keys_to_remove:
                del session[key]
            return jsonify({"status": "success"})
        
        # Check if this is a main topic request or a follow-up question
        if 'followup_question' in request.form:
            # Handle follow-up question
            followup_question = sanitize_input(request.form.get('followup_question', ''))
            original_topic = sanitize_input(request.form.get('original_topic', ''))
            explanation_type = request.form.get('explanation_type', 'simple')
            
            # Get existing conversation from session
            session_key = f"conversation_{original_topic}"
            followup_conversation = session.get(session_key, [])
            logger.info(f"Loaded {len(followup_conversation)} previous conversations from session")
            
            if followup_question and original_topic:
                logger.info(f"Generating follow-up answer for: {followup_question}")
                logger.info(f"Current conversation history length: {len(followup_conversation)}")
                
                # Get preserved original result or generate if not available
                original_result = request.form.get('original_result', '')
                if original_result:
                    import html
                    result = html.unescape(original_result)
                    # Generate related topics and questions for the original topic only if not already available
                    if not followup_questions and not related_topics:
                        _, followup_questions, related_topics = generate_explanation(original_topic, explanation_type)
                else:
                    # Fallback: generate fresh explanation
                    result, followup_questions, related_topics = generate_explanation(original_topic, explanation_type)
                
                # Build conversation context for better follow-up answers
                conversation_context = ""
                if followup_conversation:
                    conversation_context = "\n\nPrevious conversation:\n"
                    for i, exchange in enumerate(followup_conversation, 1):
                        conversation_context += f"Q{i}: {exchange['question']}\nA{i}: {exchange['answer']}\n\n"
                
                # Generate follow-up response with context
                followup_messages = [
                    {"role": "system", "content": f"You are explaining topics at a {explanation_type} level. Answer this follow-up question about {original_topic} clearly and concisely. Consider the previous conversation context if provided."},
                    {"role": "user", "content": f"Original topic: {original_topic}{conversation_context}\nCurrent follow-up question: {followup_question}"}
                ]
                
                followup_response = make_openai_request(followup_messages)
                
                if followup_response and not followup_response.startswith(("Error:", "API Error:", "Service temporarily")):
                    # Clean up formatting
                    followup_response = re.sub(r'\*\*(.*?)\*\*', r'\1', followup_response)
                    followup_response = re.sub(r'\*(.*?)\*', r'\1', followup_response)
                    followup_response = followup_response.strip()
                    
                    # Add to conversation history
                    followup_conversation.append({
                        'question': followup_question,
                        'answer': followup_response
                    })
                    
                    # Save to session
                    session[session_key] = followup_conversation
                    logger.info(f"Updated conversation history length: {len(followup_conversation)}")
                    
                    # Generate new suggested questions excluding already asked ones
                    asked_questions = [exchange['question'] for exchange in followup_conversation]
                    followup_questions = generate_new_suggested_questions(original_topic, asked_questions, explanation_type)
        else:
            # Handle main topic request
            topic = sanitize_input(request.form.get('topic', ''))
            explanation_type = request.form.get('explanation_type', 'simple')

            if not topic:
                result = "Error: Please enter a topic to explain."
            elif len(topic.strip()) < 2:
                result = "Error: Topic too short. Please enter a meaningful topic."
            else:
                logger.info(f"Generating explanation for: {topic}")
                # Add to search history only if it's a valid search
                add_search_to_history(topic)
                result, followup_questions, related_topics = generate_explanation(topic, explanation_type)
                # Clear any existing conversation history for this topic
                session_key = f"conversation_{topic}"
                if session_key in session:
                    del session[session_key]
                # Refresh recent searches after adding new one
                recent_searches = get_recent_searches()

    # Determine the current topic for display
    current_topic = None
    if request.method == "POST":
        if 'followup_question' in request.form:
            current_topic = sanitize_input(request.form.get('original_topic', ''))
        else:
            current_topic = sanitize_input(request.form.get('topic', ''))
    
    return render_template_string(
        HTML_TEMPLATE, 
        result=result, 
        followup_questions=followup_questions, 
        related_topics=related_topics,
        recent_searches=recent_searches,
        followup_conversation=followup_conversation,
        current_topic=current_topic
    )



@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files like images"""
    return send_from_directory('.', filename)

@app.route('/attached_assets/<path:filename>')
def serve_assets(filename):
    """Serve attached assets"""
    return send_from_directory('attached_assets', filename)

@app.errorhandler(404)
def not_found(error):
    return render_template_string('''
    <h1>Page Not Found</h1>
    <p><a href="/">Return to Explainr</a></p>
    '''), 404



@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return render_template_string('''
    <h1>Internal Server Error</h1>
    <p>Something went wrong. Please try again later.</p>
    <p><a href="/">Return to Explainr</a></p>
    '''), 500

if __name__ == "__main__":
    # Production-ready configuration
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    if not debug:
        # In production, disable debug mode and use proper logging
        logging.basicConfig(level=logging.INFO)

    app.run(host='0.0.0.0', port=port, debug=debug)

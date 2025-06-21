from flask import Flask, request, render_template_string, make_response, jsonify
import openai
import os
import logging
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
from datetime import datetime
import re
from typing import Optional, Dict, List, Tuple
from functools import wraps
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
class Config:
    # Check for both variable names to ensure compatibility
    OPENAI_API_KEY = os.environ.get('ExplainrOpenAIKey') or os.environ.get('OPENAI_API_KEY')
    MAX_TOPIC_LENGTH = 200
    RATE_LIMIT_REQUESTS = 10
    OPENAI_MODEL = "gpt-3.5-turbo"
    MAX_RETRIES = 3
    TIMEOUT = 30

config = Config()

# Initialize OpenAI with better error handling
if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == 'your-api-key-here':
    logger.error("OpenAI API key not configured. Please set either ExplainrOpenAIKey or OPENAI_API_KEY environment variable.")
else:
    openai.api_key = config.OPENAI_API_KEY
    logger.info("OpenAI API key configured successfully")

# Rate limiting storage (in production, use Redis)
request_timestamps = []

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
        return "OpenAI API key not configured. Please set the ExplainrOpenAIKey environment variable."
    
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
    
    if explanation_type == "example":
        style_instruction = "Add a clear, concrete example for each level."
    else:  # analogy
        style_instruction = "Add a simple, relatable analogy for each level."
    
    # Main explanation prompt
    main_prompt = f"""
    Explain "{topic}" in exactly 3 levels with clear headings:
    
    **Level 1 (Age 5):** Simple explanation using basic words
    **Level 2 (Age 15):** More detailed with some technical terms
    **Level 3 (Adult):** Comprehensive explanation with full context
    
    {style_instruction}
    
    Keep each level concise but informative. Use engaging language.
    """

    messages = [
        {"role": "system", "content": "You are an expert educator who explains complex topics clearly at different levels. Always use the exact format requested with clear level headings."},
        {"role": "user", "content": main_prompt}
    ]
    
    result = make_openai_request(messages)
    
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

# Enhanced HTML template with better error handling and accessibility
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Explainr - AI-Powered Learning Revolution</title>
    <meta name="description" content="Transform complex concepts into crystal-clear understanding with AI-powered explanations">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap">
    <style>
        :root {
            --primary-black: #0A0A0A;
            --secondary-black: #1A1A1A;
            --charcoal: #2D2D2D;
            --warm-grey: #8B8B8B;
            --light-grey: #E8E8E8;
            --cream: #FEFCF8;
            --warm-white: #FDFDFD;
            --beige: #F7F5F2;
            --accent-gold: #D4AF37;
            --accent-warm: #F5E6D3;
            --border-light: #F0F0F0;
            --border-subtle: #E5E5E5;
            --shadow-soft: rgba(0, 0, 0, 0.03);
            --shadow-medium: rgba(0, 0, 0, 0.08);
            --shadow-strong: rgba(0, 0, 0, 0.12);
        }

        * { 
            margin: 0; 
            padding: 0; 
            box-sizing: border-box; 
        }
        
        body { 
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, var(--cream) 0%, var(--beige) 100%);
            color: var(--primary-black);
            line-height: 1.6;
            min-height: 100vh;
            font-weight: 400;
        }

        .container {
            max-width: 1100px;
            margin: 3rem auto;
            padding: 0;
            background: var(--warm-white);
            border-radius: 24px;
            box-shadow: 
                0 32px 64px var(--shadow-soft),
                0 16px 32px var(--shadow-medium),
                0 0 0 1px var(--border-light);
            overflow: hidden;
        }

        .header {
            text-align: center;
            padding: 4rem 3rem 3rem;
            background: linear-gradient(135deg, var(--warm-white) 0%, var(--cream) 100%);
            position: relative;
        }

        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent 0%, var(--accent-gold) 50%, transparent 100%);
        }

        .brand-name {
            display: inline-block;
            margin-bottom: 1rem;
        }

        .brand-name .explain {
            font-family: 'Outfit', sans-serif;
            font-size: 4rem;
            font-weight: 300;
            color: var(--charcoal);
            letter-spacing: -0.04em;
        }

        .brand-name .r {
            font-family: 'Outfit', sans-serif;
            font-size: 4rem;
            font-weight: 300;
            color: var(--warm-white);
            background: var(--primary-black);
            padding: 0 0.3em;
            border-radius: 8px;
            margin-left: -0.05em;
        }

        .subtitle {
            color: var(--warm-grey);
            font-size: 1.1rem;
            font-weight: 400;
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.6;
        }

        .form-section {
            padding: 3rem;
        }

        .input-group {
            margin-bottom: 2.5rem;
        }

        .input-label {
            display: block;
            font-weight: 500;
            color: var(--primary-black);
            margin-bottom: 1rem;
            font-size: 1rem;
        }

        .input-field {
            width: 100%;
            padding: 1.25rem 1.5rem;
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            font-size: 1rem;
            font-family: inherit;
            font-weight: 400;
            transition: all 0.2s ease;
            background: var(--warm-white);
            color: var(--primary-black);
        }

        .input-field:focus {
            outline: none;
            border-color: var(--charcoal);
            box-shadow: 0 0 0 3px rgba(45, 45, 45, 0.1);
        }

        .input-field::placeholder {
            color: var(--warm-grey);
        }

        .learning-style-group {
            background: var(--beige);
            padding: 2rem;
            border-radius: 16px;
            margin: 2rem 0;
        }

        .learning-style-label {
            display: block;
            font-weight: 500;
            color: var(--primary-black);
            margin-bottom: 1.5rem;
            font-size: 1rem;
        }

        .radio-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .radio-card {
            position: relative;
            background: var(--warm-white);
            border: 1px solid var(--border-light);
            border-radius: 12px;
            padding: 1.5rem;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .radio-card:hover {
            border-color: var(--charcoal);
            box-shadow: 0 4px 12px var(--shadow-soft);
        }

        .radio-card input[type="radio"] {
            width: 18px;
            height: 18px;
            margin: 0;
            accent-color: var(--primary-black);
        }

        .radio-card.selected {
            border-color: var(--primary-black);
            background: var(--primary-black);
            color: var(--warm-white);
        }

        .radio-card .icon {
            width: 20px;
            height: 20px;
            opacity: 0.7;
        }

        .radio-card.selected .icon {
            opacity: 1;
        }

        .radio-text {
            font-weight: 500;
            font-size: 0.95rem;
        }

        .generate-btn {
            width: 100%;
            padding: 1.25rem 2rem;
            background: var(--primary-black);
            color: var(--warm-white);
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 500;
            font-family: inherit;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            position: relative;
            overflow: hidden;
        }

        .generate-btn:hover:not(:disabled) {
            background: var(--charcoal);
            transform: translateY(-1px);
            box-shadow: 0 8px 24px var(--shadow-medium);
        }

        .generate-btn:disabled {
            opacity: 0.7;
            cursor: not-allowed;
        }

        .btn-icon {
            width: 18px;
            height: 18px;
        }

        .result-section {
            background: var(--warm-white);
            padding: 3rem;
            border-top: 1px solid var(--border-light);
            position: relative;
        }

        .result-content {
            line-height: 1.8;
            color: var(--primary-black);
        }

        .result-content h2 {
            color: var(--primary-black);
            margin: 2rem 0 1rem;
            font-size: 1.5rem;
            font-weight: 600;
        }

        .result-content h3 {
            color: var(--charcoal);
            margin: 1.5rem 0 0.75rem;
            font-size: 1.2rem;
            font-weight: 500;
        }

        .result-content p {
            margin-bottom: 1rem;
            color: var(--primary-black);
        }

        .result-content strong {
            font-weight: 600;
            color: var(--primary-black);
        }

        .export-btn {
            position: absolute;
            top: 2rem;
            right: 3rem;
            background: var(--charcoal);
            color: var(--warm-white);
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .export-btn:hover {
            background: var(--primary-black);
        }

        .suggestions-section {
            background: var(--beige);
            padding: 3rem;
            border-top: 1px solid var(--border-light);
        }

        .suggestions-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--primary-black);
            margin-bottom: 1.5rem;
        }

        .topics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }

        .topic-card {
            background: var(--warm-white);
            border: 1px solid var(--border-light);
            border-radius: 12px;
            padding: 1.25rem;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: center;
            font-weight: 500;
            font-size: 0.95rem;
            color: var(--primary-black);
        }

        .topic-card:hover {
            border-color: var(--charcoal);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px var(--shadow-soft);
        }

        .loading-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(8px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }

        .loading-content {
            text-align: center;
            padding: 2rem;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 2px solid var(--light-grey);
            border-top-color: var(--primary-black);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }

        .loading-text {
            color: var(--charcoal);
            font-weight: 500;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @media (max-width: 768px) {
            .container {
                margin: 1rem;
                border-radius: 16px;
            }

            .header {
                padding: 3rem 2rem 2rem;
            }

            .brand-name .explain,
            .brand-name .r {
                font-size: 3rem;
            }

            .form-section {
                padding: 2rem;
            }

            .radio-grid {
                grid-template-columns: 1fr;
            }

            .export-btn {
                position: static;
                margin-bottom: 2rem;
                width: 100%;
            }

            .suggestions-section {
                padding: 2rem;
            }

            .topics-grid {
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="brand-name">
                <span class="explain">Explai</span><span class="r">r</span>
            </div>
            <p class="subtitle">Transform complex concepts into crystal-clear understanding with AI-powered explanations</p>
        </div>

        {% if not result %}
        <div class="suggestions-section">
            <h3 class="suggestions-title">Popular Topics</h3>
            <div class="topics-grid">
                <div class="topic-card" onclick="fillTopic('Quantum Computing')" tabindex="0" role="button">Quantum Computing</div>
                <div class="topic-card" onclick="fillTopic('Machine Learning')" tabindex="0" role="button">Machine Learning</div>
                <div class="topic-card" onclick="fillTopic('Blockchain')" tabindex="0" role="button">Blockchain</div>
                <div class="topic-card" onclick="fillTopic('Climate Change')" tabindex="0" role="button">Climate Change</div>
                <div class="topic-card" onclick="fillTopic('DNA')" tabindex="0" role="button">DNA</div>
                <div class="topic-card" onclick="fillTopic('Black Holes')" tabindex="0" role="button">Black Holes</div>
                <div class="topic-card" onclick="fillTopic('Cryptocurrency')" tabindex="0" role="button">Cryptocurrency</div>
                <div class="topic-card" onclick="fillTopic('Artificial Intelligence')" tabindex="0" role="button">Artificial Intelligence</div>
            </div>
        </div>
        {% endif %}

        <div class="form-section">
            <form method="post" id="explainForm">
                <div class="input-group">
                    <label class="input-label" for="topic">What would you like to learn about?</label>
                    <input type="text" 
                           id="topic" 
                           name="topic" 
                           class="input-field"
                           placeholder="Enter any topic, concept, or idea..." 
                           required 
                           maxlength="200" 
                           value="{{ request.form.get('topic', '') if request.form.get('topic') else '' }}">
                </div>

                <div class="learning-style-group">
                    <label class="learning-style-label">Choose your learning style:</label>
                    <div class="radio-grid">
                        <label class="radio-card" for="example">
                            <input type="radio" 
                                   id="example"
                                   name="explanation_type" 
                                   value="example" 
                                   {{ 'checked' if request.form.get('explanation_type') == 'example' or not request.form.get('explanation_type') else '' }}>
                            <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M9 11H5a2 2 0 0 0-2 2v3a2 2 0 0 0 2 2h4l3 3V8l-3 3Z"/>
                                <path d="M22 9 12 19l-3-3"/>
                            </svg>
                            <span class="radio-text">Learn through examples</span>
                        </label>
                        <label class="radio-card" for="analogy">
                            <input type="radio" 
                                   id="analogy"
                                   name="explanation_type" 
                                   value="analogy"
                                   {{ 'checked' if request.form.get('explanation_type') == 'analogy' else '' }}>
                            <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M12 3v6l4-4-4-4"/>
                                <path d="M12 21v-6l4 4-4 4"/>
                                <path d="M3 12h6l-4-4 4-4"/>
                                <path d="M21 12h-6l4 4-4 4"/>
                            </svg>
                            <span class="radio-text">Learn through analogies</span>
                        </label>
                    </div>
                </div>

                <button type="submit" class="generate-btn" id="submitBtn">
                    <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8Z"/>
                    </svg>
                    <span>Generate Explanation</span>
                </button>
            </form>
        </div>

        {% if result %}
        <div class="result-section">
            {% if 'Error:' not in result and 'API Error:' not in result %}
            <form method="post" action="/export-pdf">
                <input type="hidden" name="topic" value="{{ request.form.get('topic', '') }}">
                <input type="hidden" name="result" value="{{ result }}">
                <button type="submit" class="export-btn">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <path d="M14 2v6h6"/>
                        <path d="M16 13H8"/>
                        <path d="M16 17H8"/>
                        <path d="M10 9H8"/>
                    </svg>
                    Export PDF
                </button>
            </form>
            <div class="result-content">
                {{ result|safe }}
            </div>
        </div>

        {% if followup_questions and 'Error:' not in result %}
        <div class="suggestions-section">
            <h3 class="suggestions-title">Explore Further</h3>
            <div class="topics-grid">
                {% for question in followup_questions %}
                <div class="topic-card" onclick="fillTopic('{{ question|e }}')" tabindex="0" role="button">
                    {{ question }}
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}

        {% if related_topics and 'Error:' not in result %}
        <div class="suggestions-section">
            <h3 class="suggestions-title">Related Topics</h3>
            <div class="topics-grid">
                {% for topic in related_topics %}
                <div class="topic-card" onclick="fillTopic('{{ topic|e }}')" tabindex="0" role="button">
                    {{ topic }}
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
        {% endif %}
    </div>

    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-content">
            <div class="spinner"></div>
            <p class="loading-text">Crafting your personalized explanation...</p>
        </div>
    </div>

    <script>
        function fillTopic(topic) {
            document.getElementById('topic').value = topic;
            document.getElementById('topic').focus();
            document.querySelector('.form-section').scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        function showLoading() {
            document.getElementById('loadingOverlay').style.display = 'flex';
            document.getElementById('submitBtn').disabled = true;
        }

        // Handle radio card selection visual feedback
        document.querySelectorAll('input[name="explanation_type"]').forEach(radio => {
            radio.addEventListener('change', function() {
                document.querySelectorAll('.radio-card').forEach(card => {
                    card.classList.remove('selected');
                });
                this.closest('.radio-card').classList.add('selected');
            });
        });

        // Initialize selected state on page load
        document.addEventListener('DOMContentLoaded', function() {
            const checkedRadio = document.querySelector('input[name="explanation_type"]:checked');
            if (checkedRadio) {
                checkedRadio.closest('.radio-card').classList.add('selected');
            }

            // Clean up result content
            const resultContent = document.querySelector('.result-content');
            if (resultContent) {
                let content = resultContent.innerHTML;
                content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                content = content.replace(/\n\n/g, '</p><p>');
                content = content.replace(/\n/g, '<br>');
                resultContent.innerHTML = content;
            }
        });

        document.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                if (e.target.classList.contains('topic-card')) {
                    e.preventDefault();
                    e.target.click();
                }
            }
        });

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
    
    if request.method == "POST":
        topic = sanitize_input(request.form.get('topic', ''))
        explanation_type = request.form.get('explanation_type', 'example')
        
        if not topic:
            result = "Error: Please enter a topic to explain."
        elif len(topic.strip()) < 2:
            result = "Error: Topic too short. Please enter a meaningful topic."
        else:
            logger.info(f"Generating explanation for: {topic}")
            result, followup_questions, related_topics = generate_explanation(topic, explanation_type)

    return render_template_string(
        HTML_TEMPLATE, 
        result=result, 
        followup_questions=followup_questions, 
        related_topics=related_topics
    )

@app.route("/export-pdf", methods=["POST"])
def export_pdf():
    topic = sanitize_input(request.form.get('topic', 'Topic'))
    result = request.form.get('result', '')
    
    if not result or 'Error:' in result:
        return "No valid content to export", 400
    
    try:
        # Create PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=1*inch)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
        )
        normal_style = styles['Normal']
        
        # Build content
        story = []
        story.append(Paragraph(f"Explainr: {topic}", title_style))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Process result text
        lines = result.split('\n')
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Escape HTML characters for ReportLab
                line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                current_paragraph.append(line)
            else:
                if current_paragraph:
                    paragraph_text = '<br/>'.join(current_paragraph)
                    story.append(Paragraph(paragraph_text, normal_style))
                    story.append(Spacer(1, 12))
                    current_paragraph = []
        
        if current_paragraph:
            paragraph_text = '<br/>'.join(current_paragraph)
            story.append(Paragraph(paragraph_text, normal_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        # Create response
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        
        # Sanitize filename
        safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', topic.replace(' ', '_'))
        response.headers['Content-Disposition'] = f'attachment; filename=explainr_{safe_filename}.pdf'
        
        logger.info(f"PDF generated for topic: {topic}")
        return response
        
    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        return "Error generating PDF. Please try again.", 500

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

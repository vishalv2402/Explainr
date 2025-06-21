Latest Working Code as of 21-02-2025:
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
    <title>Explainr - Understanding Made Simple</title>
    <meta name="description" content="Understand complex concepts explained at different levels - from beginner to expert">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }
        
        h1 { 
            text-align: center; 
            font-size: 3rem; 
            margin-bottom: 10px;
            background: linear-gradient(45deg, #64748b, #334155);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 40px;
            font-size: 1.1rem;
        }
        
        .form-container {
            background: #f8fafc;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            border: 1px solid #e2e8f0;
        }
        
        label { 
            font-weight: 600; 
            color: #374151; 
            margin-bottom: 10px; 
            display: block;
            font-size: 1.1rem;
        }
        
        input[type=text] { 
            width: 100%; 
            padding: 15px 20px; 
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            font-size: 16px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
            background: white;
        }
        
        input[type=text]:focus {
            outline: none;
            border-color: #64748b;
            box-shadow: 0 0 0 3px rgba(100, 116, 139, 0.1);
            transform: translateY(-2px);
        }
        
        .radio-group { 
            margin: 20px 0; 
            background: white;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
        }
        
        .radio-group > label:first-child { 
            margin-bottom: 15px; 
            font-weight: 600;
            color: #374151;
        }
        
        .radio-group label { 
            display: flex; 
            align-items: center; 
            margin: 10px 0; 
            cursor: pointer;
            padding: 10px;
            border-radius: 8px;
            transition: background 0.2s ease;
        }
        
        .radio-group label:hover {
            background: #f3f4f6;
        }
        
        .radio-group input[type=radio] {
            margin-right: 12px;
            transform: scale(1.2);
            accent-color: #64748b;
        }
        
        button { 
            width: 100%;
            padding: 15px 30px; 
            background: linear-gradient(45deg, #64748b, #475569);
            color: white; 
            border: none; 
            border-radius: 12px; 
            font-size: 18px;
            font-weight: 600;
            cursor: pointer; 
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(100, 116, 139, 0.3);
        }
        
        button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(100, 116, 139, 0.4);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
            color: #64748b;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #64748b;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .result { 
            margin-top: 30px; 
            white-space: pre-wrap; 
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            padding: 30px; 
            border-radius: 15px; 
            border: 1px solid #cbd5e1;
            position: relative;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        }
        
        .result strong {
            color: #1e293b;
            font-size: 1.2rem;
        }
        
        .error {
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border-color: #fca5a5;
            color: #dc2626;
        }
        
        .export-btn {
            position: absolute;
            top: 20px;
            right: 20px;
            width: auto;
            padding: 8px 16px;
            font-size: 14px;
            background: linear-gradient(45deg, #6b7280, #4b5563);
            border-radius: 8px;
        }
        
        .recommendations { 
            margin-top: 25px; 
            background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
            padding: 25px; 
            border-radius: 15px; 
            border: 1px solid #cbd5e1;
        }
        
        .recommendations h3 { 
            margin-bottom: 15px; 
            color: #334155; 
            font-size: 1.3rem;
        }
        
        .topic-suggestion { 
            display: inline-block; 
            background: linear-gradient(45deg, #64748b, #475569);
            color: white; 
            padding: 8px 16px; 
            margin: 5px; 
            border-radius: 20px; 
            text-decoration: none; 
            font-size: 14px; 
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(100, 116, 139, 0.3);
            cursor: pointer;
        }
        
        .topic-suggestion:hover { 
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(100, 116, 139, 0.4);
            color: white;
            text-decoration: none;
        }
        
        .followup-questions { margin-top: 15px; }
        
        .followup-question { 
            display: block; 
            background: rgba(255, 255, 255, 0.8);
            padding: 15px 20px; 
            margin: 10px 0; 
            border-radius: 12px; 
            text-decoration: none; 
            color: #334155; 
            border: 1px solid #cbd5e1;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
            cursor: pointer;
        }
        
        .followup-question:hover { 
            background: rgba(255, 255, 255, 0.95);
            transform: translateX(5px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            color: #334155;
            text-decoration: none;
        }
        
        .suggestions-section { 
            margin-top: 30px; 
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%);
            padding: 25px; 
            border-radius: 15px; 
            border: 1px solid #cbd5e1;
        }
        
        .suggestions-section h3 { 
            margin-bottom: 20px; 
            color: #334155; 
            font-size: 1.3rem;
            text-align: center;
        }
        
        .suggestions-grid {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .form-container, .result, .recommendations, .suggestions-section {
            animation: fadeIn 0.6s ease-out;
        }
        
        .sr-only {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }
        
        @media (max-width: 768px) {
            .container { padding: 20px; }
            h1 { font-size: 2.5rem; }
            input[type=text] { padding: 12px 16px; }
            button { padding: 12px 24px; font-size: 16px; }
            .export-btn { position: static; margin: 10px 0 0 0; width: 100%; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Explainr.AI</h1>
        <p class="subtitle">Understanding complex concepts made simple, one level at a time</p>
        
        {% if not result %}
        <div class="suggestions-section">
            <h3>💡 Popular Topics to Explore</h3>
            <div class="suggestions-grid">
                <span class="topic-suggestion" onclick="fillTopic('Quantum Computing')" tabindex="0" role="button" aria-label="Explore Quantum Computing">Quantum Computing</span>
                <span class="topic-suggestion" onclick="fillTopic('Machine Learning')" tabindex="0" role="button" aria-label="Explore Machine Learning">Machine Learning</span>
                <span class="topic-suggestion" onclick="fillTopic('Blockchain')" tabindex="0" role="button" aria-label="Explore Blockchain">Blockchain</span>
                <span class="topic-suggestion" onclick="fillTopic('Climate Change')" tabindex="0" role="button" aria-label="Explore Climate Change">Climate Change</span>
                <span class="topic-suggestion" onclick="fillTopic('DNA')" tabindex="0" role="button" aria-label="Explore DNA">DNA</span>
                <span class="topic-suggestion" onclick="fillTopic('Black Holes')" tabindex="0" role="button" aria-label="Explore Black Holes">Black Holes</span>
                <span class="topic-suggestion" onclick="fillTopic('Cryptocurrency')" tabindex="0" role="button" aria-label="Explore Cryptocurrency">Cryptocurrency</span>
                <span class="topic-suggestion" onclick="fillTopic('Photosynthesis')" tabindex="0" role="button" aria-label="Explore Photosynthesis">Photosynthesis</span>
                <span class="topic-suggestion" onclick="fillTopic('Artificial Intelligence')" tabindex="0" role="button" aria-label="Explore Artificial Intelligence">Artificial Intelligence</span>
                <span class="topic-suggestion" onclick="fillTopic('The Stock Market')" tabindex="0" role="button" aria-label="Explore Stock Market">Stock Market</span>
            </div>
        </div>
        {% endif %}
        
        <div class="form-container">
            <form method="post" id="explainForm" onsubmit="showLoading()">
                <label for="topic">What do you want explained?</label>
                <input type="text" id="topic" name="topic" placeholder="Enter any topic you're curious about..." required maxlength="200" 
                       value="{{ request.form.get('topic', '') if request.form.get('topic') else '' }}">
                
                <div class="radio-group">
                    <label>Choose your explanation style:</label>
                    <label><input type="radio" name="explanation_type" value="example" 
                           {{ 'checked' if request.form.get('explanation_type') == 'example' or not request.form.get('explanation_type') else '' }}> 
                           ✨ Explain with examples</label>
                    <label><input type="radio" name="explanation_type" value="analogy"
                           {{ 'checked' if request.form.get('explanation_type') == 'analogy' else '' }}> 
                           🔗 Explain with analogies</label>
                </div>
                
                <button type="submit" id="submitBtn">🚀 Explain This Topic</button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Generating your explanation...</p>
            </div>
        </div>

        {% if result %}
        <div class="result {% if 'Error:' in result or 'API Error:' in result %}error{% endif %}">
            {% if 'Error:' not in result and 'API Error:' not in result %}
            <strong>📚 Explained in 3 levels:</strong>
            <form method="post" action="/export-pdf" style="display: inline;">
                <input type="hidden" name="topic" value="{{ request.form.get('topic', '') }}">
                <input type="hidden" name="result" value="{{ result }}">
                <button type="submit" class="export-btn">📄 Export PDF</button>
            </form>
            <br><br>
            {% endif %}
            {{ result }}
        </div>
        
        {% if followup_questions and 'Error:' not in result %}
        <div class="recommendations">
            <h3>🤔 Follow-up Questions</h3>
            <div class="followup-questions">
                {% for question in followup_questions %}
                <span class="followup-question" onclick="fillTopic('{{ question|e }}')" tabindex="0" role="button" 
                      aria-label="Ask: {{ question|e }}">❓ {{ question }}</span>
                {% endfor %}
            </div>
        </div>
        {% endif %}
        
        {% if related_topics and 'Error:' not in result %}
        <div class="recommendations">
            <h3>🔗 Related Topics to Explore</h3>
            <div class="suggestions-grid">
                {% for topic in related_topics %}
                <span class="topic-suggestion" onclick="fillTopic('{{ topic|e }}')" tabindex="0" role="button" 
                      aria-label="Explore {{ topic|e }}">{{ topic }}</span>
                {% endfor %}
            </div>
        </div>
        {% endif %}
        {% endif %}
    </div>
    
    <script>
        function fillTopic(topic) {
            document.getElementById('topic').value = topic;
            document.getElementById('topic').focus();
            // Scroll to form
            document.getElementById('explainForm').scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('submitBtn').disabled = true;
            document.getElementById('submitBtn').textContent = 'Generating...';
        }
        
        // Add keyboard support for interactive elements
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                if (e.target.classList.contains('topic-suggestion') || e.target.classList.contains('followup-question')) {
                    e.preventDefault();
                    e.target.click();
                }
            }
        });
        
        // Form validation
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

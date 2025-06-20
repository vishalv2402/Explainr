from flask import Flask, request, render_template_string, make_response
import openai
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO

app = Flask(__name__)

# 🔑 Replace this with your actual OpenAI API key
openai.api_key = os.environ.get('ExplainrOpenAIKey', 'your-api-key-here')

# 🧠 HTML form + response display (inline)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Explainr</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
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
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(100, 116, 139, 0.4);
        }
        
        button:active {
            transform: translateY(0);
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
        }
        
        .topic-suggestion:hover { 
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(100, 116, 139, 0.4);
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
        }
        
        .followup-question:hover { 
            background: rgba(255, 255, 255, 0.95);
            transform: translateX(5px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
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
                <a href="#" class="topic-suggestion" onclick="fillTopic('Quantum Computing')">Quantum Computing</a>
                <a href="#" class="topic-suggestion" onclick="fillTopic('Machine Learning')">Machine Learning</a>
                <a href="#" class="topic-suggestion" onclick="fillTopic('Blockchain')">Blockchain</a>
                <a href="#" class="topic-suggestion" onclick="fillTopic('Climate Change')">Climate Change</a>
                <a href="#" class="topic-suggestion" onclick="fillTopic('DNA')">DNA</a>
                <a href="#" class="topic-suggestion" onclick="fillTopic('Black Holes')">Black Holes</a>
                <a href="#" class="topic-suggestion" onclick="fillTopic('Cryptocurrency')">Cryptocurrency</a>
                <a href="#" class="topic-suggestion" onclick="fillTopic('Photosynthesis')">Photosynthesis</a>
                <a href="#" class="topic-suggestion" onclick="fillTopic('Artificial Intelligence')">Artificial Intelligence</a>
                <a href="#" class="topic-suggestion" onclick="fillTopic('The Stock Market')">Stock Market</a>
            </div>
        </div>
        {% endif %}
        
        <div class="form-container">
            <form method="post" id="explainForm">
                <label for="topic">What do you want explained?</label>
                <input type="text" id="topic" name="topic" placeholder="Enter any topic you're curious about..." required>
                
                <div class="radio-group">
                    <label>Choose your explanation style:</label>
                    <label><input type="radio" name="explanation_type" value="example" checked> ✨ Explain with examples</label>
                    <label><input type="radio" name="explanation_type" value="analogy"> 🔗 Explain with analogies</label>
                </div>
                
                <button type="submit">🚀 Explain This Topic</button>
            </form>
        </div>

        {% if result %}
        <div class="result">
            <strong>📚 Explained in 3 levels:</strong>
            <form method="post" action="/export-pdf">
                <input type="hidden" name="topic" value="{{ request.form.get('topic', '') }}">
                <input type="hidden" name="result" value="{{ result }}">
                <button type="submit" class="export-btn">📄 Export PDF</button>
            </form>
            <br><br>
            {{ result }}
        </div>
        
        {% if followup_questions %}
        <div class="recommendations">
            <h3>🤔 Follow-up Questions</h3>
            <div class="followup-questions">
                {% for question in followup_questions %}
                <a href="#" class="followup-question" onclick="fillTopic('{{ question }}')">❓ {{ question }}</a>
                {% endfor %}
            </div>
        </div>
        {% endif %}
        
        {% if related_topics %}
        <div class="recommendations">
            <h3>🔗 Related Topics to Explore</h3>
            <div class="suggestions-grid">
                {% for topic in related_topics %}
                <a href="#" class="topic-suggestion" onclick="fillTopic('{{ topic }}')">{{ topic }}</a>
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
        }
    </script>
</body>
</html>
'''


# 🛠️ Flask route
@app.route("/", methods=["GET", "POST"])
def explain():
    result = None
    followup_questions = None
    related_topics = None
    
    if request.method == "POST":
        topic = request.form['topic']
        explanation_type = request.form['explanation_type']
        
        if explanation_type == "example":
            style_instruction = "Add a clear example for each level."
        else:  # analogy
            style_instruction = "Add an analogy for each level."
        
        prompt = f"""
        Explain {topic} in 3 levels:
        1. Like I'm 5
        2. Like I'm 15
        3. Like I'm 30
        {style_instruction}
        """

        try:
            # Get main explanation
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a friendly explainer bot."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ])
            result = response['choices'][0]['message']['content'].strip()
            
            # Generate follow-up questions
            followup_prompt = f"Generate 3 thoughtful follow-up questions that someone might ask after learning about {topic}. Return only the questions, one per line."
            followup_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You generate relevant follow-up questions."
                    },
                    {
                        "role": "user",
                        "content": followup_prompt
                    },
                ])
            followup_questions = [q.strip() for q in followup_response['choices'][0]['message']['content'].strip().split('\n') if q.strip()]
            
            # Generate related topics
            related_prompt = f"List 5 topics closely related to {topic} that would be interesting to explore. Return only the topic names, one per line."
            related_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You suggest related topics for exploration."
                    },
                    {
                        "role": "user",
                        "content": related_prompt
                    },
                ])
            related_topics = [t.strip() for t in related_response['choices'][0]['message']['content'].strip().split('\n') if t.strip()]
            
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template_string(HTML_TEMPLATE, result=result, followup_questions=followup_questions, related_topics=related_topics)


@app.route("/export-pdf", methods=["POST"])
def export_pdf():
    topic = request.form.get('topic', 'Topic')
    result = request.form.get('result', '')
    
    if not result:
        return "No content to export", 400
    
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
    story.append(Spacer(1, 12))
    
    # Split result into lines and process them properly
    lines = result.split('\n')
    current_paragraph = []
    
    for line in lines:
        if line.strip():  # Non-empty line
            current_paragraph.append(line.strip())
        else:  # Empty line - end current paragraph
            if current_paragraph:
                # Join lines with <br/> tags for line breaks within paragraph
                paragraph_text = '<br/>'.join(current_paragraph)
                story.append(Paragraph(paragraph_text, normal_style))
                story.append(Spacer(1, 12))
                current_paragraph = []
    
    # Add any remaining paragraph
    if current_paragraph:
        paragraph_text = '<br/>'.join(current_paragraph)
        story.append(Paragraph(paragraph_text, normal_style))
        story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    # Create response
    response = make_response(buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=explainr_{topic.replace(" ", "_")}.pdf'
    
    return response


# ✅ Run the app on Replit
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=81)
Add main.py

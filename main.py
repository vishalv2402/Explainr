from flask import Flask, request, render_template_string
import openai
import os
import sqlite3
from datetime import datetime

# Initialize Flask
app = Flask(__name__)

# Set your OpenAI API key from environment variable
openai.api_key = os.environ.get('OPENAI_API_KEY', 'YOUR_OPENAI_API_KEY')

# Database setup (SQLite file-based)
DB_PATH = 'search_history.db'
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
# Create table if it doesn't exist
cursor.execute(
    '''CREATE TABLE IF NOT EXISTS searches (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           topic TEXT NOT NULL,
           result TEXT NOT NULL,
           created_at TIMESTAMP NOT NULL
       )'''
)
conn.commit()

# HTML template with history section
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Explainr</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 40px; }
        input[type=text] { width: 300px; padding: 10px; }
        button { padding: 10px 20px; background-color: #4F46E5; color: white; border: none; border-radius: 6px; }
        .result { margin-top: 30px; white-space: pre-wrap; background: #f4f4f4; padding: 20px; border-radius: 10px; }
        .history { margin-top: 50px; }
        .history-item { margin-bottom: 15px; }
        .history-item strong { display: block; }
    </style>
</head>
<body>
    <h1>🧠 Explainr</h1>
    <form method="post">
        <label for="topic">What do you want explained?</label><br><br>
        <input type="text" id="topic" name="topic" required>
        <button type="submit">Explain</button>
    </form>

    {% if result %}
    <div class="result">
        <strong>Explained in 3 levels:</strong><br><br>
        {{ result }}
    </div>
    {% endif %}

    {% if history %}
    <div class="history">
        <h2>🔄 Recent Topics</h2>
        {% for item in history %}
        <div class="history-item">
            <strong>{{ item['topic'] }}:</strong>
            <div style="white-space: pre-wrap; background: #eef; padding: 10px; border-radius: 6px;">{{ item['result'] }}</div>
        </div>
        {% endfor %}
    </div>
    {% endif %}
</body>
</html>
'''

@app.route("/", methods=["GET", "POST"])
def explain():
    result = None
    # On form submit
    if request.method == "POST":
        topic = request.form['topic']
        prompt = f"""
Explain {topic} in 3 levels:
1. Like I'm 5
2. Like I'm 15
3. Like I'm 30
"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a friendly explainer bot."},
                    {"role": "user", "content": prompt},
                ]
            )
            result = response.choices[0].message.content.strip()
            # Save to history
            cursor.execute(
                "INSERT INTO searches (topic, result, created_at) VALUES (?, ?, ?)"
                , (topic, result, datetime.utcnow())
            )
            conn.commit()
        except Exception as e:
            result = f"Error: {str(e)}"

    # Fetch last 5 history items
    cursor.execute(
        "SELECT topic, result FROM searches ORDER BY id DESC LIMIT 5"
    )
    rows = cursor.fetchall()
    history = [{'topic': row[0], 'result': row[1]} for row in rows]

    return render_template_string(HTML_TEMPLATE, result=result, history=history)

if __name__ == "__main__":
    # Use port from environment or default 81
    port = int(os.environ.get('PORT', 81))
    app.run(host='0.0.0.0', port=port)

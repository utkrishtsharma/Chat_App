[ec2-user@ip-172-31-21-125 aayusathi_app]$ cat app.py
import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from functools import wraps
from flask import Flask, render_template, request, jsonify, session
from groq import Groq
#import os
os.environ["GROQ_API_KEY"] = "key_goes_here"

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))  # Secure secret key for sessions

# Initialize Groq API client
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")

class MedicalResponseFormatter:
    """Formats AI responses for better readability and medical safety."""
    def __init__(self):
        self.emergency_keywords = [
            'chest pain', 'difficulty breathing', 'severe bleeding', 'unconscious',
            'heart attack', 'stroke', 'poisoning', 'severe allergic reaction',
            'suicide', 'overdose', 'choking', 'severe burns'
        ]
        self.disclaimer_text = (
            "⚠️ **Medical Disclaimer**: This information is for educational purposes only. "
            "Always consult qualified healthcare professionals for personalized medical advice, "
            "diagnosis, or treatment decisions."
        )

    def format_medical_response(self, raw_response: str, user_query: str) -> str:
        if self._is_emergency_query(user_query):
            return self._format_emergency_response(raw_response)
        formatted_response = self._structure_response(raw_response)
        formatted_response += f"\n\n---\n{self.disclaimer_text}"
        return formatted_response

    def _is_emergency_query(self, query: str) -> bool:
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.emergency_keywords)

    def _format_emergency_response(self, response: str) -> str:
        emergency_header = (
            "🚨 **URGENT MEDICAL ATTENTION MAY BE NEEDED** 🚨\n\n"
            "If this is a medical emergency, please:\n"
            "• **Call emergency services immediately** (911 in US, 112 in EU)\n"
            "• Go to the nearest emergency room\n"
            "• Contact your healthcare provider urgently\n\n"
            "---\n\n"
        )
        structured_response = self._structure_response(response)
        return f"{emergency_header}{structured_response}\n\n---\n{self.disclaimer_text}"

    def _structure_response(self, response: str) -> str:
        response = response.strip()
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        structured_parts = []
        current_section = []
        
        for paragraph in paragraphs:
            if (len(paragraph) < 100 and 
                (paragraph.endswith(':') or 
                 any(keyword in paragraph.lower() for keyword in 
                     ['causes', 'symptoms', 'treatment', 'prevention', 'when to seek']))):
                if current_section:
                    structured_parts.append('\n'.join(current_section))
                    current_section = []
                structured_parts.append(f"## {paragraph.rstrip(':')}")
            else:
                formatted_paragraph = self._format_paragraph(paragraph)
                current_section.append(formatted_paragraph)
        
        if current_section:
            structured_parts.append('\n'.join(current_section))
        return '\n\n'.join(structured_parts)

    def _format_paragraph(self, paragraph: str) -> str:
        if paragraph.strip().startswith(('1.', '- ', '* ')):
            lines = paragraph.split('\n')
            formatted_lines = [f"• {line.strip()[2:]}" if line.strip() else line for line in lines]
            return '\n'.join(formatted_lines)
        return paragraph

class GroqMedicalChatbot:
    """Stateless medical chatbot using Groq API."""
    def __init__(self, api_key: str, model_name: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.medical_context = self._initialize_medical_context()
        self.formatter = MedicalResponseFormatter()

    def _initialize_medical_context(self) -> str:
        return """You are a knowledgeable medical AI assistant designed to provide educational health information. 

IMPORTANT GUIDELINES:
- Provide general health information only
- Always recommend consulting healthcare professionals
- Structure responses with clear headings and bullet points
- Use markdown for formatting (## for headings, • for bullets)

RESPONSE STRUCTURE:
1. Brief overview
2. Common causes (if applicable)
3. Typical symptoms
4. General management approaches
5. When to seek professional care"""

    def generate_medical_response(self, user_query: str, conversation_history: List[Dict]) -> str:
        """Generate a medical response using Groq API."""
        messages = [{"role": "system", "content": self.medical_context}]
        # Filter conversation history to include only 'role' and 'content'
        filtered_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in conversation_history
        ]
        messages.extend(filtered_history)  # Full history including latest user message
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.3,
                top_p=0.9,
                stream=False
            )
            raw_response = response.choices[0].message.content
            return self.formatter.format_medical_response(raw_response, user_query)
        except Exception as e:
            return f"Error: Unable to process request - {str(e)}\n\n{self.formatter.disclaimer_text}"

# Initialize chatbot
medical_bot = GroqMedicalChatbot(api_key=GROQ_API_KEY)

def ensure_user_profile():
    """Decorator to ensure user profile exists in session."""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if 'user_profile' not in session:
                session['user_profile'] = {'name': 'Guest', 'chats': {}}
            return f(*args, **kwargs)
        return wrapped
    return decorator

@app.route('/')
@ensure_user_profile()
def index():
    """Render the main page."""
    return render_template('index.html', user_profile=session['user_profile'])

@app.route('/api/save_profile', methods=['POST'])
def save_profile():
    """Save user profile information."""
    data = request.json
    if 'name' in data:
        if 'user_profile' not in session:
            session['user_profile'] = {'name': data['name'], 'chats': {}}
        else:
            session['user_profile']['name'] = data['name']
        session.modified = True
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Name not provided'}), 400

@app.route('/api/new_chat', methods=['POST'])
@ensure_user_profile()
def new_chat():
    """Create a new chat conversation."""
    chat_id = str(uuid.uuid4())
    session['user_profile']['chats'][chat_id] = {
        'title': 'New Chat',
        'created_at': datetime.now().isoformat(),
        'messages': []
    }
    session.modified = True
    return jsonify({
        'success': True,
        'chat_id': chat_id,
        'chat': session['user_profile']['chats'][chat_id]
    })

@app.route('/api/get_chats', methods=['GET'])
@ensure_user_profile()
def get_chats():
    """Get all user chats."""
    return jsonify({
        'success': True,
        'chats': session['user_profile']['chats']
    })

@app.route('/api/delete_chat', methods=['POST'])
@ensure_user_profile()
def delete_chat():
    """Delete a specific chat conversation."""
    data = request.json
    chat_id = data.get('chat_id')
    if chat_id and chat_id in session['user_profile']['chats']:
        del session['user_profile']['chats'][chat_id]
        session.modified = True
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Chat not found'}), 404

@app.route('/api/chat', methods=['POST'])
@ensure_user_profile()
def chat():
    """Process a chat message using Groq API."""
    data = request.json
    message = data.get('message', '').strip()
    chat_id = data.get('chat_id')
    
    if not message:
        return jsonify({'success': False, 'error': 'Message is required'}), 400
    if not chat_id or chat_id not in session['user_profile']['chats']:
        return jsonify({'success': False, 'error': 'Invalid chat ID'}), 400
    
    chat = session['user_profile']['chats'][chat_id]
    if not chat['messages']:
        chat['title'] = message[:30] + ('...' if len(message) > 30 else '')
    
    user_message = {
        'role': 'user',
        'content': message,
        'timestamp': datetime.now().isoformat()
    }
    chat['messages'].append(user_message)
    
    response_text = medical_bot.generate_medical_response(message, chat['messages'])
    
    assistant_message = {
        'role': 'assistant',
        'content': response_text,
        'sources': [],  # Groq doesn't provide sources
        'timestamp': datetime.now().isoformat()
    }
    chat['messages'].append(assistant_message)
    session.modified = True
    
    return jsonify({
        'success': True,
        'response': response_text,
        'sources': []
    })

@app.route('/api/get_chat', methods=['GET'])
@ensure_user_profile()
def get_chat():
    """Get a specific chat conversation."""
    chat_id = request.args.get('chat_id')
    if chat_id and chat_id in session['user_profile']['chats']:
        return jsonify({
            'success': True,
            'chat': session['user_profile']['chats'][chat_id]
        })
    return jsonify({'success': False, 'error': 'Chat not found'}), 404

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true')

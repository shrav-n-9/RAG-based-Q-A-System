from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json
import subprocess
from retrieval_utils import Retriever
import rag  # Import the RAGQA class from rag.py

app = Flask(__name__, static_folder='front-end', template_folder='front-end')

# Configuration
MODEL_FILE = 'llama-3-8b-instruct-q4.gguf'
LLAMA_FILE_EXE = 'llamafile.exe'
TOP_K = 4
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.5

# Initialize RAGQA instance globally
try:
    retriever = Retriever()
    rag_instance = rag.RAGQA(retriever)
except FileNotFoundError as e:
    print(f"Error initializing RAGQA: {e}")
    exit(1)

@app.route('/')
def index():
    return send_from_directory('front-end', 'index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('front-end', path)

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get('question', '').strip()
    if not query:
        return jsonify({"error": "No question provided"}), 400

    try:
        response = rag_instance.generate_answer(query)
        return jsonify({
            "question": response["question"],
            "answer": response["answer"],
            "sources": response["sources"],
            "retrieval_count": response["retrieval_count"]
        })
    except Exception as e:
        return jsonify({"error": f"Error processing question: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
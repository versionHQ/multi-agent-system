import json
import logging
import os
from typing import Any, Dict
from dotenv import load_dotenv
from flask import Flask, request, jsonify, flash, redirect
from flask_cors import CORS, cross_origin
from waitress import serve

load_dotenv(override=True)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER, ALLOWED_EXTENSIONS = "uploads", {"pdf", "docx", "csv", "xls"}
app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["CORS_HEADERS"] = "Content-Type"
logging.getLogger("flask_cors").level = logging.DEBUG


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def hello_world():
    return """
    <p>Hello, world</p>
    <p>I am Version HQ's Flask backend.</p>
    <p>The orchestration framework is <a href="https://github.com/versionHQ/multi-agent-system" target="_blank" rel="noreferrer">here</a>.</p>
    <p>You can check our current bata from <a href="https://versi0n.io" target="_blank" rel="noreferrer">here</a>.</p>
    """


@app.route('/api/draft-instruction', methods=['POST', 'OPTIONS', 'GET'])
@cross_origin(origin='*', headers=['Access-Control-Allow-Origin'])
def draft_instruction():
    from core.main import draft_instruction, read_url
    data = request.json
    url, goal = data.get('url', None), data.get('goal', None)
    if not url:
        return jsonify({ "output": res }), 400
    
    html_source_code = read_url(url=url)
    try:
        res = draft_instruction(html_source_code=html_source_code, goal=goal)
        return jsonify({ "output": res }), 200
    except:
        return jsonify({ "output":  None }), 400


@app.route('/api/assess', methods=['POST', 'OPTIONS', 'GET'])
@cross_origin(origin='*', headers=['Access-Control-Allow-Origin'])
def run_initial_assessment():
    from core.main import assess
    try:
        res = assess()
        return  jsonify({ "output": res }), 200
    except:
        return jsonify({ "output":  None }), 400
    


if __name__ == "__main__":
    print("...start the operation...")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    serve(app, host="0.0.0.0", port=5002)

import json
import logging
import os
from typing import Any, Dict
from dotenv import load_dotenv
from flask import Flask, request, jsonify, flash, redirect
from flask_cors import CORS, cross_origin
from waitress import serve

from framework.task.model import TaskOutput

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
    return "<p>Hello, World!</p>"


@app.route('/api/base-setting', methods=['POST', 'OPTIONS', 'GET'])
@cross_origin(origin='*', headers=['Access-Control-Allow-Origin',])
def test2():
    print("test2")
    from sample.test import test_2
    res = test_2()
    return  jsonify({ "output": res }), 200


if __name__ == "__main__":
    print("...start the operation...")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    serve(app, host="0.0.0.0", port=5002)

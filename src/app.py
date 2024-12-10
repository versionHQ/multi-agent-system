import json
import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS
from waitress import serve

from src.components.task.model import TaskOutput

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


@app.route("/define-cohort")
def define_cohort(customer=Dict[str, str], client_input=Dict[str, Any]) -> TaskOutput:
    from components.task.model import ResponseField, Task
    from project.agents import cohort_analyst
    from project.tasks import TASK_DESCRIPTION_TEMPLATE

    # from project.tools import rag_tools

    task_description = TASK_DESCRIPTION_TEMPLATE.format(
        customer_data=json.dumps(customer),
        client_business=client_input.business_overview,
        target_audience=client_input.target_audience,
        instruction="""Analyze the client's business model, target audience, and customer information and define the optimal cohort timeframe based on customer lifecycle and product usage patterns.""",
        output="""Select three relevant KPIs to measure the success of the cohort-based messaging workflow and prioritize these KPIs based on their alignment with the client's business objectives and Set benchmarks for success and track the performance of the cohort-based messaging workflow.""",
    )
    task = Task(
        description=task_description,
        expected_output_raw=False,
        expected_output_json=True,
        output_json_field_list=[
            ResponseField(title="cohort_timeframe", type="int", required=True),
            ResponseField(title="kpi", type="array", required=True),
        ],
        expected_output_pydantic=True,
        context=[],
        tools=[],
        callback=None,
    )

    res = task.execute_sync(agent=cohort_analyst)  # -> TaskOutput class
    print("Task is completed. ID:", res.task_id)
    print("Raw result: ", res.raw)
    print("JSON: ", res.json_dict)
    print("Pydantic: ", res.pydantic)

    return res


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    serve(app, host="0.0.0.0", port=5002)

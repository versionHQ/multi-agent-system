# Overview

[![PyPI version](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)

A framework for orchestration and multi-agent system that design, deploy, and autopilot messaging workflows based on performance.

Agents are model agnostic.

Messaging workflows are created at individual level, and will be deployed on third-party services via `Composio`.

Visit:

- [Flask backend](https://blank-minta-krik8235-b534ee83.koyeb.app/)
- [Client interface (beta)](https://versi0n.io/)
- [Landing page](https://home.versi0n.io)
- [Github (client app)](https://github.com/krik8235/pj_m_dev)


## Mindmap

LLM-powered `agent`s and `team`s use `tool`s and their own knowledge to complete the `task` given by the client or the system.

<p align="center">
   <img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1733556715/pj_m_home/urwte15at3h0dr8mdlyo.png" alt="mindmap" width="1000">
</p>


## UI
<p align="center">
   <!-- <img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1728219760/pj_m_home/wkxvmmyah5dvhbdah1qn.png" alt="version logo" width="200">
   &nbsp; &nbsp; &nbsp; &nbsp; -->
   <img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1728302420/pj_m_home/xy58a7imyquuvkgukqxt.png" width="45%" alt="messaging workflow" />
</p>


<p align="center">
   <!-- <img alt="UI1" src="https://res.cloudinary.com/dfeirxlea/image/upload/v1732801909/pj_m_home/ibi8b1ohv2sioqdkoq7n.png" width="45%">
   &nbsp; &nbsp; -->
   <img alt="UI2" src="https://res.cloudinary.com/dfeirxlea/image/upload/v1733414200/pj_m_home/tqgg3xfpk5x4i6rh3egv.png" width="80%">
</p>


<hr />

## Table of Content
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing & Customizing](#contributing--customizing)
  - [Customizing AI Agents](#customizing-ai-agents)
  - [Modifying RAG Functionality](#modifying-rag-functionality)
  - [Package Management with uv](#package-management-with-uv)
  - [CLI Test](#cli-test)
  - [Pre-Commit Hooks](#pre-commit-hooks)
- [Trouble Shooting](#trouble-shooting)
- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
- [Overall Project Structure](#overall-project-structure)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


<hr />

## Key Features

A mulit-agent system that tailors messaging workflow, predicts its performance, and deploys it on third-party tools.

The `agent` is model agnostic. The default model is set Chat GTP 4o. We ask the client their preference and switch it accordingly using llm variable stored in the `BaseAgent` class.

Multiple `agent`s can form a `team` to complete complex tasks together.


1. **Client & Customer Analysis**:
   - Fetch clustering analysis results of each customer.
   - Retrieve outbound campaign conditions such as objective, media mix, target audience from the React client interface.
   - Employ default conditions in case no client inputs are available.

2. **Assumption Building**:
   - Call `agent-a` and let it build assumptions we will test on each user to achieve the goal.
   - Ask the client for the approval.

3. **Workflow Tailoring**:
   - Call `agent-b` and let it draft messaging workflow with 3 conditional layers.
   - Call `agent-c` and let it set up media channels and schedule based on the client inputs on audience and media mix.
   - Call `agent-d` and let it draft a first message to send.

   (They form a `team`.)

   [*brain*]
   - Within each agent, user data (jsonl) is passed to the RAGTool function in Composio.
   - This RAG system queries a Chroma DB that has been pre-trained on a repository of past outbound campaigns.
   - Utilizes LLM to generate the output on next best action, comparing with past similar campaigns in the Chroma DB.

4. **Performance Predicting**
   - Call `agent-x` from the versionHQ module, and estimate performance of the first layer of the workflow.

5. **Client Interaction**:
   - Presents the workflow, next best action, and performance prediction to the client through a React client interface.
   - Allows the user to choose whether to accept or ask for change on the action recommended.

6. **Finalizing Workflow**:
   - Collects all client feedback.
   - Passes the information back to the relative agent/s to recreate.

7. **Deploying Next Action**:
   - Use Composio to deploy on the CRM or ad platform.



## Technologies Used
**Schema, Database**
   - [Pydantic](https://docs.pydantic.dev/latest/): Data validation and serialization library for Python
   - [Pydantic_core](https://pypi.org/project/pydantic-core/): Core func packages for Pydantic
   - [Chroma DB](https://docs.trychroma.com/): Vector database for storing and querying usage data
   - [SQLite](https://www.sqlite.org/docs.html): C-language library to implements a small SQL database engine
   - [Upstage](https://console.upstage.ai/docs/getting-started/overview): Document processer for ML tasks. (Use `Document Parser API` to extract data from documents)

**LLM-curation**
   - OpenAI GPT-4: Advanced language model for analysis and recommendations
   - [LiteLLM](https://docs.litellm.ai/docs/providers): Curation platform to access LLMs

**Tools**
   - [Composio](https://composio.dev/): Conect RAG agents with external tools, Apps, and APIs to perform actions and receive triggers. We use [tools](https://composio.dev/tools) and [RAG tools](https://app.composio.dev/app/ragtool) from Composio toolset.

**Deployment**
   - Python: Primary programming language. We use 3.12 in this project
   - [Flask](https://flask.palletsprojects.com/en/stable/quickstart/): Web framework for the backend API. Communicate with the client app.
   - [Flask Cors](https://pypi.org/project/Flask-Cors/): A Flask extension for handling Cross Origin Resource Sharing (CORS), making cross-origin AJAX possible
   - [uv](https://docs.astral.sh/uv/): Python package installer and resolver
   - [pre-commit](https://pre-commit.com/): Manage and maintain pre-commit hooks
   - [Hatch/Hatchling](https://hatch.pypa.io/latest/): Build package management
   - [Koyeb](https://www.koyeb.com/docs): Serverless deployment platform


## Project Structure

```
.
framework/                  # Orchestration frameworks on Pydantic
│   ├── agent/
│   └── llm/
│   └── task/
│   └── team/
│   └── tool/
│   └── clients/            # Classes to store the client related information
│   └── _cli/               # CLI commands
│   └── ...
│
├── db/                     # Database files
│   ├── chroma.sqlite3
│   └── ...
│
sample/
├── __init__.py
├── project/                # Store a sample project
├── uploads/                # Uploaded files for the project
│
└── app.py                  # Flask application
```

## Setup

1. Install the `uv` package manager:
   ```
   pip install uv
   ```

2. Install dependencies:
   ```
   uv venv --python=python3.12
   source .venv/bin/activate

   uv pip install -r requirements.txt -v
   ```

3. Set up environment variables:
   Create a `.env` file in the project root and add the following:
   ```
   OPENAI_API_KEY=your-openai-api-key
   LITELLM_API_KEY=your-litellm-api-key
   UPSTAGE_API_KEY=your-upstage-api-key
   COMPOSIO_API_KEY=your-composio-api-key
   COMPOSIO_CLI_KEY=your-composio-cli-key
   ```

## Usage

1. Start the Flask backend:
   ```
   uv run python app.py
   ```
   The backend will be available at `http://localhost:5002`.

2. Add new API endpoint. You can call them from the [api base url](https://blank-minta-krik8235-b534ee83.koyeb.app/).

3. Frontend (production) is available at `https://versi0n.io`. Currently we are on beta.


## Contributing & Customizing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-amazing-feature`)
3. Pull the latest version of source code from the main branch (`git pull origin main`) *Address conflicts if any.
4. Commit your changes (`git add .` / `git commit -m 'Add your-amazing-feature'`)
5. Push to the branch (`git push origin feature/your-amazing-feature`)
6. Open a pull request

0. Flag with `#! REFINEME` for any improvements and `#! FIXME` for any errors.


### Customizing AI Agents

To add an agent, use `sample` directory to add new `project`. You can define an agent with a specific role, goal, and set of tools.

Your new agent needs to follow the `Agent` model defined in the `framework.agent.model.py`.

You can also add any fields and functions to the `Agent` model **universally** by modifying `framework.agent.model.py`.


### Modifying RAG Functionality

The RAG system uses Chroma DB to store and query past campaign dataset. To update the knowledge base:

1. Add new files to the `uploads/` directory. (This will not be pushed to Github.)
2. Modify the `tools.py` file to update the ingestion process if necessary.
3. Run the ingestion process to update the Chroma DB.


### Package Management with uv

- Add a package: `uv add <package>`
- Remove a package: `uv remove <package>`
- Run a command in the virtual environment: `uv run <command>`

* After updating dependencies, update `requirements.txt` accordingly or run `uv pip freeze > requirements.txt`

### CLI Test

- To test new features on CLI, import the features to `framework._cli.test_cli.py` as `test` and run the following command:
   ```
   uv run test
   ```

### Pre-Commit Hooks

1. Install pre-commit hooks:
   ```
   uv run pre-commit install
   ```

2. Run pre-commit checks manually:
   ```
   uv run pre-commit run --all-files
   ```

Pre-commit hooks help maintain code quality by running checks for formatting, linting, and other issues before each commit.

* To skip pre-commit hooks (NOT RECOMMENDED)
   ```
   git commit --no-verify -m "your-commit-message"
   ```

## Trouble Shooting

Common issues and solutions:
- API key errors: Ensure all API keys in the `.env` file are correct and up to date. Make sure to add `load_dotenv()` on the top of the python file to apply the latest environment values.
- Database connection issues: Check if the Chroma DB is properly initialized and accessible.
- Memory errors: If processing large contracts, you may need to increase the available memory for the Python process.
- Issues related to dependencies: Delete the `uv.lock` file and `.venv` and run `uv pip install -r requirements.txt -v`.
- Issues related to the AI agents or RAG system: Check the `output.log` file for detailed error messages and stack traces.
- Issues related to `Python quit unexpectedly`: Check [this stackoverflow article](https://stackoverflow.com/questions/59888499/macos-catalina-python-quit-unexpectedly-error).


## Frequently Asked Questions (FAQ)
**Q. Where can I see if the agent is working?**

> A. You can find a frontend app [here](https://beta.versi0n.io) with real-world outbound use cases. Also refer to the structure.

**Q. How do you analyze the customer?**

> A. We employ soft clustering for each customer.
> <img width="200" src="https://res.cloudinary.com/dfeirxlea/image/upload/v1732732628/pj_m_agents/ito937s5d5x0so8isvw6.png">


**Q. When should I use a team vs an agent?**

> A. In essence, use a team for intricate, evolving projects, and agents for quick, straightforward tasks.

> Use a team when:

> **Complex tasks**: You need to complete multiple, interconnected tasks that require sequential or hierarchical processing.

> **Iterative refinement**: You want to iteratively improve upon the output through multiple rounds of feedback and revision.

> Use an agent when:

> **Simple tasks**: You have a straightforward, one-off task that doesn't require significant complexity or iteration.

> **Human input**: You need to provide initial input or guidance to the agent, or you expect to review and refine the output.



## Overall Project Structure

|  | Marketing page | Client app (beta) | Orchestration/agent system | Analyics
| :---: | :---: | :---: | :---: | :---: |
| Github |  [repo_1](https://github.com/krik8235/pj_m_dev_home)  | [repo_a](https://github.com/krik8235/pj_m_dev) | **this repository** | [repo_b](https://github.com/versionHQ/clutering-analysis) |
| Website | [home](https://home.versi0n.io) | [client app](https://versi0n.io) | - | - |


<--- Remaining basic tasks --->

- llm handling - agent
- more llms integration
- simpler prompting
- team leader (do we need it?)
- tools (test)
- broader knowledge

- utils - log
- utils - time

- end to end client app test

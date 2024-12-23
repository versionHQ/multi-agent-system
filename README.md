# Overview

![MIT license](https://img.shields.io/badge/License-MIT-green) ![PyPi](https://img.shields.io/badge/pypi-v1.1.6.3-blue)
![python ver](https://img.shields.io/badge/Python-3.13.1-purple) ![pyenv ver](https://img.shields.io/badge/pyenv-2.4.23-orange)


An LLM orchestration frameworks for multi-agent systems with RAG to autopilot outbound workflows.

Agents are model agnostic.

Messaging workflows are created at individual level, and will be deployed on third-party services via `Composio`.


**Visit:**

- [Landing page](https://home.versi0n.io)
- [Client app](https://versi0n.io/)
- [Orchestration frameworks](https://github.com/versionHQ/multi-agent-system)
- [Test client app](https://github.com/versionHQ/test-client-app)


## Mindmap

LLM-powered `agent`s and `team`s use `tool`s and their own knowledge to complete the `task` given by the client or the system.

<p align="center">
   <img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1733556715/pj_m_home/urwte15at3h0dr8mdlyo.png" alt="mindmap" width="400">
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
- [Contributing](#contributing)
  - [Customizing AI Agents](#customizing-ai-agents)
  - [Modifying RAG Functionality](#modifying-rag-functionality)
  - [Package Management with uv](#package-management-with-uv)
  - [Pre-Commit Hooks](#pre-commit-hooks)
- [Trouble Shooting](#trouble-shooting)
- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

<hr />

## Key Features

A mulit-agent systems with Rag that tailors messaging workflow, predicts its performance, and deploys it on third-party tools.

The `agent` is model agnostic. The default model is set Chat GTP 4o. We ask the client their preference and switch it accordingly using llm variable stored in the `BaseAgent` class.

Multiple `agents` can form a `team` to complete complex tasks together.

**1. Analysis**
- Professional `agents` handle the analysis `tasks` on each client, customer, and product. 

**2. Messaging Workflow Creation**
- Several `teams` receive the analysis and design initial messaging workflow with several layers.
- Ask the client for their inputs
- Deploy the workflow on the third party tools using `composio`.

**3. Autopiloting**
- Responsible `agents` or `teams` autopilot executing and refining the messaging workflow.


<hr />

## Technologies Used
**Schema, Database, Data Validation**
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
   - [uv](https://docs.astral.sh/uv/): Python package installer and resolver
   - [pre-commit](https://pre-commit.com/): Manage and maintain pre-commit hooks
   - [setuptools](https://pypi.org/project/setuptools/): Build python modules


## Project Structure

```
.
src/
└── versionHQ/                  # Orchestration frameworks on Pydantic
│      ├── agent/
│      └── llm/
│      └── task/
│      └── team/
│      └── tool/
│      └── clients/            # Classes to store the client related information
│      └── cli/                # CLI commands
│      └── ...
│      │
│      ├── db/                 # Database files
│      ├── chroma.sqlite3
│      └── ...
│
└──tests/
      └── cli/   
      └── team/
      └── ...
      │        
      └── uploads/    # Uploaded files for the project

```

<hr />

## Setup

1. Install the `uv` package manager:
   ```
   brew install uv
   ```

2. Install dependencies:
   ```
   uv venv
   source .venv/bin/activate

   uv pip install -r requirements.txt -v
   ```

* In case of AssertionError/module mismatch, run Python version control using `.pyenv`
   ```
   pyenv install 3.13.1
   pyenv global 3.13.1  (optional: `pyenv global system` to get back to the system default ver.)
   uv python pin 3.13.1
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

<hr />

## Usage

1. Install `versionhq` package:
```
uv pip install versionhq
```

2. You can use the `versionhq` module in your Python app.
```
from versionhq.agent.model import Agent
agent = Agent(llm="your-llm", ...)
```

For more details:

[PyPi package](https://pypi.org/project/versionhq/)

<hr />

## Contributing

1. Fork the repository

2. Create your feature branch (`git checkout -b feature/your-amazing-feature`)

3. Create amazing features

4. Test the features using the `tests` directory.

   - Add a file to the `tests` directory.
   - Run a test.
      ```
      uv run <your file name>
      ```

   * All the `.py` files' names in the `tests` have to be ended with `_test.py`.

5. Pull the latest version of source code from the main branch (`git pull origin main`) *Address conflicts if any.
6. Commit your changes (`git add .` / `git commit -m 'Add your-amazing-feature'`)
7. Push to the branch (`git push origin feature/your-amazing-feature`)
8. Open a pull request


**Optional**
* Flag with `#! REFINEME` for any improvements needed and `#! FIXME` for any errors.

* Run a React demo app: [React demo app](https://github.com/versionHQ/test-client-app) to check it on the client endpoint.
   ```
   npm i
   npm start
   ```
   The frontend will be available at `http://localhost:3000`.

* `production` is available at `https://versi0n.io`. Currently, we are running alpha test.



### Customizing AI Agents

To add an agent, use `sample` directory to add new `project`. You can define an agent with a specific role, goal, and set of tools.

Your new agent needs to follow the `Agent` model defined in the `verionhq.agent.model.py`.

You can also add any fields and functions to the `Agent` model **universally** by modifying `verionhq.agent.model.py`.


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

<hr />

## Trouble Shooting

Common issues and solutions:
- API key errors: Ensure all API keys in the `.env` file are correct and up to date. Make sure to add `load_dotenv()` on the top of the python file to apply the latest environment values.
- Database connection issues: Check if the Chroma DB is properly initialized and accessible.
- Memory errors: If processing large contracts, you may need to increase the available memory for the Python process.
- Issues related to dependencies:`rm -rf .venv uv.lock`, `uv cache clean`  and run `uv run pip install -r requirements.txt -v`.
- Issues related to the AI agents or RAG system: Check the `output.log` file for detailed error messages and stack traces.
- Issues related to `Python quit unexpectedly`: Check [this stackoverflow article](https://stackoverflow.com/questions/59888499/macos-catalina-python-quit-unexpectedly-error).

<hr />

## Frequently Asked Questions (FAQ)
**Q. Where can I see if the agent is working?**

> A. You can find a frontend app [here](https://versi0n.io) with real-world outbound use cases.
> You can also test features [here](https://github.com/versionHQ/test-client-app) using React app.

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



<--- Remaining tasks --->

- llm handling - agent
- more llms integration
- simpler prompting
- broader knowledge

- utils - log
- utils - time

- end to end client app test

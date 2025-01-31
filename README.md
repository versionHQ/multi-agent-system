# Overview

![MIT license](https://img.shields.io/badge/License-MIT-green)
[![Publisher](https://github.com/versionHQ/multi-agent-system/actions/workflows/publish.yml/badge.svg)](https://github.com/versionHQ/multi-agent-system/actions/workflows/publish.yml)
![PyPI](https://img.shields.io/badge/PyPI->=v1.1.11.1-blue)
![python ver](https://img.shields.io/badge/Python->=3.12-purple)
![pyenv ver](https://img.shields.io/badge/pyenv-2.5.0-orange)


LLM orchestration frameworks to deploy multi-agent systems with task-based formation.

**Visit:**

- [PyPI](https://pypi.org/project/versionhq/)
- [Github (LLM orchestration framework)](https://github.com/versionHQ/multi-agent-system)
- [Use case](https://versi0n.io/) / [Quick demo](https://res.cloudinary.com/dfeirxlea/video/upload/v1737732977/pj_m_home/pnsyh5mfvmilwgt0eusa.mov)
- [Documentation](https://chief-oxygen-8a2.notion.site/Documentation-17e923685cf98001a5fad5c4b2acd79b?pvs=4) *Some components are under review.


<hr />

## Table of Content
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Key Features](#key-features)
  - [Agent formation](#agent-formation)
- [Quick Start](#quick-start)
  - [Case 1. Solo Agent:](#case-1-solo-agent)
    - [Return a structured output with a summary in string.](#return-a-structured-output-with-a-summary-in-string)
  - [Case 2. Supervising:](#case-2-supervising)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setup](#setup)
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

Generate mulit-agent systems depending on the complexity of the task, and execute the task with agents of choice.

Model-agnostic agents can handle RAG tools, tools, callbacks, and knowledge sharing among other agents.


###  Agent formation
Depending on the task complexity, agents can make a different formation.

You can specify which formation you want them to generate, or let the agent decide if you don’t have a clear plan.


|  | **Solo Agent** | **Supervising** | **Network** | **Random** |
| :--- | :--- | :--- | :--- | :--- |
| **Formation** | <img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1737893140/pj_m_agents/tglrxoiuv7kk7nzvpe1z.jpg" alt="solo" width="200"> | <img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1737893141/pj_m_agents/hxngdvnn5b5qdxo0ayl5.jpg" alt="solo" width="200"> | <img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1737893142/pj_m_agents/kyc6neg8m6keyizszcpi.jpg" alt="solo" width="200"> | <img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1737893139/pj_m_agents/hzpchlcpnpwxwaapu1hr.jpg" alt="solo" width="200"> |
| **Usage** | <ul><li>A single agent with tools, knowledge, and memory.</li><li>When self-learning mode is on - it will turn into **Random** formation.</li></ul> | <ul><li>Leader agent gives directions, while sharing its knowledge and memory.</li><li>Subordinates can be solo agents or networks.</li></ul> | <ul><li>Share tasks, knowledge, and memory among network members.</li></ul> | <ul><li>A single agent handles tasks, asking help from other agents without sharing its memory or knowledge.</li></ul> |
| **Use case** | An email agent drafts promo message for the given audience. | The leader agent strategizes an outbound campaign plan and assigns components such as media mix or message creation to subordinate agents. | An email agent and social media agent share the product knowledge and deploy multi-channel outbound campaign. | 1. An email agent drafts promo message for the given audience, asking insights on tones from other email agents which oversee other clusters. 2. An agent calls the external agent to deploy the campaign. |

<hr />

## Quick Start

**Install `versionhq` package:**

   ```
   pip install versionhq
   ```

(Python >= 3.12)


### Case 1. Solo Agent:

#### Return a structured output with a summary in string.

   ```
   from pydantic import BaseModel
   from versionhq.agent.model import Agent
   from versionhq.task.model import Task

   class CustomOutput(BaseModel):
      test1: str
      test2: list[str]

   def dummy_func(message: str, test1: str, test2: list[str]) -> str:
      return f"{message}: {test1}, {", ".join(test2)}"


   agent = Agent(role="demo", goal="amazing project goal")

   task = Task(
      description="Amazing task",
      pydantic_output=CustomOutput,
      callback=dummy_func,
      callback_kwargs=dict(message="Hi! Here is the result: ")
   )

   res = task.execute_sync(agent=agent, context="amazing context to consider.")
   print(res)
   ```

This will return `TaskOutput` instance that stores a response in plane text, JSON serializable dict, and Pydantic model: `CustomOutput` formats with a callback result, tool output (if given), and evaluation results (if given).

   ```
   res == TaskOutput(
      task_id=UUID('<TASK UUID>')
      raw='{\"test1\":\"random str\", \"test2\":[\"str item 1\", \"str item 2\", \"str item 3\"]}',
      json_dict={'test1': 'random str', 'test2': ['str item 1', 'str item 2', 'str item 3']},
      pydantic=<class '__main__.CustomOutput'>
      tool_output=None,
      callback_output='Hi! Here is the result: random str, str item 1, str item 2, str item 3',
      evaluation=None
   )
   ```

### Case 2. Supervising:

   ```
   from versionhq.agent.model import Agent
   from versionhq.task.model import Task, ResponseField
   from versionhq.team.model import Team, TeamMember

   agent_a = Agent(role="agent a", goal="My amazing goals", llm="llm-of-your-choice")
   agent_b = Agent(role="agent b", goal="My amazing goals", llm="llm-of-your-choice")

   task_1 = Task(
      description="Analyze the client's business model.",
      response_fields=[ResponseField(title="test1", data_type=str, required=True),],
      allow_delegation=True
   )

    task_2 = Task(
      description="Define the cohort.",
      response_fields=[ResponseField(title="test1", data_type=int, required=True),],
      allow_delegation=False
   )

   team = Team(
      members=[
         TeamMember(agent=agent_a, is_manager=False, task=task_1),
         TeamMember(agent=agent_b, is_manager=True, task=task_2),
      ],
   )
   res = team.kickoff()
   ```

This will return a list with dictionaries with keys defined in the `ResponseField` of each task.

Tasks can be delegated to a team manager, peers in the team, or completely new agent.

<hr />

## Technologies Used
**Schema, Data Validation**
   - [Pydantic](https://docs.pydantic.dev/latest/): Data validation and serialization library for Python.
   - [Upstage](https://console.upstage.ai/docs/getting-started/overview): Document processer for ML tasks. (Use `Document Parser API` to extract data from documents)
   - [Docling](https://ds4sd.github.io/docling/): Document parsing

**Storage**
   - [mem0ai](https://docs.mem0.ai/quickstart#install-package): Agents' memory storage and management.
   - [Chroma DB](https://docs.trychroma.com/): Vector database for storing and querying usage data.
   - [SQLite](https://www.sqlite.org/docs.html): C-language library to implements a small SQL database engine.

**LLM-curation**
   - [LiteLLM](https://docs.litellm.ai/docs/providers): Curation platform to access LLMs

**Tools**
   - [Composio](https://composio.dev/): Conect RAG agents with external tools, Apps, and APIs to perform actions and receive triggers. We use [tools](https://composio.dev/tools) and [RAG tools](https://app.composio.dev/app/ragtool) from Composio toolset.

**Deployment**
   - Python: Primary programming language. v3.13 is recommended.
   - [uv](https://docs.astral.sh/uv/): Python package installer and resolver
   - [pre-commit](https://pre-commit.com/): Manage and maintain pre-commit hooks
   - [setuptools](https://pypi.org/project/setuptools/): Build python modules

<hr />

## Project Structure

```
.
.github
└── workflows/                # Github actions
│
src/
└── versionhq/                # Orchestration frameworks
│     ├── agent/              # Components
│     └── llm/
│     └── task/
│     └── team/
│     └── tool/
│     └── cli/
│     └── ...
│     │
│     ├── db/                 # Storage
│     ├── chroma.sqlite3
│     └── ...
│
└──tests/                     # Pytest
│     └── agent/
│     └── llm/
│     └── ...
│
└── uploads/                  # Local repo to store the uploaded files

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
   uv lock --upgrade
   uv sync --all-extras
   ```

* In case of AssertionError/module mismatch, run Python version control using `.pyenv`
   ```
   pyenv install 3.12.8
   pyenv global 3.12.8  (optional: `pyenv global system` to get back to the system default ver.)
   uv python pin 3.12.8
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

## Contributing

1. Create your feature branch (`git checkout -b feature/your-amazing-feature`)

2. Create amazing features

3. Test the features using the `tests` directory.

   - Add a test function to respective components in the `tests` directory.
   - Add your `LITELLM_API_KEY`, `OPENAI_API_KEY`, `COMPOSIO_API_KEY`, `DEFAULT_USER_ID` to the Github `repository secrets` located at settings > secrets & variables > Actions.
   - Run a test.
      ```
      uv run pytest tests -vv --cache-clear
      ```

   **pytest**

   * When adding a new file to `tests`, name the file ended with `_test.py`.
   * When adding a new feature to the file, name the feature started with `test_`.

4. Pull the latest version of source code from the main branch (`git pull origin main`) *Address conflicts if any.
5. Commit your changes (`git add .` / `git commit -m 'Add your-amazing-feature'`)
6. Push to the branch (`git push origin feature/your-amazing-feature`)
7. Open a pull request


**Optional**
* Flag with `#! REFINEME` for any improvements needed and `#! FIXME` for any errors.

* Run a React demo app: [React demo app](https://github.com/versionHQ/test-client-app) to check it on the client endpoint.
   ```
   npm i
   npm start
   ```
   The frontend will be available at `http://localhost:3000`.

* `production` use case is available at `https://versi0n.io`. Currently, we are running alpha test.



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
- Issues related to the Python version: Docling/Pytorch is not ready for Python 3.13 as of Jan 2025. Use Python 3.12.x as default by running `uv venv --python 3.12.8` and `uv python pin 3.12.8`.
- Issues related to dependencies: `rm -rf uv.lock`, `uv cache clean`, `uv venv`, and run `uv pip install -r requirements.txt -v`.
- Issues related to the AI agents or RAG system: Check the `output.log` file for detailed error messages and stack traces.
- Issues related to `Python quit unexpectedly`: Check [this stackoverflow article](https://stackoverflow.com/questions/59888499/macos-catalina-python-quit-unexpectedly-error).
- `reportMissingImports` error from pyright after installing the package: This might occur when installing new libraries while VSCode is running. Open the command pallete (ctrl + shift + p) and run the Python: Restart language server task.

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

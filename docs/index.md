# Overview

![DL](https://img.shields.io/badge/Download-15K+-red)
![MIT license](https://img.shields.io/badge/License-MIT-green)
[![Publisher](https://github.com/versionHQ/multi-agent-system/actions/workflows/publish.yml/badge.svg)](https://github.com/versionHQ/multi-agent-system/actions/workflows/publish.yml)
![PyPI](https://img.shields.io/badge/PyPI-v1.1.12+-blue)
![python ver](https://img.shields.io/badge/Python-3.11+-purple)
![pyenv ver](https://img.shields.io/badge/pyenv-2.5.0-orange)


Agentic orchestration framework to deploy agent network and handle complex task automation.

**Visit:**

- [PyPI](https://pypi.org/project/versionhq/)
- [Docs](https://docs.versi0n.io)
- [Playground](https://versi0n.io/)

**Contribute:**

- [Github Repository](https://github.com/versionHQ/multi-agent-system)

<hr />

## Key Features

`versionhq` is a Python framework for agent networks that handle complex task automation without human interaction.

Agents are model-agnostic, and will improve task output, while oprimizing token cost and job latency, by sharing their memory, knowledge base, and RAG tools with other agents in the network.


###  Agent formation

Agents adapt their formation based on task complexity.

You can specify a desired formation or allow the agents to determine it autonomously (default).


|  | **Solo Agent** | **Supervising** | **Network** | **Random** |
| :--- | :--- | :--- | :--- | :--- |
| **Formation** | <img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1738818211/pj_m_agents/rbgxttfoeqqis1ettlfz.png" alt="solo" width="200"> | <img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1738818211/pj_m_agents/zhungor3elxzer5dum10.png" alt="solo" width="200"> | <img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1738818211/pj_m_agents/dnusl7iy7kiwkxwlpmg8.png" alt="solo" width="200"> | <img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1738818211/pj_m_agents/sndpczatfzbrosxz9ama.png" alt="solo" width="200"> |
| **Usage** | <ul><li>A single agent with tools, knowledge, and memory.</li><li>When self-learning mode is on - it will turn into **Random** formation.</li></ul> | <ul><li>Leader agent gives directions, while sharing its knowledge and memory.</li><li>Subordinates can be solo agents or networks.</li></ul> | <ul><li>Share tasks, knowledge, and memory among network members.</li></ul> | <ul><li>A single agent handles tasks, asking help from other agents without sharing its memory or knowledge.</li></ul> |
| **Use case** | An email agent drafts promo message for the given audience. | The leader agent strategizes an outbound campaign plan and assigns components such as media mix or message creation to subordinate agents. | An email agent and social media agent share the product knowledge and deploy multi-channel outbound campaign. | 1. An email agent drafts promo message for the given audience, asking insights on tones from other email agents which oversee other clusters. 2. An agent calls the external agent to deploy the campaign. |

<hr />


## Quick Start

**Install `versionhq` package:**

   ```
   pip install versionhq
   ```

(Python 3.11, 3.12)

### Generate agent networks and launch task execution:

   ```
   from versionhq import form_agent_network

   network = form_agent_network(
      task="YOUR AMAZING TASK OVERVIEW",
      expected_outcome="YOUR OUTCOME EXPECTATION",
   )
   res = network.launch()
   ```

   This will form a network with multiple agents on `Formation` and return `TaskOutput` object with output in JSON, plane text, Pydantic model format with evaluation.


### Solo Agent:

You can simply build an agent using `Agent` model.

By default, the agent prioritize JSON serializable output.

But you can add a plane text summary of the structured output by using callbacks.

   ```python
   from pydantic import BaseModel
   from versionhq import Agent, Task

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


This will return a `TaskOutput` object that stores response in plane text, JSON, and Pydantic model: `CustomOutput` formats with a callback result, tool output (if given), and evaluation results (if given).

   ```python
   res == TaskOutput(
      task_id=UUID('<TASK UUID>'),
      raw='{\"test1\":\"random str\", \"test2\":[\"str item 1\", \"str item 2\", \"str item 3\"]}',
      json_dict={'test1': 'random str', 'test2': ['str item 1', 'str item 2', 'str item 3']},
      pydantic=<class '__main__.CustomOutput'>,
      tool_output=None,
      callback_output='Hi! Here is the result: random str, str item 1, str item 2, str item 3', # returned a plain text summary
      evaluation=None
   )
   ```

### Supervising:

   ```
   from versionhq import Agent, Task, ResponseField, Team, TeamMember

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
docs/                         # Documentation built by MkDocs
│
src/
└── versionhq/                # Orchestration framework package
│     ├── agent/              # Core components
│     └── llm/
│     └── task/
│     └── tool/
│     └── ...
│
└──tests/                     # Pytest - by core component and use cases in the docs
│     └── agent/
│     └── llm/
│     └── ...
│
└── uploads/                  # Local directory that stores uloaded files

```

<hr />

## Setup

1. Install `uv` package manager:

      For MacOS:

      ```
      brew install uv
      ```

      For Ubuntu/Debian:

      ```
      sudo apt-get install uv
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
   echo 3.12.8 > .python-version
   ```


3. Set up environment variables:
   Create a `.env` file in the project root and add the following:
   ```
   OPENAI_API_KEY=your-openai-api-key
   LITELLM_API_KEY=your-litellm-api-key
   COMPOSIO_API_KEY=your-composio-api-key
   COMPOSIO_CLI_KEY=your-composio-cli-key
   [LLM_INTERFACE_PROVIDER_OF_YOUR_CHOICE]_API_KEY=your-api-key
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



### Documentation

* To edit the documentation, see `docs` repository and edit the respective component.

* We use `mkdocs` to update the docs. You can run the doc locally at http://127.0.0.1:8000/:

   ```
   uv run python3 -m mkdocs serve --clean
   ```

* To add a new page, update `mkdocs.yml` in the root. Refer to [MkDocs official docs](https://squidfunk.github.io/mkdocs-material/getting-started/) for more details.


### Customizing AI Agent

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

A. Visit [playground](https://versi0n.io).


**Q. How do you analyze the customer?**

A. We employ soft clustering for each customer.
<img width="200" src="https://res.cloudinary.com/dfeirxlea/image/upload/v1732732628/pj_m_agents/ito937s5d5x0so8isvw6.png">

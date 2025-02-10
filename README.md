# Overview

[![DL](https://img.shields.io/badge/Download-15K+-red)](https://clickpy.clickhouse.com/dashboard/versionhq)
![MIT license](https://img.shields.io/badge/License-MIT-green)
[![Publisher](https://github.com/versionHQ/multi-agent-system/actions/workflows/publish.yml/badge.svg)](https://github.com/versionHQ/multi-agent-system/actions/workflows/publish.yml)
![PyPI](https://img.shields.io/badge/PyPI-v1.1.12+-blue)
![python ver](https://img.shields.io/badge/Python-3.11+-purple)
![pyenv ver](https://img.shields.io/badge/pyenv-2.5.0-orange)


A Python framework for agentic orchestration that handles complex task automation without human interaction.

**Visit:**

- [PyPI](https://pypi.org/project/versionhq/)
- [Docs](https://docs.versi0n.io)
- [Github Repository](https://github.com/versionHQ/multi-agent-system)
- [Playground](https://versi0n.io/)


<hr />

## Table of Content
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Key Features](#key-features)
  - [Agent formation](#agent-formation)
- [Quick Start](#quick-start)
  - [Package installation](#package-installation)
  - [Forming a agent network](#forming-a-agent-network)
  - [Customizing AI agents](#customizing-ai-agents)
  - [Supervising](#supervising)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setting Up a Project](#setting-up-a-project)
  - [1. Installing package manager: `uv`](#1-installing-package-manager-uv)
  - [2. Installing dependencies](#2-installing-dependencies)
  - [3. Adding secrets to .env file](#3-adding-secrets-to-env-file)
- [Contributing](#contributing)
  - [Package Management with uv](#package-management-with-uv)
  - [Pre-Commit Hooks](#pre-commit-hooks)
  - [Documentation](#documentation)
- [Trouble Shooting](#trouble-shooting)
- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

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

### Package installation

   ```
   pip install versionhq
   ```

(Python 3.11 / 3.12)

### Forming a agent network

   ```python
   import versionhq as vhq

   network = vhq.form_agent_network(
      task="YOUR AMAZING TASK OVERVIEW",
      expected_outcome="YOUR OUTCOME EXPECTATION",
   )
   res = network.launch()
   ```

   This will form a network with multiple agents on `Formation` and return `TaskOutput` object with output in JSON, plane text, Pydantic model format with evaluation.


### Customizing AI agents

You can simply build an agent using `Agent` model.

By default, the agent prioritize JSON serializable outputs over plane texts.


   ```python
   import versionhq as vhq
   from pydantic import BaseModel

   class CustomOutput(BaseModel):
      test1: str
      test2: list[str]

   def dummy_func(message: str, test1: str, test2: list[str]) -> str:
      return f"""{message}: {test1}, {", ".join(test2)}"""


   agent = vhq.Agent(role="demo", goal="amazing project goal")

   task = vhq.Task(
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

### Supervising

   ```python
   import versionhq as vhq

   agent_a = vhq.Agent(role="agent a", goal="My amazing goals", llm="llm-of-your-choice")
   agent_b = vhq.Agent(role="agent b", goal="My amazing goals", llm="llm-of-your-choice")

   task_1 = vhq.Task(
      description="Analyze the client's business model.",
      response_fields=[vhq.ResponseField(title="test1", data_type=str, required=True),],
      allow_delegation=True
   )

    task_2 = vhq.Task(
      description="Define the cohort.",
      response_fields=[ResponseField(title="test1", data_type=int, required=True),],
      allow_delegation=False
   )

   team = vhq.Team(
      members=[
         vhq.Member(agent=agent_a, is_manager=False, task=task_1),
         vhq.Member(agent=agent_b, is_manager=True, task=task_2),
      ],
   )
   res = team.launch()
   ```

This will return a list with dictionaries with keys defined in the `ResponseField` of each task.

Tasks can be delegated to a team manager, peers in the team, or completely new agent.

<hr />

## Technologies Used

**Schema, Data Validation**

* [Pydantic](https://docs.pydantic.dev/latest/): Data validation and serialization library for Python.
* [Upstage](https://console.upstage.ai/docs/getting-started/overview): Document processer for ML tasks. (Use `Document Parser API` to extract data from documents)
* [Docling](https://ds4sd.github.io/docling/): Document parsing

**Storage**

* [mem0ai](https://docs.mem0.ai/quickstart#install-package): Agents' memory storage and management.
* [Chroma DB](https://docs.trychroma.com/): Vector database for storing and querying usage data.
* [SQLite](https://www.sqlite.org/docs.html): C-language library to implements a small SQL database engine.

**LLM-curation**

* [LiteLLM](https://docs.litellm.ai/docs/providers): Curation platform to access LLMs

**Tools**

* [Composio](https://composio.dev/): Conect RAG agents with external tools, Apps, and APIs to perform actions and receive triggers. We use [tools](https://composio.dev/tools) and [RAG tools](https://app.composio.dev/app/ragtool) from Composio toolset.

**Deployment**

* **Python**: Primary programming language. v3.12.x is recommended
* [uv](https://docs.astral.sh/uv/): Python package installer and resolver
* [pre-commit](https://pre-commit.com/): Manage and maintain pre-commit hooks
* [setuptools](https://pypi.org/project/setuptools/): Build python modules

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

## Setting Up a Project

### 1. Installing package manager: `uv`

   For MacOS:

   ```
   brew install uv
   ```

   For Ubuntu/Debian:
   ```
   sudo apt-get install uv
   ```


### 2. Installing dependencies

   ```
   uv venv
   source .venv/bin/activate
   uv lock --upgrade
   uv sync --all-extras
   ```

   - AssertionError/module mismatch errors: Set up default Python version using `.pyenv`
      ```
      pyenv install 3.12.8
      pyenv global 3.12.8  (optional: `pyenv global system` to get back to the system default ver.)
      uv python pin 3.12.8
      echo 3.12.8 >> .python-version
      ```

   - `pygraphviz` related errors: Run the following commands:
      ```
      brew install graphbiz
      uv pip install --config-settings="--global-option=build_ext" \
      --config-settings="--global-option=-I$(brew --prefix graphviz)/include/" \
      --config-settings="--global-option=-L$(brew --prefix graphviz)/lib/" \
      pygraphviz
      ```

      ! If the error continues, skip pygraphviz installation by:
      ```
      uv sync --all-extras --no-extra pygraphviz
      ```

   - `torch`/`Docling` related errors: Set up default Python version either `3.11.x` or `3.12.x` (same as AssertionError)

### 3. Adding secrets to .env file

Create `.env` file in the project root and add following:

   ```
   OPENAI_API_KEY=your-openai-api-key
   GEMINI_API_KEY=your-gemini-api-key
   LITELLM_API_KEY=your-litellm-api-key
   COMPOSIO_API_KEY=your-composio-api-key
   COMPOSIO_CLI_KEY=your-composio-cli-key
   [OTHER_LLM_INTERFACE_PROVIDER_OF_YOUR_CHOICE]_API_KEY=your-api-key
   ```

<hr />

## Contributing

1. Create your feature branch (`git checkout -b feature/your-amazing-feature`)

2. Create amazing features

3. Add a test funcition to the `tests` directory and run **pytest**.

   - Add secret values defined in `.github/workflows/run_test.yml` to your Github `repository secrets` located at settings > secrets & variables > Actions.

   - Run a following command:
      ```
      uv run pytest tests -vv --cache-clear
      ```

   **Building a new pytest function**

   * Files added to the `tests` directory must end in `_test.py`.

   * Test functions within the files must begin with `test_`.

   * Pytest priorities are `1. playground demo > 2. docs use cases > 3. other features`


4. Update `docs` accordingly.

5. Pull the latest version of source code from the main branch (`git pull origin main`) *Address conflicts if any.

6. Commit your changes (`git add .` / `git commit -m 'Add your-amazing-feature'`)

7. Push to the branch (`git push origin feature/your-amazing-feature`)

8. Open a pull request


**Optional**

* Flag with `#! REFINEME` for any improvements needed and `#! FIXME` for any errors.

* `Playground` is available at `https://versi0n.io`.


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

### Documentation

* To edit the documentation, see `docs` repository and edit the respective component.

* We use `mkdocs` to update the docs. You can run the doc locally at http://127.0.0.1:8000/:

   ```
   uv run python3 -m mkdocs serve --clean
   ```

* To add a new page, update `mkdocs.yml` in the root. Refer to [MkDocs documentation](https://squidfunk.github.io/mkdocs-material/getting-started/) for more details.

<hr />

## Trouble Shooting

Common issues and solutions:

* API key errors: Ensure all API keys in the `.env` file are correct and up to date. Make sure to add `load_dotenv()` on the top of the python file to apply the latest environment values.

* Database connection issues: Check if the Chroma DB is properly initialized and accessible.

* Memory errors: If processing large contracts, you may need to increase the available memory for the Python process.

* Issues related to the Python version: Docling/Pytorch is not ready for Python 3.13 as of Jan 2025. Use Python 3.12.x as default by running `uv venv --python 3.12.8` and `uv python pin 3.12.8`.

* Issues related to dependencies: `rm -rf uv.lock`, `uv cache clean`, `uv venv`, and run `uv pip install -r requirements.txt -v`.

* Issues related to the AI agents or RAG system: Check the `output.log` file for detailed error messages and stack traces.

* Issues related to `Python quit unexpectedly`: Check [this stackoverflow article](https://stackoverflow.com/questions/59888499/macos-catalina-python-quit-unexpectedly-error).

* `reportMissingImports` error from pyright after installing the package: This might occur when installing new libraries while VSCode is running. Open the command pallete (ctrl + shift + p) and run the Python: Restart language server task.

<hr />

## Frequently Asked Questions (FAQ)
**Q. Where can I see if the agent is working?**

A. Visit [playground](https://versi0n.io).


**Q. How do you analyze the customer?**

A. We employ soft clustering for each customer.
<img width="200" src="https://res.cloudinary.com/dfeirxlea/image/upload/v1732732628/pj_m_agents/ito937s5d5x0so8isvw6.png">

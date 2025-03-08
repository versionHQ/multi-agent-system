# Overview

[![DL](https://img.shields.io/badge/Download-30K+-red)](https://clickpy.clickhouse.com/dashboard/versionhq)
![MIT license](https://img.shields.io/badge/License-MIT-green)
[![Publisher](https://github.com/versionHQ/multi-agent-system/actions/workflows/publish.yml/badge.svg)](https://github.com/versionHQ/multi-agent-system/actions/workflows/publish.yml)
![PyPI](https://img.shields.io/badge/PythonSDK-v1.2.4+-blue)
![python ver](https://img.shields.io/badge/Python-3.11|3.12|3.13-purple)
![pyenv ver](https://img.shields.io/badge/pyenv-2.5.0-orange)


A Python framework for autonomous agent networks that handle task automation with multi-step reasoning.

**Visit:**

- [Playground](https://versi0n.io/)
- [Documentation](https://docs.versi0n.io)
- [Github](https://github.com/versionHQ/)
- [Python SDK](https://pypi.org/project/versionhq/)

<hr />

## Table of Content
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Key Features](#key-features)
  - [Agent Network](#agent-network)
  - [Graph Theory Concept](#graph-theory-concept)
  - [Task Graph](#task-graph)
  - [Optimization](#optimization)
- [Quick Start](#quick-start)
  - [Package installation](#package-installation)
  - [Launching an agent](#launching-an-agent)
  - [Automating workflows](#automating-workflows)
  - [Executing a single task](#executing-a-single-task)
  - [Supervising agents](#supervising-agents)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setting Up Your Project](#setting-up-your-project)
  - [Installing package manager](#installing-package-manager)
  - [Installing dependencies](#installing-dependencies)
  - [Adding env secrets to .env file](#adding-env-secrets-to-env-file)
- [Contributing](#contributing)
  - [Steps](#steps)
  - [Package Management with uv](#package-management-with-uv)
  - [Pre-Commit Hooks](#pre-commit-hooks)
  - [Documentation](#documentation)
- [Trouble Shooting](#trouble-shooting)
- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

<hr />

## Key Features

`versionhq` is a Python framework designed for automating complex, multi-step tasks using autonomous agent networks.

Users can either configure their agents and network manually or allow the system to automatically manage the process based on provided task goals.


###  Agent Network

Agents adapt their formation based on task complexity.

You can specify a desired formation or allow the agents to determine it autonomously (default).


|                 | **Solo Agent**  | **Supervising**  | **Squad**         | **Random**      |
| :---            | :---            | :---             | :---              | :---            |
| **Formation**   | <img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1738818211/pj_m_agents/rbgxttfoeqqis1ettlfz.png" alt="solo" width="200"> | <img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1738818211/pj_m_agents/zhungor3elxzer5dum10.png" alt="solo" width="200"> | <img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1738818211/pj_m_agents/dnusl7iy7kiwkxwlpmg8.png" alt="solo" width="200"> | <img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1738818211/pj_m_agents/sndpczatfzbrosxz9ama.png" alt="solo" width="200"> |
| **Usage**       | <ul><li>A single agent with tools, knowledge, and memory.</li><li>When self-learning mode is on - it will turn into **Random** formation.</li></ul> | <ul><li>Leader agent gives directions, while sharing its knowledge and memory.</li><li>Subordinates can be solo agents or networks.</li></ul> | <ul><li>Share tasks, knowledge, and memory among network members.</li></ul> | <ul><li>A single agent handles tasks, asking help from other agents without sharing its memory or knowledge.</li></ul> |
| **Use case**    | An email agent drafts promo message for the given audience. | The leader agent strategizes an outbound campaign plan and assigns components such as media mix or message creation to subordinate agents. | An email agent and social media agent share the product knowledge and deploy multi-channel outbound campaign. | 1. An email agent drafts promo message for the given audience, asking insights on tones from other email agents which oversee other clusters. 2. An agent calls the external agent to deploy the campaign. |

<hr />

### Graph Theory Concept

To completely automate task workflows, agents will build a `task-oriented network` by generating `nodes` that represent tasks and connecting them with dependency-defining `edges`.

Each node is triggered by specific events and executed by an assigned agent once all dependencies are met.

While the network automatically reconfigures itself, you retain the ability to direct the agents using `should_reform` variable.


The following code snippet explicitly demonstrates the `TaskGraph` and its visualization, saving the diagram to the `uploads` directory.

```python
import versionhq as vhq

task_graph = vhq.TaskGraph(directed=False, should_reform=True) # triggering auto formation

task_a = vhq.Task(description="Research Topic")
task_b = vhq.Task(description="Outline Post")
task_c = vhq.Task(description="Write First Draft")

node_a = task_graph.add_task(task=task_a)
node_b = task_graph.add_task(task=task_b)
node_c = task_graph.add_task(task=task_c)

task_graph.add_dependency(
   node_a.identifier, node_b.identifier,
   dependency_type=vhq.DependencyType.FINISH_TO_START, weight=5, description="B depends on A"
)
task_graph.add_dependency(
   node_a.identifier, node_c.identifier,
   dependency_type=vhq.DependencyType.FINISH_TO_FINISH, lag=1, required=False, weight=3
)

# To visualize the graph:
task_graph.visualize()

# To start executing nodes:
latest_output, outputs = task_graph.activate()

assert isinstance(last_task_output, vhq.TaskOutput)
assert [k in task_graph.nodes.keys() and v and isinstance(v, vhq.TaskOutput) for k, v in outputs.items()]
```

<hr />

### Task Graph

A `TaskGraph` represents tasks as `nodes` and their execution dependencies as `edges`, automating rule-based execution.

`Agent Networks` can handle `TaskGraph` objects by optimizing their formations.

<img src="https://res.cloudinary.com/dfeirxlea/image/upload/v1739337639/pj_m_home/zfg4ccw1m1ww1tpnb0pa.png">

<hr />

### Optimization

Autonomous agents are model-agnostic and can leverage their own and their peers' knowledge sources, memories, and tools.

Agents are optimized during network formation, but customization is possible before or after.

The following code snippet demonstrates agent customization:

```python
import versionhq as vhq

agent = vhq.Agent(role="Marketing Analyst")

# update the agent
agent.update(
   llm="gemini-2.0", # updating LLM (Valid llm_config will be inherited to the new LLM.)
   tools=[vhq.Tool(func=lambda x: x)], # adding tools
   max_rpm=3,
   knowledge_sources=["<KC1>", "<KS2>"], # adding knowledge sources. This will trigger the storage creation.
   memory_config={"user_id": "0001"}, # adding memories
   dummy="I am dummy" # <- invalid field will be automatically ignored
)
```

<hr />

## Quick Start

### Package installation

```
pip install versionhq
```

(Python 3.11 | 3.12 | 3.13)


### Launching an agent


```python
import versionhq as vhq

agent = vhq.Agent(role="Marketer")
res = agent.start()

assert isinstance(res, vhq.TaskOutput) # contains agent's response in text, JSON, Pydantic formats with usage recordes and eval scores.
```


### Automating workflows

```python
import versionhq as vhq

network = vhq.form_agent_network(
   task="draft a promo plan",
   expected_outcome="marketing plan, budget, KPI targets",
)
res, tg = network.launch()

assert isinstance(res, vhq.TaskOutput) # the latest output from the workflow
assert isinstance(tg, vhq.TaskGraph) # contains task nodes and edges that connect the nodes with dep-met conditions
```


### Executing a single task

You can simply build and execute a task using `Task` class.

```python
import versionhq as vhq
from pydantic import BaseModel

class CustomOutput(BaseModel):
   test1: str
   test2: list[str]

def dummy_func(message: str, test1: str, test2: list[str]) -> str:
   return f"""{message}: {test1}, {", ".join(test2)}"""

task = vhq.Task(
   description="Amazing task",
   pydantic_output=CustomOutput,
   callback=dummy_func,
   callback_kwargs=dict(message="Hi! Here is the result: ")
)

res = task.execute(context="testing a task function")
assert isinstance(res, vhq.TaskOutput)
```


### Supervising agents

To create an agent network with one or more manager agents, designate members using the `is_manager` tag.

```python
import versionhq as vhq

agent_a = vhq.Agent(role="Member", llm="gpt-4o")
agent_b = vhq.Agent(role="Leader", llm="gemini-2.0")

task_1 = vhq.Task(
   description="Analyze the client's business model.",
   response_fields=[vhq.ResponseField(title="test1", data_type=str, required=True),],
   allow_delegation=True
)

task_2 = vhq.Task(
   description="Define a cohort.",
   response_fields=[vhq.ResponseField(title="test1", data_type=int, required=True),],
   allow_delegation=False
)

network =vhq.AgentNetwork(
   members=[
      vhq.Member(agent=agent_a, is_manager=False, tasks=[task_1]),
      vhq.Member(agent=agent_b, is_manager=True, tasks=[task_2]), # Agent B as a manager
   ],
)
res, tg = network.launch()

assert isinstance(res, vhq.NetworkOutput)
assert not [item for item in task_1.processed_agents if "vhq-Delegated-Agent" == item]
assert [item for item in task_1.processed_agents if "agent b" == item]
```

This will return a list with dictionaries with keys defined in the `ResponseField` of each task.

Tasks can be delegated to a manager, peers within the agent network, or a completely new agent.

<hr />

## Technologies Used

**Schema, Data Validation**

* [Pydantic](https://docs.pydantic.dev/latest/): Data validation and serialization library for Python.
* [Upstage](https://console.upstage.ai/docs/getting-started/overview): Document processer for ML tasks. (Use `Document Parser API` to extract data from documents)
* [Docling](https://ds4sd.github.io/docling/): Document parsing

**Workflow, Task Graph**

* [NetworkX](https://networkx.org/documentation/stable/reference/introduction.html): A Python package to analyze, create, and manipulate complex graph networks. Ref. [Gallary](https://networkx.org/documentation/latest/auto_examples/index.html)
* [Matplotlib](https://matplotlib.org/stable/index.html): For graph visualization.
* [Graphviz](https://graphviz.org/about/): For graph visualization.

**LLM Curation**

* [LiteLLM](https://docs.litellm.ai/docs/providers): LLM orchestration platform

**Tools**

* [Composio](https://composio.dev/): Conect RAG agents with external tools, Apps, and APIs to perform actions and receive triggers. We use [tools](https://composio.dev/tools) and [RAG tools](https://app.composio.dev/app/ragtool) from Composio toolset.


**Storage**

* [mem0ai](https://docs.mem0.ai/quickstart#install-package): Agents' memory storage and management.
* [Chroma DB](https://docs.trychroma.com/): Vector database for storing and querying usage data.
* [SQLite](https://www.sqlite.org/docs.html): C-language library to implements a small SQL database engine.


**Deployment**

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
docs/                         # Documentation
mkdocs.yml                    # MkDocs config
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
└── .diagrams/  [.gitignore]  # Local directory to store graph diagrams
│
└── .logs/      [.gitignore]  # Local directory to store error/warning logs for debugging
│
│
pyproject.toml                # Project config
.env.sample                   # sample .env file

```

<hr />

## Setting Up Your Project

### Installing package manager

   For MacOS:

   ```
   brew install uv
   ```

   For Ubuntu/Debian:
   ```
   sudo apt-get install uv
   ```


### Installing dependencies

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
      brew install graphviz
      uv pip install --config-settings="--global-option=build_ext" \
      --config-settings="--global-option=-I$(brew --prefix graphviz)/include/" \
      --config-settings="--global-option=-L$(brew --prefix graphviz)/lib/" \
      pygraphviz
      ```

      * If the error continues, skip pygraphviz installation by:
      ```
      uv sync --all-extras --no-extra pygraphviz
      ```

### Adding env secrets to .env file

Create `.env` file in the project root and add secret vars following `.env.sample` file.


<hr />

## Contributing

`versionhq` is a open source project.

### Steps

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

   * Pytest priorities are `1. playground > 2. docs use cases > 3. other features`


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

* To skip pre-commit hooks
   ```
   git commit --no-verify -m "your-commit-message"
   ```

### Documentation

* To edit the documentation, see `docs` repository and edit the respective component.

* We use `mkdocs` to update the docs. You can run the docs locally at http://127.0.0.1:8000/.

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

* Issues related to dependencies: `rm -rf uv.lock`, `uv cache clean`, `uv venv`, and run `uv pip install -r requirements.txt -v`.

* Issues related to `torch` installation: Add optional dependencies by `uv add versionhq[torch]`.

* Issues related to agents and other systems: Check `.logs` directory located at the root of the project directory for error messages and stack traces.

* Issues related to `Python quit unexpectedly`: Check [this stackoverflow article](https://stackoverflow.com/questions/59888499/macos-catalina-python-quit-unexpectedly-error).

* `reportMissingImports` error from pyright after installing the package: This might occur when installing new libraries while VSCode is running. Open the command pallete (ctrl + shift + p) and run the Python: Restart language server task.

<hr />

## Frequently Asked Questions (FAQ)
**Q. Where can I see if the agent is working?**

A. Visit [playground](https://versi0n.io).

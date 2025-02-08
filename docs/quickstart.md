# Quick Start

## Package installation

   ```
   pip install versionhq
   ```

(Python 3.11, 3.12 [Recommended])


## Setting up a project

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
   echo 3.12.8 >> .python-version
   ```


3. Add secrets to `.env` file in the project root:

   ```
   OPENAI_API_KEY=your-openai-api-key
   LITELLM_API_KEY=your-litellm-api-key
   COMPOSIO_API_KEY=your-composio-api-key
   COMPOSIO_CLI_KEY=your-composio-cli-key
   [LLM_INTERFACE_PROVIDER_OF_YOUR_CHOICE]_API_KEY=your-api-key
   ```


## Forming a agent network

You can generate a network of multiple agents depending on your task complexity.

Here is a code snippet:

   ```python
   import versionhq as vhq

   network = vhq.form_agent_network(
      task="YOUR AMAZING TASK OVERVIEW",
      expected_outcome="YOUR OUTCOME EXPECTATION",
   )
   res = network.launch()
   ```

This will form a network with multiple agents on `Formation` and return results as a `TaskOutput` object, storing outputs in JSON, plane text, Pydantic model formats along with evaluation.


## Customizing your agent

If you don't need to form a network or assign a specific agent to the network, you can simply build an agent using `Agent` model.

By default, the agent prioritize JSON serializable outputs over plane text.

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

## Supervising

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
      response_fields=[vhq.ResponseField(title="test1", data_type=int, required=True),],
      allow_delegation=False
   )

   team =vhq.Team(
      members=[
         vhq.Member(agent=agent_a, is_manager=False, task=task_1),
         vhq.Member(agent=agent_b, is_manager=True, task=task_2),
      ],
   )
   res = team.launch()
   ```

This will return a list with dictionaries with keys defined in the `ResponseField` of each task.

Tasks can be delegated to a team manager, peers in the team, or completely new agent.

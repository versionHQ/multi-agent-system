# Quick Start

## Install `versionhq` package

   ```
   pip install versionhq
   ```

(Python 3.11, 3.12)


## Set up a project

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


## Launch an agent network

You can create a network of multiple agents depending on the task complexity.
Here is the snippet of quick launch.

   ```
   from versionhq import form_agent_network

   network = form_agent_network(
      task="YOUR AMAZING TASK OVERVIEW",
      expected_outcome="YOUR OUTCOME EXPECTATION",
   )
   res = network.launch()
   ```

   This will form a network with multiple agents on `Formation` and return `TaskOutput` object with output in JSON, plane text, and Pydantic model format with evaluation.


## Use a solo agent

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
   # TaskOutput object
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

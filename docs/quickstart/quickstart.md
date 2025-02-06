# Quick Start

## Install `versionhq` package

   ```
   pip install versionhq
   ```

(Python 3.11 and higher)



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

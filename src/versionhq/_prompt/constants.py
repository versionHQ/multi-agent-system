REFLECT = "Here is the orignal prompt: {original_prompt}\nHere is the original response: {original_response}\nAnalyze the original prompt and repsonse, check for any pontential issue, and create an improved response."

INTEGRATE = "Here is the original prompt: {original_prompt}\nHere are responses: {responses}. Help integrate them as a single response."

parameter_sets = [
    {
        "temperature": 0.2,
        "top_p": 0.5,
        "max_tokens": 5000,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,
        "stop": ["\n\n", "###"],
    },
    {
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 8000,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.3,
        "stop": ["\n\n"],
    },
    {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 12000,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": [],
    }
]

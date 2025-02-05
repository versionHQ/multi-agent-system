EVALUATE="""Evaluate the provided task output against the given task description, assigning a score between 0 (worst) and 1 (best) based on the specified criteria. Scores should be numerical (integers or decimals). Provide specific suggestions for improvement. Do not assign identical scores to different criteria:
Task output: {task_output}
Task description: {task_description}
Evaluation criteria: {eval_criteria}
"""

EVALUATE="""Evaluate the provided task output against the given task description, assigning a score between 0 (worst) and 1 (best) based on the specified criteria. Scores should be numerical (integers or decimals). Weight should be numerical (integers or decimals) and represents importance of the criteria to the final result. Provide specific suggestions for improvement. Do not assign identical scores to different criteria unless otherwise you have clear reasons to do so:
Task output: {task_output}
Task description: {task_description}
Evaluation criteria: {eval_criteria}
"""

SHOTS="""Here are two examples of task outputs. The first is considered excellent due to its clear planning and alignment with the goal. The second is weak due to clichéd language. Now, evaluate the given task output.
First = Excellent example: {c}
Second = Weak example: {w}
"""

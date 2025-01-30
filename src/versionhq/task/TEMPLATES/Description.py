EVALUATE="""Assess the accuracy and quality of the following task output to the task described below. Score based on the criterion (0-1, 0=worst, 1=best) and suggest improvements. Vary scores; don't assign identical values. Store criteria in the "criteria" field.
Task: {task_description}
Task Output: {task_output}
Evaluation criteria: {eval_criteria}
"""

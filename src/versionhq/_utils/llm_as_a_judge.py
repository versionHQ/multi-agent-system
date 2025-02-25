import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, cohen_kappa_score
from typing import List, Tuple, Dict, Any
from pathlib import Path


class LLMJudge:

    class MockLLM:
        def _generate(self, prompt: str) -> str:
            return str(np.random.random())


    def __init__(self, model: MockLLM = None):
        self.model = model if model else self.MockLLM()


    def judge_summary(self, original_text: str, summary: str) -> float:
        prompt = f"""Evaluate the quality of the following summary on a scale of 0 to 1, where 0 is poor and 1 is excellent.
Consider accuracy, completeness, and conciseness.
Original text: {original_text}
Summary: {summary}
Quality score:"""
        response = self.model._generate(prompt)
        score = float(response.strip())
        return score


def generate_summaries(file_path: str, data: List[Dict[str, Any]] = None, summarizer: Any = None) -> List[Tuple[str, str, float]]:
    """Generates a list of tuple with an original text, summary text, and human judge score."""
    if not data:
        with open(file_path, 'r') as file:
            data = json.load(file)
    summaries = []
    for item in data:
        original_text = item['text']
        summary = summarizer.summarize(original_text)
        human_score = item['human_score']
        summaries.append((original_text, summary, human_score))

    return summaries


def validate(judge: LLMJudge, data: List[Tuple[str, str, float]], threshold: float = 0.5):
    human_scores = []
    predicted_scores = []

    for original_text, summary, human_score in data:
        predicted_score = judge.judge_summary(original_text=original_text, summary=summary)
        human_scores.append(human_score)
        predicted_scores.append(predicted_score)

    human_binary = [1 if score >= threshold else 0 for score in human_scores]
    pred_binary = [1 if score >= threshold else 0 for score in predicted_scores]
    precision = precision_score(human_binary, pred_binary, zero_division=0)
    recall = recall_score(human_binary, pred_binary, zero_division=0)
    auroc = roc_auc_score(human_binary, pred_binary, average='weighted')
    kappa = cohen_kappa_score(human_binary, pred_binary)
    return {
        "precision": precision,
        "recall": recall,
        "auroc": auroc,
        "cohen_kappa": kappa
    }

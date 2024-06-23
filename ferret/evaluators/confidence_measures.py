import numpy as np

from statistics import mean
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error

from . import BaseEvaluator, EvaluationMetricFamily
from .evaluation import EvaluationMetricOutput
from ..explainers.explanation import Explanation


class CI_Confidence_Evaluation(BaseEvaluator):
    NAME = "confidence_indication"
    SHORT_NAME = "ci"
    LOWER_IS_BETTER = False
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0
    METRIC_FAMILY = EvaluationMetricFamily.CONFIDENCE  # Cambiato da FAITHFULNESS a CONFIDENCE

    def compute_evaluation(self, explanation: Explanation, **evaluation_args):
        text = explanation.text
        score_explanation = explanation.scores
        target_pos_idx = explanation.target_pos_idx

        # Compute Saliency Distance (SD)
        SD = self._compute_saliency_distance(score_explanation, target_pos_idx)

        # Predict confidence using logistic regression (LR)
        LR = LogisticRegression()
        LR.fit(SD.reshape(-1, 1), explanation.confidence_scores)
        predicted_confidence = LR.predict(SD.reshape(-1, 1))

        # Compute Mean Absolute Error (MAE)
        mae = mean_absolute_error(explanation.confidence_scores, predicted_confidence)

        # Return evaluation metric output
        return EvaluationMetricOutput(self, 1.0 - mae)

    def _compute_saliency_distance(self, score_explanation, target_pos_idx):
        # Implement saliency distance computation here
        # Example calculation:
        saliency_scores = score_explanation[target_pos_idx]
        other_scores = [score_explanation[idx] for idx in range(len(score_explanation)) if idx != target_pos_idx]

        # Compute SD based on your definition
        SD = np.abs(saliency_scores - np.mean(other_scores))
        return SD

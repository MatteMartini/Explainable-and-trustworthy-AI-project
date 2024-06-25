# import numpy as np

# from statistics import mean
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import mean_absolute_error

# from . import BaseEvaluator, EvaluationMetricFamily
# from .evaluation import EvaluationMetricOutput
# from ..explainers.explanation import Explanation


# class CI_Confidence_Evaluation(BaseEvaluator):
#      NAME = "confidence_indication"
#      SHORT_NAME = "ci"
#      LOWER_IS_BETTER = False
#      MIN_VALUE = 0.0
#      MAX_VALUE = 1.0
#      METRIC_FAMILY = EvaluationMetricFamily.CONFIDENCE  # Cambiato da FAITHFULNESS a CONFIDENCE

#      def compute_evaluation(self, explanation: Explanation, **evaluation_args):


#         text = explanation.text
#         score_explanation = np.array([0.5])
#         target_pos_idx = explanation.target_pos_idx


#          # Compute Saliency Distance (SD)
#         SD = self._compute_saliency_distance(score_explanation, target_pos_idx)

#         print(score_explanation.size)
#         print(SD.reshape(-1, 1).size)

#          # Predict confidence using logistic regression (LR)
#         LR = LogisticRegression()
#         LR.fit(SD.reshape(-1, 1), explanation.scores)
#         predicted_confidence = LR.predict(SD.reshape(-1, 1))

#          # Compute Mean Absolute Error (MAE)
#         mae = mean_absolute_error(explanation.scores, predicted_confidence)

#            # Return evaluation metric output
#         return EvaluationMetricOutput(self, 1.0 - mae)

#      def _compute_saliency_distance(self, score_explanation, target_pos_idx):
#          # Implement saliency distance computation here
#          # Example calculation:
#         saliency_scores = score_explanation[target_pos_idx]
#         other_scores = [score_explanation[idx] for idx in range(len(score_explanation)) if idx != target_pos_idx]

#           # Compute SD based on your definition
#         SD = np.abs(saliency_scores - np.mean(other_scores))
#         return SD





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

        # Get prediction probability of the input sentence for the target
        _, logits = self.helper._forward(text, output_hidden_states=False)
        logits = self.helper._postprocess_logits(
            logits, target_token_pos_idx=explanation.target_token_pos_idx
        )
        true_confidence = logits.softmax(-1)[0, target_pos_idx].item()

        # Compute Saliency Distance (SD)
        SD = self._compute_saliency_distance(score_explanation, target_pos_idx)

        # Predict confidence using logistic regression (LR)
        LR = LogisticRegression()
        LR.fit(SD.reshape(-1, 1), [true_confidence])  # Usa la true confidence calcolata
        predicted_confidence = LR.predict(SD.reshape(-1, 1))

        # Compute Mean Absolute Error (MAE)
        mae = mean_absolute_error([true_confidence], predicted_confidence)

        # Return evaluation metric output
        return EvaluationMetricOutput(self, 1.0 - mae)

    def _compute_saliency_distance(self, score_explanation, target_pos_idx):
        saliency_scores = score_explanation[target_pos_idx]
        
        if len(score_explanation) == 2:  # Caso con sole due classi
            other_pos_idx = 1 - target_pos_idx  # Determina l'indice della classe non target
            other_scores = score_explanation[other_pos_idx]
            
        else:  # Caso con più di due classi

            other_pos_idx = 1 - target_pos_idx  # Determina l'indice della classe non target
            other_scores = score_explanation[other_pos_idx]
            
            print(other_scores)

            max_diff = np.max(saliency_scores - other_scores)
            min_diff = np.min(saliency_scores - other_scores)
            mean_diff = np.mean(saliency_scores - other_scores)
            SD = np.concatenate([max_diff, min_diff, mean_diff])
            

        SD = np.abs(saliency_scores - np.mean(other_scores))
        return SD
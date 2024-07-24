import numpy as np

from statistics import mean
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, max_error
from scipy.special import softmax, expit


from . import BaseEvaluator, EvaluationMetricFamily
from .evaluation import EvaluationMetricOutput
from ..explainers.explanation import Explanation


class CI_Confidence_Evaluation(BaseEvaluator):
    NAME = "confidence_indication"
    SHORT_NAME = "ci"
    LOWER_IS_BETTER = False
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0
    METRIC_FAMILY = EvaluationMetricFamily.CONFIDENCE 


    def compute_evaluation(self, explanation: Explanation, **evaluation_args):


        if isinstance(explanation, list)==False:
            return None
        
        expl_target = explanation[-1]

        all_y = []
        features = []
        
        for expl in explanation:
        
            scores = expl.scores 
            all_scores = expl.all_scores
            

            scores = scores[1:-1]
            for key in all_scores: 
                all_scores[key] = all_scores[key][1:-1]

            # Take logits and prediction
            _, logits = self.helper._forward(expl.text, output_hidden_states=False)
            logits = self.helper._postprocess_logits(
                logits, target_token_pos_idx=expl.target_token_pos_idx
            )


        
            instance_logits = softmax(logits)[0] 
            
            y = instance_logits[expl.target_pos_idx]
    
            saliency_pred = np.array(scores)

            other_sals = []

            for class_idx in range(len(all_scores)):
                if class_idx != expl.target_pos_idx:
                    other_sals.append(np.array(all_scores[class_idx]))
                    
            feats = []
        
            if len(all_scores) == 2:
                feats.append(sum(saliency_pred - other_sals[0]))
                feats.append(sum(saliency_pred - other_sals[0]))
                feats.append(sum(saliency_pred - other_sals[0]))
            else:
                feats.append(sum(np.max([saliency_pred - other_sals[0],
                                        saliency_pred - other_sals[1]], axis=0)))
                feats.append(sum(np.mean([saliency_pred - other_sals[0],
                                        saliency_pred - other_sals[1]], axis=0)))
                feats.append(sum(np.min([saliency_pred - other_sals[0],
                                        saliency_pred - other_sals[1]], axis=0)))

            all_y.append(y)
            features.append(feats)

        features = MinMaxScaler().fit_transform(np.array(features))
        all_y = np.array(all_y)

        target_instance = features[-1]
        target_y = all_y[-1]
        features = features[:-1]
        all_y = all_y[:-1]

        model = LinearRegression()
        model.fit(features, all_y)

        target_instance = np.array(target_instance)
        target_instance = target_instance.reshape(1,-1)

        prediction = model.predict(target_instance)
        target_y = np.array(target_y)

        mean_abs_error = np.abs(target_y - prediction)
    
        return EvaluationMetricOutput(self, mean_abs_error[0])

        
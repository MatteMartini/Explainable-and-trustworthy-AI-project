import numpy as np

from statistics import mean
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
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
    METRIC_FAMILY = EvaluationMetricFamily.CONFIDENCE  # Cambiato da FAITHFULNESS a CONFIDENCE


    # def compute_evaluation(self, explanation: Explanation, **evaluation_args):
    #     text = explanation.text
    #     score_explanation = explanation.scores
    #     target_pos_idx = explanation.target_pos_idx

    #     Get prediction probability of the input sentence for the target
    #     _, logits = self.helper._forward(text, output_hidden_states=False)
    #     logits = self.helper._postprocess_logits(
    #         logits, target_token_pos_idx=explanation.target_token_pos_idx
    #     )
    #     true_confidence = logits.softmax(-1)[0, target_pos_idx].item()

    #     Compute Saliency Distance (SD)
    #     SD = self._compute_saliency_distance(score_explanation, target_pos_idx)

    #     Predict confidence using logistic regression (LR)
    #     LR = LogisticRegression()
    #     LR.fit(SD.reshape(-1, 1), [true_confidence])  # Usa la true confidence calcolata
    #     predicted_confidence = LR.predict(SD.reshape(-1, 1))

    #     Compute Mean Absolute Error (MAE)
    #     mae = mean_absolute_error([true_confidence], predicted_confidence)

    #     Return evaluation metric output
    #     return EvaluationMetricOutput(self, 1.0 - mae)

    # def _compute_saliency_distance(self, score_explanation, target_pos_idx):
    #     saliency_scores = score_explanation[target_pos_idx]
        
    #     if len(score_explanation) == 2:  # Caso con sole due classi
    #         other_pos_idx = 1 - target_pos_idx  # Determina l'indice della classe non target
    #         other_scores = score_explanation[other_pos_idx]
    #     else:  # Caso con più di due classi
    #         other_scores = np.mean([score_explanation[i] for i in range(len(score_explanation)) if i != target_pos_idx], axis=0)
    #         max_diff = np.max(saliency_scores - other_scores)
    #         min_diff = np.min(saliency_scores - other_scores)
    #         mean_diff = np.mean(saliency_scores - other_scores)
    #         SD = np.array([max_diff, min_diff, mean_diff])

    #     SD = np.abs(saliency_scores - np.mean(other_scores))
    #     return SD

############################################################################################
############################################################################################


    # def compute_evaluation(self, explanation: Explanation, **evaluation_args):
    #     # Assuming explanation object contains necessary attributes
    #     saliencies = explanation.scores
    #     target_pos_idx = explanation.target_pos_idx

    #     # Get prediction probability of the input sentence for the target
    #     _, logits = self.helper._forward(explanation.text, output_hidden_states=False)
    #     logits = self.helper._postprocess_logits(
    #         logits, target_token_pos_idx=explanation.target_token_pos_idx
    #     )
    #     true_confidence = logits.softmax(-1)[0, target_pos_idx].item()

    #     # Prepare features and true labels
    #     features, y = self._prepare_features(saliencies, logits, target_pos_idx)

    #     # Evaluate model
    #     return self._evaluate(features, y, true_confidence)

    # def _prepare_features(self, saliencies, logits, target_pos_idx):
    #     features = []
    #     y = []
    #     classes = list(range(saliencies.shape[0]))

    #     for i in range(len(logits)):
    #         _cls = target_pos_idx
    #         instance_logits = softmax(logits[i])

    #         confidence_pred = instance_logits[_cls]
    #         saliency_pred = np.array(saliencies[_cls])

    #         left_classes = classes.copy()
    #         left_classes.remove(_cls)
    #         other_sals = [np.array(saliencies[c_]) for c_ in left_classes]

    #         feats = self._compute_saliency_distance(saliency_pred, other_sals)
    #         features.append(feats)
    #         y.append(confidence_pred)

    #     features = MinMaxScaler().fit_transform(np.array(features))
    #     return features, np.array(y)

    # def _compute_saliency_distance(self, saliency_pred, other_sals):
    #     if len(other_sals) == 1:
    #         return np.sum(saliency_pred - other_sals[0])

    #     max_diff = np.max([saliency_pred - other_sals[0], saliency_pred - other_sals[1]], axis=0)
    #     min_diff = np.min([saliency_pred - other_sals[0], saliency_pred - other_sals[1]], axis=0)
    #     mean_diff = np.mean([saliency_pred - other_sals[0], saliency_pred - other_sals[1]], axis=0)
    #     return [sum(max_diff), sum(mean_diff), sum(min_diff)]

    # def _evaluate(self, features, y, true_confidence):
    #     rs = ShuffleSplit(n_splits=5, random_state=2)
    #     scores = []
    #     coefs = []

    #     for train_index, test_index in rs.split(features):
    #         X_train, y_train, X_test, y_test = features[train_index], y[train_index], features[test_index], y[test_index]
    #         reg = LinearRegression().fit(X_train, y_train)
    #         test_pred = reg.predict(X_test)

    #         scores.append(mean_absolute_error(y_test, test_pred))

    #     mae = np.mean(scores)
    #     return 1.0 - mae
    
    # def compute_evaluation(self, explanation: Explanation):
    #     # Estrazione delle variabili dall'oggetto Explanation
    #     saliencies = explanation.scores
    #     target_pos_idx = explanation.target_pos_idx

    #     # Ottenere i logits e la predizione
    #     _, logits = self.helper._forward(explanation.text, output_hidden_states=False)
    #     logits = self.helper._postprocess_logits(
    #         logits, target_token_pos_idx=explanation.target_token_pos_idx
    #     )
    #     true_confidence = logits.softmax(-1)[0, target_pos_idx].item()
        
    #     # Calcolare class_preds e classi uniche
    #     class_preds = np.argmax(logits, axis=1)
    #     classes = np.unique(class_preds)

    #     all_y = []
    #     features = []

    #     # Ciclo su ciascuna istanza nelle salienze
    #     for i, instance_saliency in enumerate(saliencies):
    #         _cls = class_preds[i]
    #         instance_logits = softmax(logits[i])

    #         confidence_pred = instance_logits[_cls]
    #         saliency_pred = np.array(instance_saliency[_cls])

    #         left_classes = classes.copy()
    #         left_classes = np.delete(left_classes, np.where(left_classes == _cls))
    #         other_sals = [np.array(instance_saliency[c_]) for c_ in left_classes]
    #         feats = []

    #         if len(classes) == 2:
    #             feats.append(sum(saliency_pred - other_sals[0]))
    #             feats.append(sum(saliency_pred - other_sals[0]))
    #             feats.append(sum(saliency_pred - other_sals[0]))
    #         else:
    #             feats.append(sum(np.max([saliency_pred - other_sals[0],
    #                                     saliency_pred - other_sals[1]], axis=0)))
    #             feats.append(sum(np.mean([saliency_pred - other_sals[0],
    #                                     saliency_pred - other_sals[1]], axis=0)))
    #             feats.append(sum(np.min([saliency_pred - other_sals[0],
    #                                     saliency_pred - other_sals[1]], axis=0)))

    #         all_y.append(confidence_pred)
    #         features.append(feats)   
    #     # features è una lista di vettori di caratteristiche che rappresentano le differenze di saliency tra la classe predetta e le altre classi

    #     # Normalizzazione delle caratteristiche
    #     features = MinMaxScaler().fit_transform(np.array(features))
    #     all_y = np.array(all_y)

    #     rs = ShuffleSplit(n_splits=5, random_state=2)
    #     scores = []
    #     coefs = []

    #     for train_index, test_index in rs.split(features):
    #         X_train, y_train, X_test, y_test = features[train_index], all_y[train_index], features[test_index], all_y[test_index]
            
    #         reg = LinearRegression().fit(X_train, y_train)
    #         test_pred = reg.predict(X_test)

    #         # Calcolo della metrica
    #         all_metrics = [mean_absolute_error(y_test, test_pred)]
    #         scores.append(all_metrics)
    #         coefs.append(reg.coef_)
    #     return all_metrics

    def compute_evaluation(self, explanation: Explanation, **evaluation_args):


        if isinstance(explanation, list):
            explanation = explanation[0]
            
        # Estrazione delle variabili dall'oggetto Explanation
        scores = explanation.scores #COSI PRENDI SOLO QUELLI DELLA CLASSE PREDETTA, IL CUI INDICE è DATO DA target_pos_idx
        all_scores = explanation.all_scores

        scores = scores[1:-1]
        for key in all_scores:
             all_scores[key] = all_scores[key][1:-1]

        #print("all_scores")
        #print(all_scores)
        
        target_pos_idx = explanation.target_pos_idx  #indice classe da spiegare
        
        # Ottenere i logits e la predizione
        _, logits = self.helper._forward(explanation.text, output_hidden_states=False)
        logits = self.helper._postprocess_logits(
            logits, target_token_pos_idx=explanation.target_token_pos_idx
        )

        true_confidence = logits.softmax(-1)[0, target_pos_idx].item() #questo ti da il softmax(logit) della classe target come lo fa lui, quello di sotto commentato della classe con prob piu alta 
        # Calcolare class_preds e classi uniche

        instance_logits = softmax(logits)[0] #trasformo i 3 logits dell'istanza in probabilita

        
        #print(instance_logits)
       #print(target_pos_idx) #=1
        #print(true_confidence)

        y = instance_logits[target_pos_idx]
        saliency_pred = np.array(scores)

        other_sals = []

        for class_idx in range(len(all_scores)):
            if class_idx != target_pos_idx:
                other_sals.append(np.array(all_scores[class_idx]))
                
        #print("other sals")
        #print(other_sals)
        
       
        if len(logits) == 2:
             result =  (sum(saliency_pred - other_sals[0]))
          
        else:

            mean = np.mean([other_sals[0],other_sals[1]], axis=0)
            diff = saliency_pred - mean
            result = np.sum(diff)

           
        #min_value = -1 * len(saliency_pred)
        #max_value = 1 * len(saliency_pred)

        #normalized_result = (result - min_value) / (max_value - min_value)
        normalized_result = expit(result)
        
        #print("normalized")
        #print(normalized_result)
        mean_abs_error = np.abs(y - normalized_result)


        
        return EvaluationMetricOutput(self, mean_abs_error)

       
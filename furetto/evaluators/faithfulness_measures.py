import copy
import pdb
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.stats import kendalltau

import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

from ..explainers.explanation import Explanation, ExplanationWithRationale
from . import BaseEvaluator, EvaluationMetricFamily
from .evaluation import EvaluationMetricOutput
from .perturbation import PertubationHelper
from .utils_from_soft_to_discrete import (
    _check_and_define_get_id_discrete_rationale_function,
    get_discrete_explanation_topK,
    parse_evaluator_args,
    parse_evaluator_args_model,
)


def _compute_aopc(scores):
    from statistics import mean

    return mean(scores)


class AOPC_Comprehensiveness_Evaluation(BaseEvaluator):
    NAME = "aopc_comprehensiveness"
    SHORT_NAME = "aopc_compr"

    LOWER_IS_BETTER = False
    MIN_VALUE = -1.0
    MAX_VALUE = 1.0
    BEST_VALUE = 1.0
    METRIC_FAMILY = EvaluationMetricFamily.FAITHFULNESS

    def compute_evaluation(
        self, explanation: Explanation, **evaluation_args
    ) -> EvaluationMetricOutput:
        """Evaluate an explanation on the AOPC Comprehensiveness metric.

        Args:
            explanation (Explanation): the explanation to evaluate
            target (int): class label for which the explanation is evaluated
            evaluation_args (dict):  additional evaluation args.
                We currently support multiple approaches to define the hard rationale from
                soft score rationales, based on:
                - th : token greater than a threshold
                - perc : more than x% of the tokens
                - k: top k values

        Returns:
            Evaluation : the AOPC Comprehensiveness score of the explanation
        """

        remove_first_last, only_pos, removal_args, _ = parse_evaluator_args(
            evaluation_args
        )
        
        
        
    
        if isinstance(explanation, list):
            return None
            
        text = explanation.text
        target_pos_idx = explanation.target_pos_idx
        target_token_pos_idx = explanation.target_token_pos_idx
        score_explanation = explanation.scores
        helper_type = explanation.helper_type

        if (
            removal_args["remove_tokens"] == True
            and helper_type == "token-classification"
        ):
            removal_args["remove_tokens"] = False
            warnings.warn(
                "NER does not support token removal. 'remove_tokens' set to False"
            )

        # TODO - use tokens
        # Get prediction probability of the input sencence for the target
        _, logits = self.helper._forward(text, output_hidden_states=False)
        logits = self.helper._postprocess_logits(
            logits, target_token_pos_idx=target_token_pos_idx
        )

        baseline = logits.softmax(-1)[0, target_pos_idx].item()

        # TODO This part needs serious revision if metrics have to be general across modalities
        # Tokenized input
        item = self.helper._tokenize(text)
        input_len = item["attention_mask"].sum().item()
        input_ids = item["input_ids"][0][:input_len].tolist()

        # If remove_first_last, first and last token id (CLS and ) are removed
        if remove_first_last == True:
            input_ids = input_ids[1:-1]
            if self.tokenizer.cls_token == explanation.tokens[0]:
                score_explanation = score_explanation[1:-1]

        discrete_expl_ths = list()
        id_tops = list()
        get_discrete_rationale_function = (
            _check_and_define_get_id_discrete_rationale_function(
                removal_args["based_on"]
            )
        )

        thresholds = removal_args["thresholds"]
        last_id_top = None
        for v in thresholds:

            # Get rationale from score explanation
            id_top = get_discrete_rationale_function(score_explanation, v, only_pos)

            # If the rationale is the same, we do not include it. In this way, we will not consider in the average the same omission.
            if (
                id_top is not None
                and last_id_top is not None
                and set(id_top) == last_id_top
            ):
                id_top = None

            id_tops.append(id_top)

            if id_top is None:
                continue

            last_id_top = set(id_top)

            # Comprehensiveness
            # The only difference between comprehesivenss and sufficiency is the computation of the removal.

            # For the comprehensiveness: we remove the terms in the discrete rationale.
            sample = np.array(copy.copy(input_ids))

            if removal_args["remove_tokens"]:
                discrete_expl_th_token_ids = np.delete(sample, id_top)
            else:
                sample[id_top] = self.tokenizer.mask_token_id
                discrete_expl_th_token_ids = sample

            discrete_expl_th = self.tokenizer.decode(
                discrete_expl_th_token_ids, skip_special_tokens=False
            )

            discrete_expl_ths.append(discrete_expl_th)

        if discrete_expl_ths == list():
            return EvaluationMetricOutput(self, 0)

        # Prediction probability for the target and post process logits
        _, logits = self.helper._forward(discrete_expl_ths, output_hidden_states=False)
        logits = self.helper._postprocess_logits(
            logits, target_token_pos_idx=target_token_pos_idx
        )

        probs_removing = logits.softmax(-1)[:, target_pos_idx].cpu().numpy()

        # compute probability difference
        removal_importance = baseline - probs_removing
        #  compute AOPC comprehensiveness
        aopc_comprehesiveness = _compute_aopc(removal_importance)
        evaluation_output = EvaluationMetricOutput(self, aopc_comprehesiveness)
        return evaluation_output

    # def aggregate_score(self, score, total, **aggregation_args):
    #     return super().aggregate_score(score, total, **aggregation_args)


class AOPC_Sufficiency_Evaluation(BaseEvaluator):
    NAME = "aopc_sufficiency"
    SHORT_NAME = "aopc_suff"
    LOWER_IS_BETTER = True
    MIN_VALUE = -1.0
    MAX_VALUE = 1.0
    BEST_VALUE = 0.0
    METRIC_FAMILY = EvaluationMetricFamily.FAITHFULNESS

    def compute_evaluation(
        self, explanation: Explanation, **evaluation_args
    ) -> EvaluationMetricOutput:
        """Evaluate an explanation on the AOPC Sufficiency metric.

        Args:
            explanation (Explanation): the explanation to evaluate
            target (int): class label for which the explanation is evaluated
            evaluation_args (dict):  additional evaluation args

        Returns:
            Evaluation : the AOPC Sufficiency score of the explanation
        """

        remove_first_last, only_pos, removal_args, _ = parse_evaluator_args(
            evaluation_args
        )


        
        if isinstance(explanation, list):
            return None
        text = explanation.text
        score_explanation = explanation.scores
        target_pos_idx = explanation.target_pos_idx
        target_token_pos_idx = explanation.target_token_pos_idx
        helper_type = explanation.helper_type

        if (
            removal_args["remove_tokens"] == True
            and helper_type == "token-classification"
        ):
            removal_args["remove_tokens"] = False
            warnings.warn(
                "NER does not support token removal. 'remove_tokens' set to False"
            )

        # TO DO - use tokens
        # Get prediction probability of the input sencence for the target
        _, logits = self.helper._forward(text, output_hidden_states=False)
        logits = self.helper._postprocess_logits(
            logits, target_token_pos_idx=target_token_pos_idx
        )
        baseline = logits.softmax(-1)[0, target_pos_idx].item()

        # Tokenized sentence
        item = self.helper._tokenize(text)
        # Get token ids of the sentence
        input_len = item["attention_mask"].sum().item()
        input_ids = item["input_ids"][0][:input_len].tolist()

        # If remove_first_last, first and last token id (CLS and ) are removed
        if remove_first_last == True:
            input_ids = input_ids[1:-1]
            if self.tokenizer.cls_token == explanation.tokens[0]:
                score_explanation = score_explanation[1:-1]

        discrete_expl_ths = list()
        id_tops = list()

        get_discrete_rationale_function = (
            _check_and_define_get_id_discrete_rationale_function(
                removal_args["based_on"]
            )
        )

        thresholds = removal_args["thresholds"]
        last_id_top = None
        for v in thresholds:

            # Get rationale
            id_top = get_discrete_rationale_function(score_explanation, v, only_pos)
            # If the rationale is the same, we do not include it. In this way, we will not consider in the average the same omission.
            if (
                id_top is not None
                and last_id_top is not None
                and set(id_top) == last_id_top
            ):
                id_top = None

            id_tops.append(id_top)

            if id_top is None:
                continue

            last_id_top = set(id_top)

            # Sufficiency
            # The only difference between comprehesivenss and sufficiency is the computation of the removal.
            # For the sufficiency: we keep only the terms in the discrete rationale.

            sample = np.array(copy.copy(input_ids))

            # We take the tokens in the original order
            id_top = np.sort(id_top)

            if removal_args["remove_tokens"]:
                discrete_expl_th_token_ids = sample[id_top]
            else:
                mask_not_top = np.ones(sample.size, dtype=bool)
                mask_not_top[id_top] = False
                sample[mask_not_top] = self.tokenizer.mask_token_id
                discrete_expl_th_token_ids = sample
            ##############################################

            discrete_expl_th = self.tokenizer.decode(
                discrete_expl_th_token_ids, skip_special_tokens=False
            )
            discrete_expl_ths.append(discrete_expl_th)

        if discrete_expl_ths == []:
            return EvaluationMetricOutput(self, 1)

        # Prediction probability for the target
        _, logits = self.helper._forward(discrete_expl_ths, output_hidden_states=False)
        logits = self.helper._postprocess_logits(
            logits, target_token_pos_idx=target_token_pos_idx
        )
        probs_removing = logits.softmax(-1)[:, target_pos_idx].cpu().numpy()

        # Compute probability difference
        removal_importance = baseline - probs_removing

        aopc_sufficiency = _compute_aopc(removal_importance)

        evaluation_output = EvaluationMetricOutput(self, aopc_sufficiency)
        return evaluation_output

    # def aggregate_score(self, score, total, **aggregation_args):
    #     return super().aggregate_score(score, total, **aggregation_args)


class TauLOO_Evaluation(BaseEvaluator):
    NAME = "tau_leave-one-out_correlation"
    SHORT_NAME = "taucorr_loo"
    METRIC_FAMILY = EvaluationMetricFamily.FAITHFULNESS
    LOWER_IS_BETTER = False
    MAX_VALUE = 1.0
    MIN_VALUE = -1.0
    BEST_VALUE = 1.0

    def compute_evaluation(
        self, explanation: Explanation, **evaluation_args
    ) -> EvaluationMetricOutput:
        """Evaluate an explanation on the tau-LOO metric,
        i.e., the Kendall tau correlation between the explanation scores and leave one out (LOO) scores,
        computed by leaving one feature out and computing the change in the prediciton probability

        Args:
            explanation (Explanation): the explanation to evaluate
            target (int): class label for which the explanation is evaluated
            evaluation_args (dict):  additional evaluation args

        Returns:
            Evaluation : the tau-LOO score of the explanation
        """


        if isinstance(explanation, list):
            return None
            
        text = explanation.text
        score_explanation = explanation.scores
        target_pos_idx = explanation.target_pos_idx
        target_token_pos_idx = explanation.target_token_pos_idx
        helper_type = explanation.helper_type

        remove_first_last = evaluation_args.get("remove_first_last", True)

        if remove_first_last:
            if self.tokenizer.cls_token == explanation.tokens[0]:
                score_explanation = score_explanation[1:-1]

        _, logits = self.helper._forward(text, output_hidden_states=False)
        logits = self.helper._postprocess_logits(
            logits, target_token_pos_idx=target_token_pos_idx
        )

        baseline = logits.softmax(-1)[0, target_pos_idx].item()

        item = self.helper._tokenize(text)
        input_len = item["attention_mask"].sum().item()
        input_ids = item["input_ids"][0][:input_len].tolist()
        if remove_first_last == True:
            input_ids = input_ids[1:-1]

        # TODO: for a very long input these end up being many samples we need to process. Think about showing a progress bar here
        perturbation_helper = PertubationHelper(self.helper.tokenizer)
        samples_ids = perturbation_helper.edit_one_token(
            input_ids,
            strategy="remove" if helper_type != "token-classification" else "mask",
        )
        samples = self.helper.tokenizer.batch_decode(
            samples_ids, skip_special_tokens=False
        )

        _, logits = self.helper._forward(samples, output_hidden_states=False)
        logits = self.helper._postprocess_logits(
            logits, target_token_pos_idx=target_token_pos_idx
        )
        leave_one_out_removal = logits.softmax(-1)[:, target_pos_idx].cpu()

        occlusion_importance = leave_one_out_removal - baseline
        loo_scores = -1 * occlusion_importance.numpy()
        kendalltau_score = kendalltau(loo_scores, score_explanation)[0]

        evaluation_output = EvaluationMetricOutput(self, kendalltau_score)
        return evaluation_output

    # def aggregate_score(self, score, total, **aggregation_args):
    #     return super().aggregate_score(score, total, **aggregation_args)

class Sensitivity_Evaluation(BaseEvaluator):
    NAME = "auc_sensitivity"
    SHORT_NAME = "sens"

    LOWER_IS_BETTER = True
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0
    BEST_VALUE = 0.0
    METRIC_FAMILY = EvaluationMetricFamily.FAITHFULNESS

    def compute_evaluation(
        self, explanation: Explanation, **evaluation_args
    ) -> EvaluationMetricOutput:
        """Evaluate an explanation on the Sensitivity metric.

        Args:
            explanation (Explanation): the explanation to evaluate
            target (int): class label for which the explanation is evaluated
            evaluation_args (dict):  additional evaluation args.
                We currently support multiple approaches to define the hard rationale from
                soft score rationales, based on:
                - th : token greater than a threshold
                - perc : more than x% of the tokens
                - k: top k values

        Returns:
            Evaluation : the Sensitivity score of the explanation
        """
    
        remove_first_last, only_pos, _, top_k, name_model = parse_evaluator_args_model(
            evaluation_args
        )
        
        removal_args = {
        "remove_tokens": True,
        "based_on": "k"
        }
        if isinstance(explanation, list):
            return None
            
        text = explanation.text
        tokenizer = self.helper.tokenizer
        score_explanation = explanation.scores
        target_pos_idx = explanation.target_pos_idx
        target_token_pos_idx = explanation.target_token_pos_idx
        helper_type = explanation.helper_type

        if (
            removal_args["remove_tokens"] == True
            and helper_type == "token-classification"
        ):
            removal_args["remove_tokens"] = False
            warnings.warn(
                "Sensitivity does not support token removal. 'remove_tokens' set to False"
            )

        # Tokenized sentence
        item = self.helper._tokenize(text)
        # Get token ids of the sentence
        input_len = item["attention_mask"].sum().item()
        input_ids = item["input_ids"][0][:input_len].tolist()

        #if remove_first_last, first and last token id (CLS and ) are removed
        if remove_first_last == True:
            input_ids = input_ids[1:-1]
            if self.tokenizer.cls_token == explanation.tokens[0]:
                score_explanation = score_explanation[1:-1]

        # Supponiamo che name_model, tokenizer, text, score_explanation, get_discrete_explanation_topK, input_len siano già definiti.
        n = input_len - 1
        n = min(n, top_k)
        thresholds = [i for i in range(1, n)] 
        epsilons = []

        # Carica il modello di classificazione e il tokenizer
        name = name_model
        classification_model = AutoModelForSequenceClassification.from_pretrained(name)
        base_model = AutoModel.from_pretrained(name)

        # Tokenizza la frase
        inputs = tokenizer(text, return_tensors='pt')
        if remove_first_last == True:
            inputs = {
                'input_ids': inputs['input_ids'][:, 1:-1],
                'attention_mask': inputs['attention_mask'][:, 1:-1]
            }

        # Ottieni le rappresentazioni vettoriali (embedding) dei token
        with torch.no_grad():
            outputs = base_model(**inputs)
            token_embeddings = outputs.last_hidden_state

        for v in thresholds:
            # Get rationale
            perturbation_vector = get_discrete_explanation_topK(score_explanation, v, only_pos=False)
            
            # Aggiungi debug
            print(f"Threshold: {v}, Perturbation Vector: {perturbation_vector}")

            # Parametri per l'attacco PGD
            alpha = 0.1  # passo per ogni iterazione
            num_steps = 5  # numero di iterazioni

            # Ottieni la previsione originale del modello
            with torch.no_grad():
                original_logits = classification_model(**inputs).logits
                y = torch.argmax(original_logits, dim=-1)
            
            # Funzione di attacco PGD
            def pgd_attack(embeddings, epsilon, alpha, num_steps, perturbation_vector, y):
                perturbed_embeddings = embeddings.clone().detach().requires_grad_(True)

                # Funzione di ottimizzazione PGD
                optimizer = torch.optim.Adam([perturbed_embeddings], lr=alpha)

                for step in range(num_steps):
                    optimizer.zero_grad()
                    # Calcola le previsioni del modello di classificazione
                    classificator_inputs = {
                        'attention_mask': inputs['attention_mask'],
                        'inputs_embeds': perturbed_embeddings
                    }
                    logits = classification_model(**classificator_inputs).logits

                    # Calcola la perdita
                    loss = F.cross_entropy(logits, y)

                    # Calcola i gradienti
                    loss.backward()

                    # Applica il gradiente solo agli indici di perturbation_vector
                    with torch.no_grad():
                        grad = perturbed_embeddings.grad.clone()
        
                        # Ensure the mask has the correct shape
                        mask = torch.tensor(perturbation_vector, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                        mask = mask.expand_as(grad)  # Expand the mask to match the shape of grad
                        # print(mask.shape)
                        # torch.Size([1, 7, 1])
                        # torch.Size([1, 7, 768])
                        #print(grad)
                        
                        grad = grad * mask
                        #print(grad)

                        # Debugging gradiente
                        #print(f"Step: {step}, Epsilon: {epsilon}")

                        perturbed_embeddings += alpha * grad.sign()

                        # Proiezione per mantenere la perturbazione entro il limite epsilon
                        perturbation = torch.clamp(perturbed_embeddings - embeddings, -epsilon, epsilon)
                        perturbed_embeddings = (embeddings + perturbation).detach().requires_grad_(True)
                        
                    # Reset gradienti
                    classification_model.zero_grad()
                    if perturbed_embeddings.grad is not None:
                        perturbed_embeddings.grad.zero_()

                return perturbed_embeddings

            # Ricerca binaria per trovare il minimo epsilon che cambia la previsione del modello
            def binary_search_epsilon(token_embeddings, perturbation_vector, alpha, num_steps, y, tol=1e-3, max_iter=15):
                low = 0.0
                high = 1.0
                best_epsilon = low

                for i in range(max_iter):
                    mid = (low + high) / 2.0
                    perturbed_embeddings = pgd_attack(token_embeddings, mid, alpha, num_steps, perturbation_vector, y)

                    # Costruisci l'input finale per il modello di classificazione usando gli embeddings perturbati
                    classificator_inputs = {
                        'attention_mask': inputs['attention_mask'],
                        'inputs_embeds': perturbed_embeddings
                    }

                    # Ottieni le previsioni del modello di classificazione
                    with torch.no_grad():
                        classification_outputs = classification_model(**classificator_inputs)
                        logits = classification_outputs.logits
                        predictions = torch.argmax(logits, dim=-1)

                    # Debug della ricerca binaria
                    #print(f"Epsilon: {mid:.4f}, Predictions: {predictions}, Target: {y}, Iteration#: {i}")

                    # Controlla se la previsione è cambiata
                    if predictions != y:
                        best_epsilon = mid
                        high = mid
                    else:
                        low = mid

                    if high - low < tol:
                        break

                return best_epsilon

            # Trova il minimo epsilon che cambia la previsione del modello
            epsilon = binary_search_epsilon(token_embeddings, perturbation_vector, alpha, num_steps, y)
            epsilons.append(epsilon)

        sens_auc = np.trapz(epsilons, thresholds)
        print(f"Epsilons: {epsilons}, Sensitivity AUC: {sens_auc}")

        evaluation_output = EvaluationMetricOutput(self, sens_auc)
        return evaluation_output

        # def aggregate_score(self, score, total, **aggregation_args):
        #     return super().aggregate_score(score, total, **aggregation_args)
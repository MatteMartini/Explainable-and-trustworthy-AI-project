from typing import List, Union

import numpy as np

from furetto.explainers.explanation import Explanation, ExplanationWithRationale

from ..modeling import create_helper
from .evaluation import EvaluationMetricOutput
from .faithfulness_measures import AOPC_Comprehensiveness_Evaluation




from .plausibility_measures import AUPRC_PlausibilityEvaluation



from .confidence_measures import CI_Confidence_Evaluation



class AOPC_Comprehensiveness_Evaluation_by_class:
    NAME = "aopc_class_comprehensiveness"
    SHORT_NAME = "aopc_class_compr"
    # Higher is better
    BEST_SORTING_ASCENDING = False
    TYPE_METRIC = "class_faithfulness"

    def __init__(
        self,
        model,
        tokenizer,
        task_name,
        aopc_compr_eval: AOPC_Comprehensiveness_Evaluation = None,
    ):
        if aopc_compr_eval is None:
            if model is None or tokenizer is None:
                raise ValueError("Please specify a model and a tokenizer.")

            self.helper = create_helper(model, tokenizer, task_name)
            self.aopc_compr_eval = AOPC_Comprehensiveness_Evaluation(
                model, tokenizer, task_name
            )
        else:
            self.aopc_compr_eval = aopc_compr_eval

    def compute_evaluation(
        self,
        class_explanation: List[Union[Explanation, ExplanationWithRationale]],
        **evaluation_args
    ):

        """
        Each element of the list is the explanation for a target class
        """

        evaluation_args["only_pos"] = True

        aopc_values = []
        for target, explanation in enumerate(class_explanation):
            aopc_values.append(
                self.aopc_compr_eval.compute_evaluation(
                    explanation, target, **evaluation_args
                ).score
            )
        aopc_class_score = np.mean(aopc_values)
        evaluation_output = EvaluationMetricOutput(self.SHORT_NAME, aopc_class_score)
        return evaluation_output

    def aggregate_score(self, score, total, **aggregation_args):
        return score / total



class AUPRC_PlausibilityEvaluation_by_class:
   NAME = "auprc_class_plausibility"
   SHORT_NAME = "auprc_class_plaus"
   BEST_SORTING_ASCENDING = False
   TYPE_METRIC = "class_plausibility"

   def __init__(
       self,
       model,
       tokenizer,
       task_name,
       auprc_plaus_eval: AUPRC_PlausibilityEvaluation = None,
   ):
       if auprc_plaus_eval is None:
           if model is None or tokenizer is None:
               raise ValueError("Please specify a model and a tokenizer.")

           self.helper = create_helper(model, tokenizer, task_name)
           self.auprc_plaus_eval = AUPRC_PlausibilityEvaluation(
               model, tokenizer, task_name
           )
       else:
           self.auprc_plaus_eval = auprc_plaus_eval

   def compute_evaluation(
       self,
       class_explanation: List[Union[Explanation, ExplanationWithRationale]],
       **evaluation_args
   ):

       aup_values = []
       for target, explanation in enumerate(class_explanation):
           aup_values.append(
               self.auprc_plaus_eval.compute_evaluation(
                   explanation, target, **evaluation_args
               ).score
           )
       auprc_class_score = np.mean(aup_values)
       evaluation_output = EvaluationMetricOutput(self.SHORT_NAME, auprc_class_score)
       return evaluation_output

   def aggregate_score(self, score, total, **aggregation_args):
       return score / total
   






class CI_Confidence_Evaluation_by_class:
   NAME = "ci_class_confidence"
   SHORT_NAME = "ci_class_conf"
   BEST_SORTING_ASCENDING = False
   TYPE_METRIC = "class_confidence"

   def __init__(
       self,
       model,
       tokenizer,
       task_name,
       ci_conf_eval: CI_Confidence_Evaluation = None,
   ):
       if ci_conf_eval is None:
           if model is None or tokenizer is None:
               raise ValueError("Please specify a model and a tokenizer.")

           self.helper = create_helper(model, tokenizer, task_name)
           self.ci_conf_eval = CI_Confidence_Evaluation(
               model, tokenizer, task_name
           )
       else:
           self.ci_conf_eval = ci_conf_eval

   def compute_evaluation(
       self,
       class_explanation: List[Union[Explanation, ExplanationWithRationale]],
       **evaluation_args
   ):

       ci_values = []
       for target, explanation in enumerate(class_explanation):
           ci_values.append(
               self.ci_conf_eval.compute_evaluation(
                   explanation, target, **evaluation_args
               ).score
           )
       ci_class_score = np.mean(ci_values)
       evaluation_output = EvaluationMetricOutput(self.SHORT_NAME, ci_class_score)
       return evaluation_output

   def aggregate_score(self, score, total, **aggregation_args):
       return score / total

import pdb
from typing import Optional, Tuple, Union

import torch
from captum.attr import InputXGradient, IntegratedGradients, Saliency
from cv2 import multiply
import numpy as np

from . import BaseExplainer
from .explanation import Explanation
from .utils import parse_explainer_args


class GradientExplainer(BaseExplainer):
    NAME = "Gradient"

    def __init__(
        self,
        model,
        tokenizer,
        model_helper: Optional[str] = None,
        multiply_by_inputs: bool = True,
        **kwargs,
    ):
        super().__init__(model, tokenizer, model_helper, **kwargs)

        self.multiply_by_inputs = multiply_by_inputs
        if self.multiply_by_inputs:
            self.NAME += " (x Input)"

    def compute_feature_importance(
        self,
        text: Union[str, Tuple[str, str]],
        target: Union[int, str] = 1,
        target_token: Optional[Union[int, str]] = None,
        **kwargs,
    ):
        def func(input_embeds):
            outputs = self.helper.model(
                inputs_embeds=input_embeds, attention_mask=item["attention_mask"]
            )
            logits = self.helper._postprocess_logits(
                outputs.logits, target_token_pos_idx=target_token_pos_idx
            )
            return logits

        # Sanity checks
        target_pos_idx = self.helper._check_target(target)
        target_token_pos_idx = self.helper._check_target_token(text, target_token)
        text = self.helper._check_sample(text)

        item = self._tokenize(text)
        item = {k: v.to(self.device) for k, v in item.items()}
        input_len = item["attention_mask"].sum().item()
        dl = (
            InputXGradient(func, **self.init_args)
            if self.multiply_by_inputs
            else Saliency(func, **self.init_args)
        )

        inputs = self.get_input_embeds(text)

        # Initialize a dictionary to store scores for all classes
        all_token_scores = {}
        num_classes = self.helper.model.config.num_labels

        for class_idx in range(num_classes):
            attr = dl.attribute(inputs, target=class_idx, **kwargs)
            attr = attr[0, :input_len, :].detach().cpu()

            # Pool over hidden size
            attr = attr.sum(-1).numpy()

            all_token_scores[class_idx] = attr

        #print("Gred")
        # Ad esempio, stampa i punteggi di importanza per ciascuna classe
        # for class_idx, scores in all_token_scores.items():
        #     print(f"Class {class_idx} token scores: {scores}")



        # Creation of the output Explanation with all importances
        output = Explanation(
            text=text,
            tokens=self.get_tokens(text),
            scores=all_token_scores[target_pos_idx],
            all_scores=all_token_scores,  # Include all importances
            all_scores2={},  # Include all importances
            explainer=self.NAME,
            helper_type=self.helper.HELPER_TYPE,
            target_pos_idx=target_pos_idx,
            target_token_pos_idx=target_token_pos_idx,
            target=self.helper.model.config.id2label[target_pos_idx],
            target_token=self.helper.tokenizer.decode(
                item["input_ids"][0, target_token_pos_idx].item()
            )
            if self.helper.HELPER_TYPE == "token-classification"
            else None,
        )
        return output



class IntegratedGradientExplainer(BaseExplainer):
    NAME = "Integrated Gradient"

    def __init__(
        self,
        model,
        tokenizer,
        model_helper: Optional[str] = None,
        multiply_by_inputs: bool = True,
        **kwargs,
    ):
        super().__init__(model, tokenizer, model_helper, **kwargs)

        self.multiply_by_inputs = multiply_by_inputs
        if self.multiply_by_inputs:
            self.NAME += " (x Input)"

    def _generate_baselines(self, input_len):
        ids = (
            [self.helper.tokenizer.cls_token_id]
            + [self.helper.tokenizer.pad_token_id] * (input_len - 2)
            + [self.helper.tokenizer.sep_token_id]
        )
        embeddings = self.helper._get_input_embeds_from_ids(
            torch.tensor(ids, device=self.device)
        )
        return embeddings.unsqueeze(0)

    def compute_feature_importance(
        self,
        text: Union[str, Tuple[str, str]],
        target: Union[int, str] = 1,
        target_token: Optional[Union[int, str]] = None,
        show_progress: bool = False,
        **kwargs,
    ):
        # Sanity checks
        target_pos_idx = self.helper._check_target(target)
        target_token_pos_idx = self.helper._check_target_token(text, target_token)
        text = self.helper._check_sample(text)

        def func(input_embeds):
            attention_mask = torch.ones(
                *input_embeds.shape[:2], dtype=torch.uint8, device=self.device
            )
            _, logits = self.helper._forward_with_input_embeds(
                input_embeds, attention_mask, show_progress=show_progress
            )
            logits = self.helper._postprocess_logits(
                logits, target_token_pos_idx=target_token_pos_idx
            )
            return logits

        item = self._tokenize(text)
        input_len = item["attention_mask"].sum().item()
        dl = IntegratedGradients(
            func, multiply_by_inputs=self.multiply_by_inputs, **self.init_args
        )
        inputs = self.get_input_embeds(text)
        baselines = self._generate_baselines(input_len)

        # Initialize a dictionary to store scores for all classes
        all_token_scores = {}
        num_classes = self.helper.model.config.num_labels

        for class_idx in range(num_classes):
            attr = dl.attribute(inputs, baselines=baselines, target=class_idx, **kwargs)
            attr = attr[0, :input_len, :].detach().cpu()

            # Pool over hidden size
            attr = attr.sum(-1).numpy()

            all_token_scores[class_idx] = attr

        #print("Gred InntgrT")
        # Ad esempio, stampa i punteggi di importanza per ciascuna classe
        # for class_idx, scores in all_token_scores.items():
        #     print(f"Class {class_idx} token scores: {scores}")


        # Creation of the output Explanation with all importances
        output = Explanation(
            text=text,
            tokens=self.get_tokens(text),
            scores=all_token_scores[target_pos_idx],
            all_scores=all_token_scores,  # Include all importances
            all_scores2={},  # Include all importances
            explainer=self.NAME,
            helper_type=self.helper.HELPER_TYPE,
            target_pos_idx=target_pos_idx,
            target_token_pos_idx=target_token_pos_idx,
            target=self.helper.model.config.id2label[target_pos_idx],
            target_token=self.helper.tokenizer.decode(
                item["input_ids"][0, target_token_pos_idx].item()
            )
            if self.helper.HELPER_TYPE == "token-classification"
            else None,
        )
        return output

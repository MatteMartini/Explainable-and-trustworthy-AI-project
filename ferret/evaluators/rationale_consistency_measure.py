import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler

from . import BaseEvaluator, EvaluationMetricFamily
from .evaluation import EvaluationMetricOutput
from ..explainers.explanation import Explanation
from .utils_from_soft_to_discrete import (
    parse_evaluator_args_model,
)




# Function to register hook
def get_activation(name, activations_dict):
            def hook(model, input, output):
                if isinstance(output, torch.Tensor):
                    activations_dict[name] = output.detach()
                elif isinstance(output, dict) and "logits" in output:
                    activations_dict[name] = output["logits"].detach()  

            return hook



def get_activation_distances(model,model_rand,tokenizer,explanation,device):

        activations_model = {}
        activations_model_rand = {}

        # Register hooks on all layers for both models
        for name, module in model.named_modules():
            module.register_forward_hook(get_activation(name, activations_model))

        for name, module in model_rand.named_modules():
            module.register_forward_hook(get_activation(name, activations_model_rand))

        # Process input
        inputs = tokenizer(explanation.text, return_tensors="pt", padding=True, truncation=True).to(device)

        # Perform forward pass on both models
        with torch.no_grad():
            try:
                outputs_model = model(**inputs)
                outputs_model_rand = model_rand(**inputs)
            except Exception as e:
                print(f"Error during forward pass: {e}")
               

        # Compute activation distances across layers
        activation_distances = []
        for layer_name in activations_model.keys():
            activation_model = activations_model[layer_name].cpu().numpy().flatten()
            activation_model_rand = activations_model_rand[layer_name].cpu().numpy().flatten()
            activation_distance = np.mean((activation_model - activation_model_rand))
            
            activation_distances.append(activation_distance) 

        
        activation_distances_mean = np.mean(np.abs(activation_distances))
        return activation_distances_mean



def get_score_distances(explanation):
    
        # Extract saliency scores
        saliency_pred =explanation.all_scores
        saliency_predRand =explanation.all_scores_rand


        #remove first and last token
        for key in saliency_predRand:

             saliency_predRand[key] = saliency_predRand[key][1:-1]
             if len(saliency_predRand[key]) != len(saliency_pred[key]):
                  saliency_pred[key] = saliency_pred[key][1:-1] 
        
        

        keys = saliency_pred.keys()
        score_distances = []

        for key in keys:
            value1 = saliency_pred[key]
            value2 = saliency_predRand[key]

            if np.issubdtype(value1.dtype, np.number) and np.issubdtype(value2.dtype, np.number):
                score_distance = np.abs(value1 - value2)
                score_distances.append(score_distance)
            else:
                raise ValueError(f"Values for key '{key}' are not numeric: {value1}, {value2}")
        
        score_distances_mean = np.mean(score_distances)
        return score_distances_mean





class Rationale_ConsistencyEvaluation(BaseEvaluator):

    NAME = "rationale_consistency"
    SHORT_NAME = "rat_cons"
    LOWER_IS_BETTER = False
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0
    METRIC_FAMILY = EvaluationMetricFamily.CONSISTENCY  # Cambiato da FAITHFULNESS a CONSISTTENCY


    def compute_evaluation(self, explanation: Explanation, **evaluation_args):
        

        remove_first_last, only_pos, _, top_k, name_model = parse_evaluator_args_model(
            evaluation_args
        )
        
        if isinstance(explanation, list) == False:  
            return None
      
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the pre-trained model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(name_model).to(device)
        tokenizer = AutoTokenizer.from_pretrained(name_model,force_download=True)

        # Initialize the model with random weights
        model_rand = AutoModelForSequenceClassification.from_pretrained(name_model).to(device)

        # Re-initialize the weights to random values
        def initialize_weights(module):
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                module.reset_parameters()
            elif isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        model_rand.apply(initialize_weights)



        act_dist = []
        scr_dist = []

        for i in range(len(explanation)):

            activation_distances = get_activation_distances(model,model_rand,tokenizer,explanation[i],device)
            score_distances = get_score_distances(explanation[i])

            act_dist.append(activation_distances)
            scr_dist.append(score_distances)


        act_dist = np.array(act_dist)
        scr_dist = np.array(scr_dist)

        act_dist=MinMaxScaler().fit_transform([[_d] for _d in act_dist])
        scr_dist=MinMaxScaler().fit_transform([[_d] for _d in scr_dist])
        
        
        #Compute Correlation
        rho, _ = spearmanr(act_dist, scr_dist)
            

        return EvaluationMetricOutput(self, rho)


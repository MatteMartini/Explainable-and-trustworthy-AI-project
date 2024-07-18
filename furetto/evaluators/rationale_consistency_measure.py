import numpy as np
from scipy.special import softmax
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer,BertConfig
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler

from . import BaseEvaluator, EvaluationMetricFamily
from .evaluation import EvaluationMetricOutput
from ..explainers.explanation import Explanation





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
            
            activation_distances.append(activation_distance)  # Aggiungi a lista

        
        activation_distances_mean = np.mean(np.abs(activation_distances))
        return activation_distances_mean



def get_score_distances(explanation):
    
        # Extract saliency scores
        saliency_pred =explanation.all_scores
        saliency_predRand =explanation.all_scores2


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

            # Verifica che i valori siano numerici e array numpy
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
        
        
        if isinstance(explanation, list) == False:  
            return None
      
        
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the pre-trained model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize the model with random weights
        model_rand = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

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

        
        
        #Compute Correlation
        rho, _ = spearmanr(act_dist, scr_dist)
        #rho = (activation_distances_mean - score_distances_mean)
        
        

        return EvaluationMetricOutput(self, rho)





        



# VERSIONE LOGIT

# def compute_evaluation(self, explanation: Explanation, **evaluation_args):
#         model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
#         device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         # Load the pre-trained model and tokenizer
#         model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
        
#         # Initialize the model with random weights
#         model_rand = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        
#         # Re-initialize the weights to random values
#         def initialize_weights(module):
#             if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
#                 module.reset_parameters()
#             elif isinstance(module, torch.nn.LayerNorm):
#                 module.bias.data.zero_()
#                 module.weight.data.fill_(1.0)

#         model_rand.apply(initialize_weights)

#         # Process input
#         inputs = tokenizer(explanation.text, return_tensors="pt", padding=True, truncation=True).to(device)
        
#         # Perform forward pass on both models
#         with torch.no_grad():
#             try:
#                 outputs_model = model(**inputs)
#                 outputs_model_rand = model_rand(**inputs)
#             except Exception as e:
#                 print(f"Error during forward pass: {e}")
#                 return EvaluationMetricOutput(self, float('nan'))
        
#         # Extract logits (final layer output before classification)
#         logits_model = outputs_model.logits.detach().cpu().numpy().ravel()
#         logits_model_rand = outputs_model_rand.logits.detach().cpu().numpy().ravel()

#         print(logits_model)
#         print(logits_model_rand)

#         # Calculate the distance between logits of trained and random models
#         activation_distances = np.abs(logits_model - logits_model_rand)

#         # Extract saliency scores
#         saliency_pred = np.array(explanation.scores)
        
#         # Compute score distance
#         score_distance = np.abs(saliency_pred - explanation.scores[0])

#         print("score distance")
#         print(score_distance)
#         print("act distance")
#         print(activation_distances)
        
#         # Ensure score_distance has the same length as activation_distances
#         if len(score_distance) != len(activation_distances):
#             min_length = min(len(score_distance), len(activation_distances))
#             score_distance = score_distance[:min_length]
#             activation_distances = activation_distances[:min_length]
        
#         # Normalize distances
#         scaler = MinMaxScaler()
#         normalized_activation_distances = scaler.fit_transform(activation_distances.reshape(-1, 1)).flatten()
#         normalized_score_distance = scaler.fit_transform(score_distance.reshape(-1, 1)).flatten()

#         # Compute Spearman correlation
#         rho, p_value = spearmanr(normalized_activation_distances, normalized_score_distance)
        
#         return EvaluationMetricOutput(self, rho)







# Matteo 1

    # def compute_evaluation(self, explanation: Explanation, **evaluation_args):
    #     # Estrazione delle variabili dall'oggetto Explanation
    #     scores = explanation.scores #COSI PRENDI SOLO QUELLI DELLA CLASSE PREDETTA, IL CUI INDICE è DATO DA target_pos_idx
    #     all_scores = explanation.scores
    #     target_pos_idx = explanation.target_pos_idx  #indice classe da spiegare

    #     # Ottenere i logits e la predizione
    #     _, logits = self.helper._forward(explanation.text, output_hidden_states=False)
    #     logits = self.helper._postprocess_logits(
    #         logits, target_token_pos_idx=explanation.target_token_pos_idx
    #     )

    #     true_confidence = logits.softmax(-1)[0, target_pos_idx].item() #questo ti da il softmax(logit) della classe target come lo fa lui, quello di sotto commentato della classe con prob piu alta 
    #     # Calcolare class_preds e classi uniche

    #     instance_logits = softmax(logits)[0] #trasformo i 3 logits dell'istanza in probabilita

        
    #     print(instance_logits)
    #    #print(target_pos_idx) #=1
    #     print(true_confidence)

    #     y = instance_logits[target_pos_idx]
    #     print(y)
    #     saliency_pred = np.array(scores)

    #     other_sals = []

    #     for class_idx in range(len(all_scores)):
    #         if class_idx != target_pos_idx:
    #             other_sals.append(np.array(all_scores[class_idx]))

    #     feats = []

    #     if len(logits) == 2:
    #         feats.append(sum(saliency_pred - other_sals[0]))
    #         feats.append(sum(saliency_pred - other_sals[0]))
    #         feats.append(sum(saliency_pred - other_sals[0]))
    #     else:
    #         feats.append(sum(np.max([saliency_pred - other_sals[0],
    #                                 saliency_pred - other_sals[1]], axis=0)))
    #         feats.append(sum(np.mean([saliency_pred - other_sals[0],
    #                                 saliency_pred - other_sals[1]], axis=0)))
    #         feats.append(sum(np.min([saliency_pred - other_sals[0],
    #                                 saliency_pred - other_sals[1]], axis=0)))
    #         # features è una lista di vettori di caratteristiche che rappresentano le differenze di saliency tra la classe predetta e le altre classi

    #     features = np.array([feats])

    #     y = np.array([y]) #CHE IN REALTà HA SEMPLICEMENTE UN VALORE, CHE è LA SOFTMAX SULLA LOGIT PIU ALTA, E CIOE QUELLA RELATIVA ALLA CLASSE PREDETTA CHE SARà ASSEGNATA ALL'ISTANZA(FRASE)

    #     # Normalizzazione delle caratteristiche (CHE è UN VETTORE DI 3 ELEMENTI)
    #     features = MinMaxScaler().fit_transform(features)

    #     # Addestramento del modello
    #     reg = LinearRegression().fit(features, y)

    #     # Predizione sulla stessa istanza (non è una pratica comune, ma è l'unica opzione con una singola istanza)
    #     pred = reg.predict(features)

    #     # Calcolo delle metriche
    #     confidence_pred = pred[0]
    #     mean_abs_error = np.abs(y[0] - confidence_pred)

    #     print(mean_abs_error)
    #     return EvaluationMetricOutput(self, mean_abs_error)
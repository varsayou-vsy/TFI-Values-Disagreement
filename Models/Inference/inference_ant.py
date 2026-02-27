import numpy as np
import pandas as pd
import json, sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

sys.path.append("/Proyecto/Value-disagreement/Python/Utilities")
import Dict_Object #, text_cleansing



# Thresholds per value
with open("/Proyecto/Value-disagreement/Python/Models/best_thresholds_per_model.json") as f:
    thresholds = json.load(f)

# Best model per value
with open("/Proyecto/Value-disagreement/Python/Models/best_model_per_value.json") as f:
    best_model_per_value = json.load(f)



# Create GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
roberta_model = AutoModelForSequenceClassification.from_pretrained("/Proyecto/Value-disagreement/Python/Models/Results/roberta-base_table-valueALL_seed0/checkpoint-11617/")
deberta_model = AutoModelForSequenceClassification.from_pretrained("/Proyecto/Value-disagreement/Python/Models/Results/microsoft/deberta-v3-base_table-valueALL_seed4/checkpoint-8712/")

# Move models to GPU
roberta_model.to(device)
deberta_model.to(device)



#roberta_tokenizer = AutoTokenizer.from_pretrained("/Proyecto/Value-disagreement/Python/Models/Results/roberta-base_table-valueALL_seed0/checkpoint-11617/")
roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# Add special value tokens (10 values)
roberta_tokenizer.add_special_tokens({"additional_special_tokens": [f"<{x}>" for x in Dict_Object.ValueConstants.SCHWARTZ_VALUES]})

#deberta_tokenizer = AutoTokenizer.from_pretrained("/Proyecto/Value-disagreement/Python/Models/Results/microsoft/deberta-v3-base_table-valueALL_seed4/checkpoint-8712/")
deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
# Add special value tokens (10 values)
deberta_tokenizer.add_special_tokens({"additional_special_tokens": [f"<{x}>" for x in Dict_Object.ValueConstants.SCHWARTZ_VALUES]})



# INFERENCE

df_all = pd.read_csv(r"/Proyecto/Value-disagreement/Datos/osfstorage-archive/cleaned/deba_comments_final_sample.csv",
                     sep="|"
                     )

for v in Dict_Object.ValueConstants.SCHWARTZ_VALUES:

    df_all['value']=v
    df_all

    # Formated input strings for each row: <VALUE> [SEP] TEXT
    roberta_formatted_texts = [f"<{row['value']}> {roberta_tokenizer.sep_token} {row['body_cleand']}"
                            for _, row in df_all.iterrows()]

    deberta_formatted_texts = [f"<{row['value']}> {deberta_tokenizer.sep_token} {row['body_cleand']}"
                            for _, row in df_all.iterrows()]

    batch_size = 128
    preds, probs, roberta_prb, deberta_prb, ensemble_prb = [],[],[],[],[]

    for i in range(0, len(deberta_formatted_texts), batch_size):
        roberta_batch_texts = roberta_formatted_texts[i:i+batch_size]
        deberta_batch_texts = deberta_formatted_texts[i:i+batch_size]
        batch_values = df_all["value"].iloc[i:i+batch_size].tolist()

        # Tokenize
        roberta_inputs = roberta_tokenizer(roberta_batch_texts, padding='max_length', max_length=256, truncation=True, return_tensors="pt").to(device)
        deberta_inputs = deberta_tokenizer(deberta_batch_texts, padding='max_length', max_length=256, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            roberta_logits = roberta_model(**roberta_inputs).logits
            deberta_logits = deberta_model(**deberta_inputs).logits
        
        # Convert logits to probabilities
        roberta_probs = torch.sigmoid(roberta_logits).squeeze(-1)
        deberta_probs = torch.sigmoid(deberta_logits).squeeze(-1)
        ensemble_probs = (roberta_probs + deberta_probs) / 2
        
        for j, value in enumerate(batch_values):
            best_model = best_model_per_value[value]
            #best_model = 'roberta-base'
            #best_model = 'microsoft_deberta-v3-base'
            #best_model = 'ensemble'
            threshold = thresholds[best_model][value]

            if best_model == "roberta-base":
                prob = roberta_probs[j].item()
            elif best_model == "microsoft_deberta-v3-base":
                prob = deberta_probs[j].item()
            else:
                prob = ensemble_probs[j].item()
                
            pred = int(prob >= threshold)

            #print(f"Value: {value}, Threshold Used: {threshold}, Prob: {prob:.3f}, Pred: {pred}")
            
            preds.append(pred)
            probs.append(prob)
            
    # Save predictions
    df_all["pred"] = preds
    df_all["prob"] = probs

    print("RoBERTa:")
    print(f"Mean prob: {roberta_probs.mean():.3f}, Std: {roberta_probs.std():.3f}")

    print("DeBERTa:")
    print(f"Mean prob: {deberta_probs.mean():.3f}, Std: {deberta_probs.std():.3f}")

    print("Ensemble:")
    print(f"Mean prob: {ensemble_probs.mean():.3f}, Std: {ensemble_probs.std():.3f}")

    print(df_all['pred'].value_counts())

    print(df_all[df_all['pred']==1].sample(15)[['body_cleand','pred','prob']])

    df_all[['id', 'author','topic', 'body_cleand','value', 'pred', 'prob']].to_csv(f"/Proyecto/Value-disagreement/Python/Models/Inference/deba_usr_comments/deba_{v}_usrs_comments.csv",
                                                                                   header=True, index=False, quoting=csv.QUOTE_ALL, quotechar='"', sep="|", escapechar="|"
                                                                                   )

    df_all.drop(columns=['value', 'pred', 'prob'], inplace=True)

    #del roberta_model, deberta_model, roberta_inputs, deberta_inputs
    torch.cuda.empty_cache()
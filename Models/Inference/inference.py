import numpy as np
import pandas as pd
import json, sys, os, csv, logging
from tqdm import tqdm

import torch
import torch.nn.functional as F
#from torch.cuda.amp import autocast
from torch import autocast

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

sys.path.append("/Proyecto/Value-disagreement/Python/Utilities")
import Dict_Object #, text_cleansing



# Setup logging
logging.basicConfig(filename="inference_log.txt",
                    filemode="a",
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s"
                    )

# Toggle precision and batch-saving
use_fp16 = True
save_per_batch = True

# Output directory (per-value)
output_dir = "/Proyecto/Value-disagreement/Python/Models/Inference/deba_usr_comments"
os.makedirs(output_dir, exist_ok=True)



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

df_all = pd.read_csv(r"/Proyecto/Value-disagreement/Datos/osfstorage-archive/cleaned/deba_comments_final_all.csv",
                     sep="|"
                     )
#for v in ['SELF-DIRECTION']:
for v in Dict_Object.ValueConstants.SCHWARTZ_VALUES:

    output_path = os.path.join(output_dir, f"deba_{v}_usrs_comments.csv")

    # Initialize output CSV with header if file doesn't exist
    if not os.path.exists(output_path):
        with open(output_path, "w", newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out, delimiter='|', quoting=csv.QUOTE_ALL, quotechar='"', escapechar="|")
            writer.writerow(['id','author','body_cleand','value','pred','prob'])

    df_all['value']=v

    # Formated input strings for each row: <VALUE> [SEP] TEXT
    roberta_formatted_texts = [f"<{row['value']}> {roberta_tokenizer.sep_token} {row['body_cleand']}"
                            for _, row in df_all.iterrows()]

    deberta_formatted_texts = [f"<{row['value']}> {deberta_tokenizer.sep_token} {row['body_cleand']}"
                            for _, row in df_all.iterrows()]

    batch_size = 256
    
    # Progress bar setup
    num_batches = (len(df_all) + batch_size - 1) // batch_size
    progress_bar = tqdm(range(0, len(df_all), batch_size), desc=f"Processing {v}", ncols=100)

    #for i in range(0, len(deberta_formatted_texts), batch_size):
    for i in progress_bar:
        roberta_batch_texts = roberta_formatted_texts[i:i+batch_size]
        deberta_batch_texts = deberta_formatted_texts[i:i+batch_size]
        batch_df = df_all.iloc[i:i+batch_size].copy()
        batch_values = batch_df["value"].tolist()


        batch_df = df_all.iloc[i:i+batch_size].copy()
        batch_values = batch_df["value"].tolist()



        # Tokenize
        roberta_inputs = roberta_tokenizer(roberta_batch_texts, padding='max_length', max_length=256, truncation=True, return_tensors="pt").to(device)
        deberta_inputs = deberta_tokenizer(deberta_batch_texts, padding='max_length', max_length=256, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            if use_fp16:
                #with autocast():
                with autocast("cuda"):
                    roberta_logits = roberta_model(**roberta_inputs).logits
                    deberta_logits = deberta_model(**deberta_inputs).logits
            else:
                roberta_logits = roberta_model(**roberta_inputs).logits
                deberta_logits = deberta_model(**deberta_inputs).logits

        # Convert logits to probabilities
        roberta_probs = torch.sigmoid(roberta_logits).squeeze(-1)
        deberta_probs = torch.sigmoid(deberta_logits).squeeze(-1)
        ensemble_probs = (roberta_probs + deberta_probs) / 2
        
        batch_preds, batch_probs = [], []

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
            batch_preds.append(pred)
            batch_probs.append(prob)

            #print(f"Value: {value}, Threshold Used: {threshold}, Prob: {prob:.3f}, Pred: {pred}")
            
        # Write batch results to CSV
        if save_per_batch:
            batch_df["pred"] = batch_preds
            batch_df["prob"] = batch_probs

            with open(output_path, "a", newline='', encoding='utf-8') as f_out:
                writer = csv.writer(f_out, delimiter='|', quoting=csv.QUOTE_ALL, quotechar='"', escapechar="|")
                for _, row in batch_df.iterrows():
                    writer.writerow([row['id'], row['author'],row['body_cleand'], row['value'],row['pred'], row['prob']])

        # Optional logging (every 100 batches)
        if i % (100 * batch_size) == 0:
            logging.info(f"{v}: Processed {i} rows")

    # Final log entry per value
    logging.info(f"Completed inference for value '{v}' — saved to {output_path}")

    df_all.drop(columns=['value'], inplace=True)

    #del roberta_model, deberta_model, roberta_inputs, deberta_inputs
    torch.cuda.empty_cache()
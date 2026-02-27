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

def _run_logits(model, inputs, device, use_fp16: bool):
    """Forward pass returning logits with AMP if CUDA."""
    with torch.inference_mode():
        if use_fp16 and device.type == "cuda":
            with torch.autocast(device_type="cuda"):
                return model(**inputs).logits
        else:
            return model(**inputs).logits

def safe_sigmoid_probs(model, tokenizer, texts, device, use_fp16: bool, max_length: int = 256):
    """
    Tokenize + forward -> sigmoid probs.
    If CUDA OOM happens, split the batch and retry recursively.
    """
    # Tokenize
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
        logits = _run_logits(model, inputs, device, use_fp16)
        probs = torch.sigmoid(logits).squeeze(-1)
        # free ASAP
        del logits, inputs
        return probs

    except torch.cuda.OutOfMemoryError:
        # Important: clear cached blocks and retry with smaller chunks
        del inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

        n = len(texts)
        if n <= 1:
            raise  # can't split further

        mid = n // 2
        p1 = safe_sigmoid_probs(model, tokenizer, texts[:mid], device, use_fp16, max_length)
        p2 = safe_sigmoid_probs(model, tokenizer, texts[mid:], device, use_fp16, max_length)
        return torch.cat([p1, p2], dim=0)

# Toggle precision and batch-saving
use_fp16 = True
save_per_batch = True

# Output directory (per-value)
output_dir = "/Proyecto/Value-disagreement/Python/Models/Inference/deba_usr_comments"
os.makedirs(output_dir, exist_ok=True)

# Thresholds per value
with open("/Proyecto/Value-disagreement/Python/Models/best_threshold_per_value.json") as f:
    best_threshold_per_value = json.load(f)

# Best model per value
with open("/Proyecto/Value-disagreement/Python/Models/best_model_per_value.json") as f:
    best_model_per_value = json.load(f)


# Sanity check keys
#for v in Dict_Object.ValueConstants.SCHWARTZ_VALUES:
for v in ["conformity"]:
    k = v.upper()
    assert k in best_model_per_value, f"Missing value in best_model_per_value: {k}"
    assert k in best_threshold_per_value, f"Missing threshold in best_threshold_per_value: {k}"
    mc = best_model_per_value[k]
    assert mc in ["roberta-base", "microsoft_deberta-v3-base", "ensemble_avg"], f"Unexpected model_choice: {mc}"


# Create GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
roberta_model = AutoModelForSequenceClassification.from_pretrained("/Proyecto/Value-disagreement/Python/Models/Results/roberta-base_table-valueALL_seed1/checkpoint-11617")
deberta_model = AutoModelForSequenceClassification.from_pretrained("/Proyecto/Value-disagreement/Python/Models/Results/microsoft/deberta-v3-base_table-valueALL_seed0/checkpoint-13069")

# Move models to GPU
roberta_model.to(device)
deberta_model.to(device)

roberta_model.eval()
deberta_model.eval()

roberta_tokenizer = AutoTokenizer.from_pretrained("/Proyecto/Value-disagreement/Python/Models/Results/roberta-base_table-valueALL_seed1")
"""roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Add special value tokens (10 values)
roberta_tokenizer.add_special_tokens({"additional_special_tokens": [f"<{x}>" for x in Dict_Object.ValueConstants.SCHWARTZ_VALUES]})
roberta_model.resize_token_embeddings(len(roberta_tokenizer))"""

deberta_tokenizer = AutoTokenizer.from_pretrained("/Proyecto/Value-disagreement/Python/Models/Results/microsoft/deberta-v3-base_table-valueALL_seed0")
"""deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

# Add special value tokens (10 values)
deberta_tokenizer.add_special_tokens({"additional_special_tokens": [f"<{x}>" for x in Dict_Object.ValueConstants.SCHWARTZ_VALUES]})
deberta_model.resize_token_embeddings(len(deberta_tokenizer))"""

assert "<achievement>" in roberta_tokenizer.get_vocab(), "RoBERTa tokenizer missing <achievement>"
assert "<achievement>" in deberta_tokenizer.get_vocab(), "DeBERTa tokenizer missing <achievement>"


# TEST
"""def get_dataset(table):
    # --- ### VALUENET    
    if table == "valueALL":
        value_set = pd.read_csv(r"/Proyecto/Value-disagreement/Datos/valueALL/value_all.csv", sep='|')

        value_train_set = np.loadtxt(r"/Proyecto/Value-disagreement/Datos/valueALL/value_all_train.csv", delimiter=',', dtype=int)
        value_val_set = np.loadtxt(r"/Proyecto/Value-disagreement/Datos/valueALL/value_all_val.csv", delimiter=',', dtype=int)
        value_test_set = np.loadtxt(r"/Proyecto/Value-disagreement/Datos/valueALL/value_all_test.csv", delimiter=',', dtype=int)

    return value_set, value_train_set, value_val_set, value_test_set

# cargar todo
value_set, value_train_set, value_val_set, value_test_set = get_dataset("valueALL")


df_all = value_set.iloc[value_test_set].reset_index(drop=True)
df_all = df_all.rename(columns={"scenario": "body_cleand", "uid": "id"})
df_all["author"] = df_all["id"].astype(str)
df_all"""


# INFERENCE
df_all = pd.read_csv(r"/Proyecto/Value-disagreement/Datos/osfstorage-archive/cleaned/deba_comments_final_all.csv", sep="|")


# Pre-carga del texto una vez (rápido y evita conversiones repetidas)
all_bodies = df_all["body_cleand"].astype(str).tolist()

#for v in Dict_Object.ValueConstants.SCHWARTZ_VALUES:
for v in ["conformity"]:

    output_path = os.path.join(output_dir, f"deba_{v}_usrs_comments.csv")

    # Initialize output CSV with header if file doesn't exist
    if not os.path.exists(output_path):
        with open(output_path, "w", newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out, delimiter='|', quoting=csv.QUOTE_ALL, quotechar='"', escapechar="|")
            #writer.writerow(['id','author','body_cleand','value','pred','prob'])
            writer.writerow(['id','author','value','pred','prob'])

    #df_all['value'] = v

    # Formated input strings for each row: <VALUE> [SEP] TEXT
    #roberta_formatted_texts = [f"<{row['value']}> {roberta_tokenizer.sep_token} {row['body_cleand']}"
    #                        for _, row in df_all.iterrows()]

    #deberta_formatted_texts = [f"<{row['value']}> {deberta_tokenizer.sep_token} {row['body_cleand']}"
    #                        for _, row in df_all.iterrows()]

    # definir modelo por valor
    model_choice = best_model_per_value[v.upper()]
    use_roberta = model_choice in ["roberta-base", "ensemble_avg"]
    use_deberta = model_choice in ["microsoft_deberta-v3-base", "ensemble_avg"]

    batch_size = 256
    
    # Progress bar setup
    num_batches = (len(df_all) + batch_size - 1) // batch_size
    progress_bar = tqdm(range(0, len(df_all), batch_size), desc=f"Processing {v}", ncols=100)
    
    #for i in range(0, len(deberta_formatted_texts), batch_size):
    for i in progress_bar:
        #roberta_batch_texts = roberta_formatted_texts[i:i+batch_size]
        #deberta_batch_texts = deberta_formatted_texts[i:i+batch_size]
        #batch_df = df_all.iloc[i:i+batch_size].copy()
        #batch_values = batch_df["value"].tolist()

        batch_df = df_all.iloc[i:i+batch_size][['id','author']].copy()
        batch_df["value"] = v

        # construimos SOLO el batch (sin listas globales)
        batch_bodies = all_bodies[i:i+batch_size]
        #batch_values = [v] * len(batch_bodies)  # antes venía de batch_df["value"], pero ya no existe

        # Version OPTimizada por consumo de ensamble
        roberta_batch_texts = None
        deberta_batch_texts = None
        
        if use_roberta:
            roberta_batch_texts = [f"<{v}> {roberta_tokenizer.sep_token} {txt}" for txt in batch_bodies]
            #roberta_inputs = roberta_tokenizer(roberta_batch_texts, padding=True, max_length=256, truncation=True, return_tensors="pt").to(device)
            roberta_probs = safe_sigmoid_probs(roberta_model, roberta_tokenizer, roberta_batch_texts, device=device, use_fp16=use_fp16, max_length=256)
            # liberar textos (CPU) si quieres
            roberta_batch_texts = None
            #if device.type == "cuda":
                #torch.cuda.empty_cache()
                                                                                                                                           
        if use_deberta:
            deberta_batch_texts = [f"<{v}> {deberta_tokenizer.sep_token} {txt}" for txt in batch_bodies]
            #deberta_inputs = deberta_tokenizer(deberta_batch_texts, padding=True, max_length=256, truncation=True, return_tensors="pt").to(device)
            deberta_probs = safe_sigmoid_probs(deberta_model, deberta_tokenizer, deberta_batch_texts,device=device, use_fp16=use_fp16, max_length=256)
            deberta_batch_texts = None
            #if device.type == "cuda":
                #torch.cuda.empty_cache()

        
        # probabilidad final
        if model_choice == "roberta-base":
            final_probs = roberta_probs
        elif model_choice == "microsoft_deberta-v3-base":
            final_probs = deberta_probs
        elif model_choice == "ensemble_avg":
            final_probs = (roberta_probs + deberta_probs) / 2
        else:
            raise ValueError(f"Unknown model_choice: {model_choice}")

        batch_preds, batch_probs = [], []

        #for j, value in enumerate(batch_values):
        #    threshold = thresholds[model_choice][value.upper()]
        threshold = best_threshold_per_value[v.upper()]
        for j in range(len(batch_bodies)):

            prob = final_probs[j].item()
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
                #for _, row in batch_df.iterrows():
                #    writer.writerow([row['id'], row['author'],row['body_cleand'], row['value'],row['pred'], row['prob']])
                for _, row in batch_df.iterrows():
                    writer.writerow([row['id'], row['author'], row["value"], row['pred'], row['prob']])



        # Optional logging (every 100 batches)
        if i % (100 * batch_size) == 0:
            logging.info(f"{v}: Processed {i} rows")

    # Final log entry per value
    logging.info(f"Completed inference for value '{v}' — saved to {output_path}")

    #df_all.drop(columns=['value'], inplace=True)

    #del roberta_model, deberta_model, roberta_inputs, deberta_inputs
    torch.cuda.empty_cache()
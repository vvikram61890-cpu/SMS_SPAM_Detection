import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset, DatasetDict
import torch
import os
import nltk
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments
)

# --- NLTK Setup (Ensure packages are available for legacy check, though BERT doesn't use them) ---
required_nltk_packages = ['stopwords', 'wordnet'] 
for package in required_nltk_packages:
    try:
        nltk.data.find(f'corpora/{package}')
    except LookupError:
        nltk.download(package, quiet=True)

# --- Configuration ---
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3 
RANDOM_SEED = 42
OUTPUT_DIR = "./bert_spam_detector_final"
LEARNING_RATE = 2e-5 

# --- I. Data Loading and Preparation ---

# 1. Load the data 
try:
    df = pd.read_csv('SMSSpamCollection', 
                     sep='\t', 
                     header=None, 
                     names=['label_str', 'sms'],  
                     encoding='latin-1')
except FileNotFoundError:
    print("Error: SMSSpamCollection file not found. Please ensure the file is in the project directory.")
    exit()

# 2. Encode Labels: 'ham' -> 0, 'spam' -> 1
label_map = {'ham': 0, 'spam': 1}
df['label'] = df['label_str'].map(label_map)

# EDA and cleanup
df['length'] = df['sms'].apply(len)
df = df.drop(columns=['label_str']) 

print("Label Mapping:", label_map) 
print("Total Messages:", len(df))

# 3. Create Train, Validation, and Test Split (80/10/10)
X_train, X_temp, y_train, y_temp = train_test_split(
    df['sms'], df['label'], test_size=0.2, random_state=RANDOM_SEED, stratify=df['label']
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
)

# 4. Convert to Hugging Face Dataset format
raw_datasets = DatasetDict({
    'train': Dataset.from_dict({'sms': X_train.tolist(), 'label': y_train.tolist()}),
    'validation': Dataset.from_dict({'sms': X_val.tolist(), 'label': y_val.tolist()}),
    'test': Dataset.from_dict({'sms': X_test.tolist(), 'label': y_test.tolist()})
})


# --- II. Tokenization and Encoding ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["sms"], 
        truncation=True, 
        padding="max_length", 
        max_length=MAX_LENGTH
    )

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

tokenized_datasets = tokenized_datasets.remove_columns(["sms"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# --- III. Model Training (Fine-Tuning using Trainer API) ---

# 1. Load Model for Sequence Classification 
id2label = {0: "HAM", 1: "SPAM"}
label2id = {"HAM": 0, "SPAM": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# 2. Define Evaluation Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        # Target 'spam' (label 1)
        "f1": f1_score(labels, predictions, pos_label=1, zero_division=0),
        "precision": precision_score(labels, predictions, pos_label=1, zero_division=0),
        "recall": recall_score(labels, predictions, pos_label=1, zero_division=0),
    }
    return metrics

# 3. Define Training Arguments (COMPATIBLE WITH OLDER TRANSFORMERS)
training_args = TrainingArguments(
    output_dir=f"./{OUTPUT_DIR}/results",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=f"./{OUTPUT_DIR}/logs",
    logging_steps=50,
    # For older transformers versions, use evaluation_strategy instead of eval_strategy
    evaluation_strategy="epoch", 
    save_strategy="epoch",       
    load_best_model_at_end=True,
    metric_for_best_model="f1", 
    seed=RANDOM_SEED
)

# 4. Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 5. Train the Model
print("\nStarting BERT Fine-Tuning... (This will take significant time)")
trainer.train()


# --- IV. Final Evaluation on Test Set ---
print("\n--- Final BERT Evaluation on Test Set ---")
test_results = trainer.evaluate(tokenized_datasets["test"])

accuracy = test_results['eval_accuracy']
f1 = test_results['eval_f1']
precision = test_results['eval_precision']
recall = test_results['eval_recall']

print(f"Final Test Accuracy: {accuracy:.4f}")
print(f"Final Test F1-Score (Spam): {f1:.4f}")
print(f"Final Test Precision (Spam): {precision:.4f}")
print(f"Final Test Recall (Spam): {recall:.4f}")
print("-" * 40)


# --- V. Artifact Saving for Deployment ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
print(f"\nSaving final model and tokenizer to: {OUTPUT_DIR}...")

# 1. Save the model
trainer.save_model(OUTPUT_DIR)

# 2. Save the tokenizer 
tokenizer.save_pretrained(OUTPUT_DIR)

print("BERT model and tokenizer saved successfully. Next step: Update Streamlit (app.py) to use this model!")
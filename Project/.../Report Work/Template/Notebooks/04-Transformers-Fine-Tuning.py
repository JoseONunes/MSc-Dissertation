#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datasets import load_from_disk

# Load tokenised dataset
dataset_dict = load_from_disk("../data/processed/tokenised_asap_split")

# Make sure labels column is correct
if "score_scaled" in dataset_dict["train"].features:
    dataset_dict = dataset_dict.rename_column("score_scaled", "labels")


# In[2]:


from transformers import RobertaForSequenceClassification, TrainingArguments

# Load RoBERTa with regression head
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=1)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2
)


# In[3]:


import evaluate
import numpy as np

mse_metric = evaluate.load("mse")
r2_metric = evaluate.load("r_squared")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    labels = labels.squeeze()
    mse = mse_metric.compute(predictions=predictions, references=labels)
    r2 = r2_metric.compute(predictions=predictions, references=labels)
    return {
        "mse": mse["mse"],
        "r2": r2["r_squared"]
    }


# In[5]:


from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    tokenizer=None,
    compute_metrics=compute_metrics,
)

trainer.train()
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)


# In[ ]:





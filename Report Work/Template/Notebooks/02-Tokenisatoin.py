#!/usr/bin/env python
# coding: utf-8

# #  Tokenisation for Transformer-Based AES
# 
# This notebook prepares the cleaned ASAP dataset for input into transformer-based models by applying the RoBERTa tokenizer. It includes:
# 
# - Loading the normalised, cleaned dataset
# - Initialising the RoBERTa tokenizer
# - Applying padding and truncation
# - Outputting a tokenised HuggingFace-compatible dataset
# 
# Tokenisation aligns with the preprocessing strategy described in Section 4.2 of the dissertation.
# 

# In[4]:


import pandas as pd
from transformers import RobertaTokenizerFast
from datasets import Dataset


# In[ ]:


# Load your cleaned and normalised dataset
df = pd.read_csv("../data/processed/asap_cleaned.csv")

# Check a sample
df[['essay_id', 'essay_set', 'essay', 'score_scaled']].head()


# In[ ]:


# Load the RoBERTa tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# Set max length for tokenisation
MAX_LENGTH = 512


# In[ ]:


# HuggingFace Datasets expects columns as dictionary entries
dataset = Dataset.from_pandas(df[['essay', 'score_scaled']])  # only keep needed fields

# Tokenisation function
def tokenize_function(example):
    return tokenizer(
        example['essay'],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

# Apply tokenizer across dataset
tokenised_dataset = dataset.map(tokenize_function, batched=True)


# In[ ]:


from datasets import DatasetDict

# Split into 80% train, 10% val, 10% test
split_dataset = tokenised_dataset.train_test_split(test_size=0.2, seed=42)
val_test_split = split_dataset['test'].train_test_split(test_size=0.5, seed=42)

# Combine into DatasetDict
dataset_dict = DatasetDict({
    'train': split_dataset['train'],
    'validation': val_test_split['train'],
    'test': val_test_split['test']
})

# Check sizes
print(dataset_dict)


# In[ ]:


dataset_dict.save_to_disk("../data/processed/tokenised_asap_split")


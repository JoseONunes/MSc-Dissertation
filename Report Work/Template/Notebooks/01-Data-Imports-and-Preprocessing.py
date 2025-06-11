#!/usr/bin/env python
# coding: utf-8

# #  Data Preprocessing: ASAP Dataset
# 
# This notebook holds all of the preprocessing steps required for the projects relevant datasets. It includes:
# - Loading and inspecting the original ASAP data
# - Cleaning and filtering essay entries
# - Handling scoring ranges and label normalisation
# - Preparing tokenised inputs for model training
# 
# All steps in this notebook follow the methodology described in Chapter 4 of the dissertation and are designed to ensure reproducibility and compatibility with the HuggingFace transformer framework.
# 

# ##  Loading the ASAP Dataset
# 
# The following section loads the original Automated Student Assessment Prize (ASAP) dataset from the "raw" data directory. The dataset includes approximately 13,000 student essays across eight distinct writing prompts, each with unique scoring rubrics.
# 
# Key columns of interest include:
# - `essay_id`: unique identifier for each essay
# - `essay_set`: the prompt ID (18)
# - `essay`: the full essay text
# - `domain1_score`: the primary human-assigned score    ######should this also include the rest of the scoring, why am I just blindly trusting the domain1_score
# 
# Additional fields (e.g., `rater_1_domain1`, `rater_2_domain1`) may be used for advanced analysis but are not essential to the initial modelling phase.
# 

# In[1]:


import pandas as pd

# Load in the ASAP dataset - Rel3 is used as it is the most up to date and contains the least errors
df = pd.read_csv("../data/raw/asap-aes/training_set_rel3.tsv", sep='\t', encoding='latin1')

# Basic info - for checks
print(f" Loaded dataset with shape: {df.shape}")
print("\n Column names:")
print(df.columns.tolist())

# Preview a few entries
df[['essay_id', 'essay_set', 'domain1_score', 'essay']].head()


# In[7]:


# Drop any rows with missing essays or scores
df = df.dropna(subset=['essay', 'domain1_score'])

# Strip whitespace and remove essays with too few words
df['essay'] = df['essay'].str.strip()
df['word_count'] = df['essay'].apply(lambda x: len(x.split()))
df = df[df['word_count'] >= 50]  # remove very short essays

# Reset index after filtering
df = df.reset_index(drop=True)

# Show updated shape and word count stats
print(f" Cleaned dataset shape: {df.shape}")
print(df['word_count'].describe())


# In[8]:


# Get per-prompt score ranges
prompt_stats = df.groupby('essay_set')['domain1_score'].agg(['min', 'max']).rename(columns={'min': 'min_score', 'max': 'max_score'})
print(" Score ranges by prompt:")
print(prompt_stats)

# Merge these stats back into the main DataFrame
df = df.merge(prompt_stats, left_on='essay_set', right_index=True)

# Apply MinMax normalisation
df['score_scaled'] = (df['domain1_score'] - df['min_score']) / (df['max_score'] - df['min_score'])

# Preview scaled scores
df[['essay_id', 'essay_set', 'domain1_score', 'score_scaled']].head()


# In[9]:


#Saving and Visualize the cleaned dataset
df.to_csv("../data/Processed/asap_cleaned.csv", index=False)
import matplotlib.pyplot as plt
df['score_scaled'].hist(bins=30)
plt.title("Distribution of Normalised Scores")
plt.xlabel("score_scaled")
plt.ylabel("Count")
plt.show()


# # ^^ that looks like a lot of full mark scores? am I happy with that? ^^

# 

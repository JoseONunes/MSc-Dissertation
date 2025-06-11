#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from datasets import load_dataset
from tqdm.notebook import tqdm

# Optional: for QWK
get_ipython().system('pip install --quiet scikit-learn scipy')
from sklearn.metrics import cohen_kappa_score

# Load your cleaned CSV
df = pd.read_csv("../Data/Processed/asap_cleaned.csv")
df.head()


# In[3]:


# Use 'essay' column as input text
texts = df['essay'].astype(str).values

# Use normalised score as target (assumes already scaled to [0, 1])
labels = df['score_scaled'].values


# In[4]:


# First split: Train + Temp (for val/test)
X_train, X_temp, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.2, random_state=42)

# Second split: Temp  Validation + Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size: {len(X_test)}")


# In[5]:


vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)


# In[6]:


ridge = Ridge(alpha=1.0)  # You can tune alpha later if desired
ridge.fit(X_train_vec, y_train)

# Predict on validation and test
val_preds_ridge = ridge.predict(X_val_vec)
test_preds_ridge = ridge.predict(X_test_vec)


# In[7]:


def qwk(y_true, y_pred, min_rating=0, max_rating=1):
    """
    Quadratic Weighted Kappa. Assumes inputs scaled to [0, 1].
    For scoring purposes, predictions are mapped back to 0-12 scale (ASAP-style).
    """
    y_pred_rounded = np.round(y_pred * 12).astype(int)
    y_true_rounded = np.round(y_true * 12).astype(int)
    return cohen_kappa_score(y_true_rounded, y_pred_rounded, weights="quadratic")

mse_ridge = mean_squared_error(y_test, test_preds_ridge)
qwk_ridge = qwk(y_test, test_preds_ridge)

print(f"Ridge Regression - MSE: {mse_ridge:.4f}, QWK: {qwk_ridge:.4f}")


# In[8]:


svr = SVR(kernel="rbf", C=1.0, epsilon=0.1)
svr.fit(X_train_vec, y_train)

# Predict
val_preds_svr = svr.predict(X_val_vec)
test_preds_svr = svr.predict(X_test_vec)


# In[9]:


mse_svr = mean_squared_error(y_test, test_preds_svr)
qwk_svr = qwk(y_test, test_preds_svr)

print(f"Support Vector Regression - MSE: {mse_svr:.4f}, QWK: {qwk_svr:.4f}")


# In[10]:


rf = RandomForestRegressor(
    n_estimators=100,        # Number of trees
    max_depth=None,          # You can limit this to avoid overfitting
    random_state=42,
    n_jobs=-1                # Use all available cores
)
rf.fit(X_train_vec, y_train)

# Predict
val_preds_rf = rf.predict(X_val_vec)
test_preds_rf = rf.predict(X_test_vec)


# In[11]:


mse_rf = mean_squared_error(y_test, test_preds_rf)
qwk_rf = qwk(y_test, test_preds_rf)

print(f"Random Forest Regressor - MSE: {mse_rf:.4f}, QWK: {qwk_rf:.4f}")


# In[12]:


results_df = pd.DataFrame({
    "Model": ["Ridge Regression", "Support Vector Regression", "Random Forest Regressor"],
    "MSE": [mse_ridge, mse_svr, mse_rf],
    "QWK": [qwk_ridge, qwk_svr, qwk_rf]
})

# Round for cleaner display
results_df = results_df.round(4)
display(results_df)


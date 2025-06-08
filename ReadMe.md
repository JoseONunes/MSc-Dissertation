# Transformer-Based Automated Essay Scoring (AES)

This project aims to develop a transformer-based Automated Essay Scoring (AES) system with a focus on **explainability**, **fairness**, and **generalisation** across writing prompts. The model will be benchmarked on the ASAP dataset and evaluated using standard metrics such as QWK, MSE, and SHAP visualisations.

## 📌 Project Objectives

- Fine-tune a RoBERTa-based model for holistic essay scoring
- Compare performance against traditional ML baselines (e.g., Ridge, SVM)
- Evaluate generalisability across prompts and writing styles
- Apply SHAP and fairness metrics to assess model transparency and bias
- Explore future extensions including rubric-based feedback and handwriting support

## 🛠️ Technologies

- Python 3.11
- Jupyter Notebooks (via VS Code)
- Transformers (HuggingFace)
- scikit-learn, pandas, numpy
- SHAP, Fairlearn, matplotlib, seaborn
- LaTeX (VS Code for dissertation writing)

## 📂 Folder Structure

```text
Report Work/
├── Final Report/        # All of the latex files related to the final report submission
├── Project Plan:Brief/        # All of the latex files related to the final report submission
project/
├── data/
│   ├── raw/             # Original ASAP dataset
│   └── processed/       # Cleaned and tokenised data
├── notebooks/           # Jupyter notebooks (e.g. data preprocessing)
├── src/                 # Modular source code for models, utils, evaluation
├── models/              # Saved model weights/checkpoints
├── outputs/             # Predictions, SHAP plots, etc.
├── reports/
│   └── figures/         # Visuals for dissertation
├── requirements.txt     # Python dependencies
.gitignore               # 
README.md                # You are here
```

## ✅ Current Status

- [x] Project structure initialised
- [x] Virtual environment set up (`venv`)
- [x] VS Code Jupyter integration working
- [x] Dataset cleaning and label normalisation planned
- [ ] Code for loading and preprocessing ASAP dataset — *in progress*
- [ ] Tokenisation with RoBERTa tokenizer
- [ ] Baseline and transformer model training

---

## 📖 Dissertation Structure

The project dissertation is being written in LaTeX and follows a report structure with sections on literature review, methodology, experimental results, and discussion. Integration between code output and the report is managed via shared folders for figures and logs.

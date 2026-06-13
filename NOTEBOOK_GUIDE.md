# Notebook Guide

This project includes a notebook workflow so the model is not just a saved `.pkl` file.

## Notebooks

```text
notebooks/01_data_overview.ipynb
notebooks/02_eda.ipynb
notebooks/03_feature_engineering_review.ipynb
notebooks/04_train_v5_model.ipynb
notebooks/05_model_evaluation_and_predictions.ipynb
```

## Run order

```text
01_data_overview
02_eda
03_feature_engineering_review
04_train_v5_model
05_model_evaluation_and_predictions
```

## Training

The training notebook is:

```text
notebooks/04_train_v5_model.ipynb
```

It trains the specialist models, learns blend weights, calibrates probabilities, evaluates the test period, and saves the final model.

## Start Jupyter

```bash
pip install -r requirements.txt
jupyter notebook
```

## Start Streamlit

```bash
streamlit run deployment/app_streamlit.py
```

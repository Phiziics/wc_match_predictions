# GitHub Push Instructions

## 1. Open terminal in this folder

```bash
cd wc_match_predictions
```

## 2. Create the local git repo

```bash
git init
git add .
git commit -m "Initial commit: Match Edge AI V5 World Cup predictor"
```

## 3. Create a new GitHub repo

Go to GitHub and create a new repository named:

```text
wc_match_predictions
```

Do not add a README, .gitignore, or license on GitHub because this project already has files.

## 4. Connect local project to GitHub

Replace `YOUR_USERNAME` with your GitHub username:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/wc_match_predictions.git
git push -u origin main
```

## 5. If GitHub blocks the model file

If the `.pkl` model file is too large, use Git LFS:

```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git add models/match_edge_ai_v5_calibrated_specialist_ensemble.pkl
git commit -m "Track model files with Git LFS"
git push
```

## 6. Run locally

```bash
pip install -r requirements.txt
streamlit run deployment/app_streamlit.py
```

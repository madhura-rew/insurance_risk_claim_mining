# Insurance Claim Risk Mining — Streamlit App

A simple Streamlit website for the **Insurance Claim Risk Mining** student project. It uses the same `fraud_oracle.csv` dataset and follows the notebook's basic workflow: exploratory analysis, Logistic Regression and Random Forest, model evaluation, feature importance, and a simple risk-score demo.

## Pages

- **Overview** — total claims, genuine/fraud-labelled counts, fraud rate, and model comparison.
- **Explore claims** — explore fraud-labelled rates by accident area, past claims, policy type, vehicle category, and police-report status.
- **Model performance** — model metrics, ROC curves, Random Forest confusion matrix, and feature importance.
- **Score a claim** — enter selected claim details and receive a Random Forest probability score and the notebook's illustrative risk band.

## Files

```text
insurance_claim_risk_mining_streamlit/
├── app.py
├── fraud_oracle.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Run locally

Python 3.10 or 3.11 is recommended.

1. Install Python.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (optional but recommended):

   **Windows**
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install packages:
   ```bash
   pip install -r requirements.txt
   ```

5. Start the website:
   ```bash
   streamlit run app.py
   ```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Deploy with Streamlit Community Cloud

1. Create a new public GitHub repository, for example `insurance-claim-risk-mining`.
2. Upload these project files to the repository root (`app.py`, `requirements.txt`, `fraud_oracle.csv`, `README.md`, `.gitignore`).
3. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
4. Select **Create app**, choose your repository and branch, and set the main file path to `app.py`.
5. Click **Deploy**. The first launch trains both models, so allow a little time for the initial load.

## Important interpretation notes

- The dataset is imbalanced: roughly 5.99% of records are fraud-labelled.
- Accuracy alone is not enough to evaluate this task. The notebook's Random Forest has high accuracy but very low fraud recall at the default threshold.
- The claim-score form fills unedited model inputs with typical values from the training data. Treat it as a demonstration of the pipeline, not a production-grade individual claim assessment.
- The risk bands (Low < 30, Medium 30–<70, High ≥70) are inherited from the notebook and have not been validated.
- A score is **not proof of fraud**. Do not use this prototype to automatically reject claims or accuse claimants.
- Check that you have permission to redistribute the dataset before making the GitHub repository public. If not, keep the CSV out of GitHub and provide it privately or add it through Streamlit secrets/storage appropriate to your setup.

## Model workflow

- `FraudFound_P` is the target (`0` = genuine-labelled, `1` = fraud-labelled).
- `PolicyNumber` is dropped as an identifier.
- Categorical features: most-frequent imputation + one-hot encoding.
- Numerical features: median imputation + standard scaling.
- Stratified 80:20 train–test split (`random_state=42`).
- Logistic Regression and Random Forest are compared using accuracy, precision, recall, F1-score and ROC-AUC.

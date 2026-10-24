import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Insurance Claim Risk Mining",
    page_icon="🛡️",
    layout="wide",
)

DATA_PATH = Path(__file__).parent / "fraud_oracle.csv"
TARGET = "FraudFound_P"
ID_COLUMN = "PolicyNumber"

# ------------------------------------------------------------
# Styling — fixed metric cards and dark theme
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main application background */
    .stApp {
        background-color: #0e1117;
        color: #f8fafc;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #262730;
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    /* FIX: Metric cards */
    div[data-testid="stMetric"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        padding: 18px 20px !important;
        border-radius: 12px !important;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label p,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] *,
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(120deg, #172554, #1d4ed8);
        padding: 25px 28px;
        border-radius: 16px;
        color: white;
        margin-bottom: 20px;
    }

    .hero h1 {
        color: white !important;
        margin-bottom: 5px;
    }

    .hero p {
        color: #dbeafe !important;
        margin: 0;
    }

    /* General text */
    h1, h2, h3, h4, p, label, li {
        color: #f1f5f9;
    }

    /* Dataframes remain legible */
    div[data-testid="stDataFrame"] {
        border: 1px solid #334155;
        border-radius: 8px;
    }

    /* Inputs */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #1e293b;
    }

    /* Links */
    a {
        color: #93c5fd !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🛡️ Insurance Claim Risk Mining</h1>
      <p>Explore historical claim patterns, compare machine-learning models, and estimate a claim's model-based risk score.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
if not DATA_PATH.exists():
    st.error("Dataset not found. Put fraud_oracle.csv in the same folder as app.py.")
    st.stop()


@st.cache_data
def load_data(path, modified_time):
    return pd.read_csv(path)


@st.cache_resource
def train_models(data):
    data = data.copy()
    X = data.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
    y = data[TARGET].astype(int)

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        exclude=["object", "category"]
    ).columns.tolist()

    def make_preprocessor():
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("categorical", categorical_transformer, categorical_features),
                ("numerical", numerical_transformer, numerical_features),
            ]
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    logistic = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )

    forest = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                    max_depth=None,
                ),
            ),
        ]
    )

    logistic.fit(X_train, y_train)
    forest.fit(X_train, y_train)

    models = {
        "Logistic Regression": logistic,
        "Random Forest": forest,
    }

    metrics = []
    probabilities = {}
    predictions = {}

    for name, model in models.items():
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        predictions[name] = pred
        probabilities[name] = prob

        metrics.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1 Score": f1_score(y_test, pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, prob),
            }
        )

    fitted_preprocessor = forest.named_steps["preprocessor"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = forest.named_steps["classifier"].feature_importances_

    importance_df = (
        pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "models": models,
        "metrics": pd.DataFrame(metrics).set_index("Model"),
        "probabilities": probabilities,
        "predictions": predictions,
        "feature_importance": importance_df,
    }


data = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)

if TARGET not in data.columns:
    st.error(f"Expected target column '{TARGET}' was not found in the dataset.")
    st.stop()

with st.spinner("Preparing the data and training the models for this session..."):
    results = train_models(data)

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("Dashboard controls")

page = st.sidebar.radio(
    "Choose a section",
    ["Overview", "Explore claims", "Model performance", "Score a claim"],
)

st.sidebar.caption(
    "Models are trained on the included dataset when the app starts. "
    "First load may take a little while."
)
st.sidebar.divider()
st.sidebar.caption(
    "Educational prototype only. A risk score is not proof of fraud."
)

# ------------------------------------------------------------
# Overview
# ------------------------------------------------------------
if page == "Overview":
    total = len(data)
    fraud = int(data[TARGET].sum())
    genuine = total - fraud
    fraud_rate = 100 * fraud / total

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total claims", f"{total:,}")
    c2.metric("Fraud-labelled", f"{fraud:,}")
    c3.metric("Genuine-labelled", f"{genuine:,}")
    c4.metric("Fraud rate", f"{fraud_rate:.2f}%")

    st.subheader("Claim distribution")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    counts = data[TARGET].value_counts().reindex([0, 1], fill_value=0)
    bars = ax.bar(
        ["Genuine claims", "Fraud-labelled claims"],
        counts.values,
        color=["#2563eb", "#dc2626"],
        width=0.55,
    )

    ax.set_ylabel("Number of claims", color="white")
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#64748b")
    ax.spines["left"].set_color("#64748b")
    ax.bar_label(bars, fmt="%.0f", padding=3, color="white")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.info(
        "The dataset is imbalanced: only about 6% of records are fraud-labelled. "
        "That is why accuracy should not be interpreted on its own."
    )

    st.subheader("Quick model comparison")
    st.dataframe(
        results["metrics"].style.format("{:.3f}"),
        use_container_width=True,
    )

    st.caption(
        "Evaluation uses a stratified 80:20 train–test split with "
        "random_state=42, matching the notebook workflow."
    )

# ------------------------------------------------------------
# Explore claims
# ------------------------------------------------------------
elif page == "Explore claims":
    st.subheader("Explore fraud patterns")

    chart_choice = st.selectbox(
        "Choose a claim attribute",
        [
            "AccidentArea",
            "PastNumberOfClaims",
            "PolicyType",
            "VehicleCategory",
            "PoliceReportFiled",
        ],
    )

    grouped = (
        data.groupby(chart_choice, dropna=False)[TARGET]
        .agg(claim_count="size", fraud_rate="mean")
        .reset_index()
    )

    grouped["fraud_rate"] *= 100
    grouped = grouped.sort_values("fraud_rate", ascending=False)

    left, right = st.columns([1.15, 1])

    with left:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        bars = ax.bar(
            grouped[chart_choice].astype(str),
            grouped["fraud_rate"],
            color="#2563eb",
        )

        ax.set_title(
            f"Fraud-labelled rate by {chart_choice}", color="white"
        )
        ax.set_ylabel("Fraud-labelled records (%)", color="white")
        ax.set_xlabel(chart_choice, color="white")
        ax.tick_params(colors="white", axis="both", rotation=35)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["bottom"].set_color("#64748b")
        ax.spines["left"].set_color("#64748b")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown("**Group summary**")
        st.dataframe(
            grouped.rename(
                columns={
                    chart_choice: "Category",
                    "claim_count": "Claim count",
                    "fraud_rate": "Fraud rate (%)",
                }
            ).style.format({"Fraud rate (%)": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "These are descriptive relationships in this dataset, not proof that "
        "an attribute causes fraud. Small groups may have unstable rates."
    )

    with st.expander("Preview the dataset"):
        st.dataframe(data.head(100), use_container_width=True)

# ------------------------------------------------------------
# Model performance
# ------------------------------------------------------------
elif page == "Model performance":
    st.subheader("Model comparison")
    metrics = results["metrics"]

    st.dataframe(
        metrics.style.format("{:.3f}"),
        use_container_width=True,
    )

    metric = st.selectbox(
        "Metric to compare",
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    vals = metrics[metric]
    bars = ax.bar(
        vals.index,
        vals.values,
        color=["#0f766e", "#2563eb"],
        width=0.55,
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric, color="white")
    ax.set_title(
        f"{metric}: Logistic Regression vs Random Forest",
        color="white",
    )
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#64748b")
    ax.spines["left"].set_color("#64748b")
    ax.bar_label(bars, fmt="%.3f", padding=3, color="white")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.subheader("ROC curves")

    y_test = results["y_test"]
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    for name, prob in results["probabilities"].items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc_value = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_value:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random baseline")
    ax.set_xlabel("False positive rate", color="white")
    ax.set_ylabel("True positive rate", color="white")
    ax.set_title("ROC curve", color="white")
    ax.tick_params(colors="white")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.subheader("Random Forest confusion matrix")

    cm = confusion_matrix(
        y_test,
        results["predictions"]["Random Forest"],
        labels=[0, 1],
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Predicted genuine", "Predicted fraud"])
    ax.set_yticks([0, 1], labels=["Actual genuine", "Actual fraud"])
    ax.set_title("Random Forest confusion matrix", color="white")
    ax.tick_params(colors="white")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="black", fontsize=12,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Top 15 Random Forest features")

    top = results["feature_importance"].head(15).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.barh(top["Feature"], top["Importance"], color="#0f766e")
    ax.set_xlabel("Feature importance", color="white")
    ax.set_title("Top features in the fitted Random Forest", color="white")
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.warning(
        "The Random Forest may show high accuracy but very low fraud recall "
        "at its default decision threshold. Use recall, precision, and the "
        "confusion matrix—not accuracy alone—when interpreting results."
    )

# ------------------------------------------------------------
# Score a claim
# ------------------------------------------------------------
elif page == "Score a claim":
    st.subheader("Estimate a claim's model-based score")

    st.write(
        "Enter a few visible claim details. The remaining model inputs are "
        "filled with typical values from the training data. This is a "
        "demonstration, not a real insurance decision."
    )

    X = results["X"]
    X_train = results["X_train"]

    categorical = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical = X.select_dtypes(
        exclude=["object", "category"]
    ).columns.tolist()

    base = {}

    for col in categorical:
        mode_values = X_train[col].mode(dropna=True)
        base[col] = mode_values.iloc[0] if not mode_values.empty else ""

    for col in numerical:
        base[col] = float(X_train[col].median())

    editable_categorical = [
        c for c in [
            "AccidentArea", "PastNumberOfClaims", "PolicyType",
            "VehicleCategory", "PoliceReportFiled", "Fault", "Sex",
            "MaritalStatus", "VehiclePrice", "AgeOfVehicle",
            "AgeOfPolicyHolder",
        ] if c in categorical
    ]

    editable_numeric = [
        c for c in ["Age", "DriverRating", "Deductible", "WeekOfMonth"]
        if c in numerical
    ]

    with st.form("claim_form"):
        st.markdown("**Claim details**")
        columns = st.columns(2)
        user_values = {}

        for idx, col in enumerate(editable_categorical):
            with columns[idx % 2]:
                choices = sorted(
                    X_train[col].dropna().astype(str).unique().tolist()
                )

                if not choices:
                    continue

                default = str(base[col])
                index = choices.index(default) if default in choices else 0

                user_values[col] = st.selectbox(
                    col.replace("_", " "),
                    choices,
                    index=index,
                )

        for idx, col in enumerate(editable_numeric):
            with columns[idx % 2]:
                lo = float(X_train[col].min())
                hi = float(X_train[col].max())
                default = float(base[col])

                if lo == hi:
                    user_values[col] = st.number_input(
                        col.replace("_", " "),
                        value=default,
                    )
                else:
                    user_values[col] = st.number_input(
                        col.replace("_", " "),
                        min_value=lo,
                        max_value=hi,
                        value=min(max(default, lo), hi),
                        step=1.0,
                    )

        submitted = st.form_submit_button(
            "Estimate risk score",
            type="primary",
        )

    if submitted:
        row = base.copy()
        row.update(user_values)
        input_df = pd.DataFrame([row], columns=X.columns)

        probability = float(
            results["models"]["Random Forest"]
            .predict_proba(input_df)[0, 1]
        )

        score = probability * 100
        risk_level = (
            "Low Risk" if score < 30
            else ("Medium Risk" if score < 70 else "High Risk")
        )

        left, right = st.columns(2)
        left.metric("Model fraud probability score", f"{score:.2f}/100")
        right.metric("Notebook-style risk band", risk_level)

        st.progress(min(max(probability, 0.0), 1.0))

        st.caption(
            "Risk bands are the project's original illustrative cut-offs: "
            "Low < 30, Medium 30–<70, High ≥70. They have not been validated "
            "for real-world use."
        )

        st.info(
            "This result is generated by a model trained on historical labels. "
            "It does not establish that a claim is fraudulent. Any real use "
            "would require validation, threshold tuning, and human review."
        )

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.divider()
st.caption(
    "Insurance Claim Risk Mining • Student project prototype • "
    "Built with Python, pandas, scikit-learn, Matplotlib and Streamlit"
)

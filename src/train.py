import os
import mlflow
import mlflow.sklearn
import warnings
import pandas as pd
import matplotlib.pyplot as plt

# Suppress Pydantic namespace warnings and sklearn/mlflow future warnings globally
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Allow local file store tracking
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from data_preprocessing import download_data, preprocess_data

def train_models():
    print("Downloading and preparing data...")
    
    # Resolve absolute path for raw data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    raw_path = os.path.join(project_root, "data", "raw", "heart.csv")
    
    # Download raw data if it doesn't exist
    if not os.path.exists(raw_path):
        download_data(raw_path)
        
    # Preprocess data (this automatically generates heart_cleaned.csv and returns train/test splits)
    X_train, X_test, y_train, y_test = preprocess_data(raw_path)
    
    # Define robust preprocessing pipeline
    preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # Point tracking to local SQLite and set active MLflow experiment
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(project_root, 'mlflow.db')}")
    mlflow.set_experiment("Heart_Disease_Classification")
    
    model_metrics_history = {}
    
    for name, model in models.items():
        print(f"Training model: {name}...")
        with mlflow.start_run(run_name=name):
            clf = Pipeline(steps=[
                ('preprocessor', preprocessor), 
                ('classifier', model)
            ])
            
            # Fit model
            clf.fit(X_train, y_train)
            
            # Predictions and Probabilities
            preds = clf.predict(X_test)
            probs = clf.predict_proba(X_test)[:, 1]
            
            # Compute Evaluation Metrics
            acc = float(accuracy_score(y_test, preds))
            prec = float(precision_score(y_test, preds, zero_division=0))
            rec = float(recall_score(y_test, preds, zero_division=0))
            roc = float(roc_auc_score(y_test, probs))
            
            # Save to history dictionary for final comparison chart
            model_metrics_history[name] = [acc, prec, rec, roc]
            
            # Log Parameters & Metrics to MLflow
            mlflow.log_param("model_type", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("roc_auc", roc)
            
            # Generate and Log Individual Performance Bar Chart as an MLflow Artifact
            artifacts_dir = os.path.join(project_root, "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            plot_path = os.path.join(artifacts_dir, f"{name}_metrics.png")
            
            metric_names = ['Accuracy', 'Precision', 'Recall', 'ROC-AUC']
            plt.figure(figsize=(6, 4))
            plt.bar(metric_names, model_metrics_history[name], color=['#4CAF50', '#2196F3', '#FF9800', '#E91E63'])
            plt.ylim(0, 1.05)
            plt.title(f'{name} Performance Metrics')
            plt.ylabel('Score')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            plt.close()
            
            mlflow.log_artifact(plot_path)
            
            # Log Model Artifact with an input example to infer signature cleanly
            mlflow.sklearn.log_model(
                clf, 
                artifact_path="model",
                input_example=X_test.iloc[:5] if hasattr(X_test, "iloc") else X_test[:5]
            )
            
            print(f"--> Successfully completed {name}. Accuracy: {acc:.4f}, ROC-AUC: {roc:.4f}\n")

    # === GENERATE COMPARATIVE GROUPED BAR CHART REPORT ===
    print("Generating model comparison report graph...")
    plt.figure(figsize=(8, 5))
    
    x = range(len(metric_names))
    width = 0.35  # Width of the bars
    
    lr_scores = model_metrics_history.get("LogisticRegression", [0, 0, 0, 0])
    rf_scores = model_metrics_history.get("RandomForest", [0, 0, 0, 0])
    
    plt.bar([i - width/2 for i in x], lr_scores, width, label='Logistic Regression', color='#2196F3')
    plt.bar([i + width/2 for i in x], rf_scores, width, label='Random Forest', color='#FF9800')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison: Logistic Regression vs Random Forest')
    plt.xticks(x, metric_names)
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    comparison_plot_path = os.path.join(artifacts_dir, "model_comparison_report.png")
    plt.savefig(comparison_plot_path, dpi=300)
    plt.close()
    
    # Log the comparative report as a separate MLflow run
    with mlflow.start_run(run_name="Model_Comparison_Report"):
        mlflow.log_artifact(comparison_plot_path)
        print("Comparison report saved and logged successfully to MLflow artifacts.")

if __name__ == "__main__":
    train_models()
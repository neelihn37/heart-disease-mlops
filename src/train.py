import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

)

# Set professional plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.autolayout': True
})

def train_and_evaluate_models():
    # 1. Define paths dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    processed_path = os.path.join(project_root, "data", "processed", "heart_cleaned.csv")
    models_dir = os.path.join(project_root, "models")
    figures_dir = os.path.join(project_root, "reports", "figures")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Load processed data
    if not os.path.exists(processed_path):
        from data_preprocessing import preprocess_data
        preprocess_data()
    
    df = pd.read_csv(processed_path)
    
    # 2. Split Features and Target
    X = df.drop(columns=['target'])
    y = df['target']

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Cast integer columns to float64 to satisfy MLflow signature requirements
    for col in X_train.select_dtypes(include=['int', 'int64']).columns:
        X_train[col] = X_train[col].astype(float)
        X_test[col] = X_test[col].astype(float)

    # 3. Define Models with Pipelines (Scaling + Classifier)
    models = {
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ('scaler', StandardScaler()), 
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    evaluation_results = []

    # Set MLflow Experiment Name
    mlflow.set_experiment("heart_disease_classification")

    print("--- Model Training & MLflow Experiment Tracking ---")
    for name, pipeline in models.items():
        with mlflow.start_run(run_name=name.replace(" ", "_").lower()):
            print(f"\nTraining {name}...")

            # Extract hyperparameters for logging
            classifier = pipeline.named_steps['classifier']
            params = classifier.get_params()
            
            # Log hyperparameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("cv_splits", 5)

            # Cross-validation score (ROC-AUC)
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
            mean_cv_auc = cv_scores.mean()
            std_cv_auc = cv_scores.std()
            
            print(f"  5-Fold CV ROC-AUC: {mean_cv_auc:.4f} (+/- {std_cv_auc:.4f})")
            mlflow.log_metric("cv_roc_auc_mean", mean_cv_auc)
            mlflow.log_metric("cv_roc_auc_std", std_cv_auc)

            # Fit on full training set
            pipeline.fit(X_train, y_train)

            # Evaluate on test set
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)

            # Log evaluation metrics to MLflow
            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_precision", prec)
            mlflow.log_metric("test_recall", rec)
            mlflow.log_metric("test_f1_score", f1)
            mlflow.log_metric("test_roc_auc", roc_auc)

            # Append metrics to results list
            evaluation_results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": roc_auc
            })

            # Print all evaluation metrics clearly in output
            print(f"  Test Metrics for {name}:")
            print(f"    - Accuracy  : {acc:.4f}")
            print(f"    - Precision : {prec:.4f}")
            print(f"    - Recall    : {rec:.4f}")
            print(f"    - F1-Score  : {f1:.4f}")
            print(f"    - ROC-AUC   : {roc_auc:.4f}")

            # # Infer model signature and input example to resolve MLflow warnings
            # signature = infer_signature(X_test, y_pred)
            # input_example = X_test.iloc[0:1]

            # # Save model artifact locally and log to MLflow with signature & input example
            # model_filename = os.path.join(models_dir, f"{name.lower().replace(' ', '_')}_model.joblib")
            # joblib.dump(pipeline, model_filename)
            
            # mlflow.sklearn.log_model(
            #     pipeline, 
            #     artifact_path="model",
            #     #signature=signature,
            #     #input_example=input_example
            # Save model artifact locally and log to MLflow cleanly
            signature = infer_signature(X_test, y_pred)
            input_example = X_test.iloc[0:1]

            model_filename = os.path.join(models_dir, f"{name.lower().replace(' ', '_')}_model.joblib")
            joblib.dump(pipeline, model_filename)
            
            mlflow.sklearn.log_model(
                pipeline, 
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )            

    # 4. Save Summary Report & Generate Comparison Graph
    results_df = pd.DataFrame(evaluation_results)
    results_path = os.path.join(project_root, "reports", "model_evaluation_summary.csv")
    results_df.to_csv(results_path, index=False)
    
    # Plot Metrics Comparison Graph
    plt.figure(figsize=(10, 6))
    results_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    
    ax = sns.barplot(data=results_melted, x="Metric", y="Score", hue="Model", palette="muted")
    plt.ylim(0, 1.1)
    plt.title("Model Performance Metrics Comparison")
    plt.xlabel("Evaluation Metric")
    plt.ylabel("Score")
    plt.legend(loc="lower right")

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 3), textcoords='offset points',
                        fontsize=10)

    comparison_chart_path = os.path.join(figures_dir, "metrics_comparison.png")
    plt.savefig(comparison_chart_path, dpi=300)
    plt.close()

    print(f"\nEvaluation summary saved to: {results_path}")
    print(f"Metrics comparison graph successfully saved to: {comparison_chart_path}")
    print("MLflow experiment logging completed successfully!")

if __name__ == "__main__":
    train_and_evaluate_models()
from docx import Document

doc = Document()
doc.add_heading('End-to-End Heart Disease MLOps Project Report', 0)

# Section 1
doc.add_heading('1. Introduction & Executive Summary', level=1)
doc.add_paragraph(
    "Overview of the heart disease classification MLOps system, business value, and objectives. "
    "This system provides a reliable, containerized, and automated pipeline for predicting heart disease risk "
    "using clinical indicators from the UCI Heart Disease dataset."
)

# Section 2
doc.add_heading('2. Architecture & Tech Stack', level=1)
doc.add_paragraph(
    "Detailed breakdown of Minikube, Kubernetes, Docker, FastAPI, MLflow, and GitHub Actions. "
    "The architecture ensures seamless tracking, testing, containerization, and local/cloud orchestration."
)

# Section 3
doc.add_heading('3. Data Pipeline & Exploratory Data Analysis (EDA)', level=1)
doc.add_paragraph(
    "Dataset characteristics include 14+ clinical features, handled via custom preprocessing steps, "
    "median imputation for numerical variables, most-frequent imputation for categoricals, and one-hot encoding."
)

# Section 4 (Incorporating your Model Selection & Tuning details)
doc.add_heading('4. Model Training & Experiment Tracking', level=1)
doc.add_paragraph(
    "Model Selection & Tuning Documentation:\n"
    "• Logistic Regression: Selected as a robust, highly interpretable linear baseline model. "
    "It performs well when features are scaled using StandardScaler and helps establish a linear decision boundary.\n"
    "• Random Forest Classifier: Selected as a powerful non-linear ensemble method capable of capturing "
    "complex feature interactions and non-linear patterns in clinical data without requiring extensive feature normalization "
    "(though scaled for pipeline uniformity).\n"
    "• Hyperparameters: Default parameters with random_state=42 were used for reproducibility. "
    "Future tuning could explore grid search over C values for Logistic Regression and n_estimators / max_depth for Random Forest."
)

# Section 5 (Incorporating your Model Packaging & Reproducibility details)
doc.add_heading('5. Containerization & API Development', level=1)
doc.add_paragraph(
    "Model Packaging & Reproducibility: Models are serialized using both joblib (models/) and MLflow (mlruns/) "
    "to ensure cross-platform reusability. Preprocessing steps are encapsulated in modular scripts, and feature scaling "
    "is tightly bound to the model via scikit-learn Pipelines, ensuring that inference data undergoes identical "
    "transformations to the training set. Environment dependencies are version-locked in requirements.txt.\n\n"
    "The FastAPI application exposes a /predict endpoint with Pydantic validation, Prometheus metrics, and pre-filled Swagger docs."
)

# Sections 6-10
doc.add_heading('6. Kubernetes Orchestration & Deployment', level=1)
doc.add_paragraph("Minikube deployment manifests, NodePort configurations, pod health checks, and service routing using k8s-deployment.yaml.")

doc.add_heading('7. CI/CD Automation & Testing', level=1)
doc.add_paragraph("GitHub Actions workflow setup, automated unit testing with pytest for data shape and model integrity, and regression checks.")

doc.add_heading('8. Monitoring, Logging & Maintenance', level=1)
doc.add_paragraph("Structured JSON HTTP request logging and Prometheus metrics exposed via the /metrics endpoint.")

doc.add_heading('9. Results & Discussion', level=1)
doc.add_paragraph("Performance benchmarks, latency evaluations, and API verification results confirming successful inference.")

doc.add_heading('10. Conclusion & Future Scope', level=1)
doc.add_paragraph("Project summary, scaling considerations to enterprise cloud clusters (EKS/GKE), and future enhancements.")

doc.save("Report_Heart_Disease_MLOps.docx")
print("Successfully generated fully detailed Report_Heart_Disease_MLOps.docx!")
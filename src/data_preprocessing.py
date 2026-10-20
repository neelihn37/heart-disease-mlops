import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def download_data(save_path=None):
    if save_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
        save_path = os.path.join(project_root, "data", "raw", "heart.csv")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    print(f"Downloading raw dataset from UCI repository...")
    df = pd.read_csv(url, names=columns, na_values="?")
    df.to_csv(save_path, index=False)
    print(f"Raw dataset successfully saved to: {save_path}")
    return df

def preprocess_data(df_or_path=None):
    if df_or_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
        df_or_path = os.path.join(project_root, "data", "raw", "heart.csv")

    if isinstance(df_or_path, str):
        if not os.path.exists(df_or_path):
            df_or_path = download_data(df_or_path)
        else:
            df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()
    
    # 1. Impute missing values
    for col in ['ca', 'thal']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    # 2. Binarize target: 0 = No disease, 1-4 = Disease present
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # 3. Encode categorical features (One-Hot Encoding for nominal variables)
    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    # 4. Save cleaned and encoded dataset to processed folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    processed_path = os.path.join(processed_dir, "heart_cleaned.csv")
    df.to_csv(processed_path, index=False)
    print(f"Cleaned and encoded dataset successfully saved to: {processed_path}")
    
    # 5. Split into train/test sets
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = download_data()
    preprocess_data(df)
    print("Data pipeline executed successfully!")
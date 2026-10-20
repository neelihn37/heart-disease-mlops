import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def perform_eda():
    # Define paths dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    raw_data_path = os.path.join(project_root, "data", "raw", "heart.csv")
    figures_dir = os.path.join(project_root, "reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Load raw data for accurate distribution & correlation plots before one-hot encoding
    if not os.path.exists(raw_data_path):
        from data_preprocessing import download_data
        download_data(raw_data_path)
    
    df = pd.read_csv(raw_data_path)
    df['target_binary'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    print("Generating EDA visualizations...")

    # 1. Class Balance Plot
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(x='target_binary', hue='target_binary', data=df, palette='Set2', legend=False)
    plt.title('Heart Disease Class Distribution')
    plt.xlabel('Target (0 = No Disease, 1 = Disease)')
    plt.ylabel('Patient Count')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.savefig(os.path.join(figures_dir, 'class_balance.png'), dpi=300)
    plt.close()

    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', mask=mask, 
                linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Heatmap')
    plt.savefig(os.path.join(figures_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()

    # 3. Feature Distributions (Histograms)
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df[numeric_cols].hist(bins=15, figsize=(14, 10), color='skyblue', edgecolor='black', grid=False)
    plt.suptitle('Numerical Feature Distributions', y=1.02)
    plt.savefig(os.path.join(figures_dir, 'feature_histograms.png'), dpi=300)
    plt.close()

    print(f"All EDA visualizations successfully saved to: {figures_dir}")

if __name__ == "__main__":
    perform_eda()
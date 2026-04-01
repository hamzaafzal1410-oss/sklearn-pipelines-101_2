
#!/usr/bin/env python3
"""
Train script for Titanic Survival Prediction using sklearn Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import seaborn as sns

def load_data():
    """Load the Titanic dataset"""
    df = sns.load_dataset('titanic')

    # Select relevant features
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    target = 'survived'

    X = df[features]
    y = df[target]

    return X, y

def create_pipeline():
    """Create sklearn pipeline with preprocessing and model"""

    # Define column types
    numerical_features = ['age', 'fare', 'sibsp', 'parch']
    categorical_features = ['sex', 'embarked']
    # pclass is ordinal, treat as numerical

    # Numerical pipeline
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # Full pipeline with model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

    return pipeline

def evaluate_model(pipeline, X, y):
    """Perform cross-validation and return metrics"""

    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    # Stratified K-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation
    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    # Calculate mean and std for each metric
    metrics = {}
    for metric in scoring.keys():
        scores = cv_results[f'test_{metric}']
        metrics[metric] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores.tolist()
        }

    return metrics

def print_metrics_table(metrics):
    """Print metrics in a formatted table"""

    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS (5-fold stratified)")
    print("="*60)
    print(f"{'Metric':<12} {'Mean':<8} {'Std':<8} {'Scores'}")
    print("-"*60)

    for metric, values in metrics.items():
        mean_str = f"{values['mean']:.3f}"
        std_str = f"{values['std']:.3f}"
        scores_str = ", ".join([f"{s:.3f}" for s in values['scores']])
        print(f"{metric:<12} {mean_str:<8} {std_str:<8} [{scores_str}]")

    print("="*60)

def save_model(pipeline, filepath):
    """Save trained model to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"\nModel saved to: {filepath}")

def main():
    """Main training function"""
    print("Starting Titanic Survival Prediction Training...")

    # Load data
    print("Loading data...")
    X, y = load_data()
    print(f"Dataset shape: {X.shape}")

    # Create pipeline
    print("Creating pipeline...")
    pipeline = create_pipeline()

    # Evaluate model
    print("Performing cross-validation...")
    metrics = evaluate_model(pipeline, X, y)

    # Print results
    print_metrics_table(metrics)

    # Train final model on full data
    print("Training final model on full dataset...")
    pipeline.fit(X, y)

    # Save model
    model_path = "artifacts/model.joblib"
    save_model(pipeline, model_path)

    print("\nTraining completed successfully!")
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()
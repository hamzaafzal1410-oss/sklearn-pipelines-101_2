#!/usr/bin/env python3
"""
Prediction script for Titanic Survival Model
"""

import pandas as pd
import joblib
import sys
import os

def load_model(model_path="artifacts/model.joblib"):
    """Load trained model"""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run 'python src/train.py' first to train the model.")
        sys.exit(1)

    model = joblib.load(model_path)
    return model

def make_prediction(model, passenger_data):
    """
    Make prediction for a single passenger

    Args:
        model: Trained pipeline
        passenger_data: dict with passenger features

    Returns:
        dict: prediction results
    """
    # Convert to DataFrame
    df = pd.DataFrame([passenger_data])

    # Make prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    result = {
        'prediction': int(prediction),
        'survival_probability': float(probability[1]),
        'death_probability': float(probability[0]),
        'survived': 'Yes' if prediction == 1 else 'No'
    }

    return result

def main():
    """Main prediction function"""
    if len(sys.argv) < 8:
        print("Usage: python src/predict.py <pclass> <sex> <age> <sibsp> <parch> <fare> <embarked>")
        print("Example: python src/predict.py 3 male 25 0 0 7.25 S")
        print("\nOr with JSON: python src/predict.py json '{\"pclass\": 3, \"sex\": \"male\", \"age\": 25, \"sibsp\": 0, \"parch\": 0, \"fare\": 7.25, \"embarked\": \"S\"}'")
        sys.exit(1)

    # Load model
    model = load_model()

    # Parse arguments
    if sys.argv[1] == 'json' and len(sys.argv) >= 3:
        # JSON mode
        import json
        try:
            passenger_data = json.loads(sys.argv[2])
        except json.JSONDecodeError:
            print("Error: Invalid JSON format for passenger data")
            sys.exit(1)
    else:
        # Individual arguments mode
        try:
            passenger_data = {
                'pclass': int(sys.argv[1]),
                'sex': sys.argv[2],
                'age': float(sys.argv[3]),
                'sibsp': int(sys.argv[4]),
                'parch': int(sys.argv[5]),
                'fare': float(sys.argv[6]),
                'embarked': sys.argv[7]
            }
        except (ValueError, IndexError) as e:
            print(f"Error parsing arguments: {e}")
            print("Use: python src/predict.py <pclass> <sex> <age> <sibsp> <parch> <fare> <embarked>")
            sys.exit(1)

    # Make prediction
    result = make_prediction(model, passenger_data)

    # Print results
    print("\n" + "="*40)
    print("TITANIC SURVIVAL PREDICTION")
    print("="*40)
    print(f"Prediction: {result['survived']}")
    print(f"Survival Probability: {result['survival_probability']:.3f}")
    print(f"Death Probability: {result['death_probability']:.3f}")
    print("="*40)

if __name__ == "__main__":
    main()
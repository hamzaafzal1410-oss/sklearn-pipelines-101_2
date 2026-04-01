# Sklearn Pipelines 101 - Titanic Survival Prediction

A complete machine learning project demonstrating sklearn pipelines for predicting Titanic passenger survival using the classic Titanic dataset.

## Project Overview

This project showcases:
- **Data preprocessing** with ColumnTransformer
- **ML pipeline** using sklearn Pipeline
- **Cross-validation** for robust evaluation
- **Model persistence** with joblib
- **Command-line interface** for training and prediction

## Dataset

**Titanic Dataset** from seaborn (originally from Kaggle)
- **Features**: passenger class, sex, age, siblings/spouses, parents/children, fare, embarkation port
- **Target**: survival (0 = died, 1 = survived)
- **Size**: 891 passengers

## Setup

1. **Clone/download the repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run EDA notebook** (optional):
   ```bash
   jupyter notebook notebooks/01_eda.ipynb
   ```

## Training the Model

Run the training script to build the pipeline, perform cross-validation, and save the model:

```bash
python src/train.py
```

**What it does**:
- Loads the Titanic dataset
- Creates a preprocessing pipeline (imputation, scaling, encoding)
- Trains a LogisticRegression model
- Performs 5-fold stratified cross-validation
- Prints evaluation metrics
- Saves the trained pipeline to `artifacts/model.joblib`

## Evaluation Metrics

Cross-validation results (5-fold stratified):

| Metric     | Mean    | Std     | Scores |
|------------|---------|---------|--------|
| accuracy  | 0.787   | 0.015   | [0.788, 0.770, 0.787, 0.775, 0.815] |
| precision | 0.745   | 0.030   | [0.731, 0.721, 0.788, 0.712, 0.773] |
| recall    | 0.678   | 0.048   | [0.710, 0.647, 0.603, 0.691, 0.739] |
| f1        | 0.709   | 0.027   | [0.721, 0.682, 0.683, 0.701, 0.756] |
| roc_auc   | 0.831   | 0.018   | [0.854, 0.809, 0.813, 0.829, 0.850] |

## Making Predictions

Use the trained model to predict survival for new passengers:

```bash
# Individual arguments (recommended)
python src/predict.py 3 male 25 0 0 7.25 S

# Or with JSON
python src/predict.py json '{"pclass": 3, "sex": "male", "age": 25, "sibsp": 0, "parch": 0, "fare": 7.25, "embarked": "S"}'
```

**Example output**:
```
========================================
TITANIC SURVIVAL PREDICTION
========================================
Prediction: No
Survival Probability: 0.179
Death Probability: 0.821
========================================
```

**Required features** for prediction:
- `pclass`: Passenger class (1, 2, or 3)
- `sex`: "male" or "female"
- `age`: Age in years (float)
- `sibsp`: Number of siblings/spouses aboard (int)
- `parch`: Number of parents/children aboard (int)
- `fare`: Passenger fare (float)
- `embarked`: Port of embarkation ("C", "Q", or "S")

## Pipeline Architecture

```
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ['age', 'fare', 'sibsp', 'parch']),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first'))
        ]), ['sex', 'embarked'])
    ])),
    ('classifier', LogisticRegression())
])
```

## Key Features

- **Automated preprocessing**: Handles missing values, scaling, and encoding
- **No data leakage**: Preprocessing fitted only on training data during CV
- **Reproducible**: Same preprocessing applied to training and new data
- **Serializable**: Entire pipeline saved as single object
- **Production-ready**: Easy to deploy and use for predictions

## Files Structure

```
sklearn-pipelines-101/
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory data analysis
├── src/
│   ├── train.py              # Training script
│   └── predict.py            # Prediction script
├── artifacts/
│   └── model.joblib          # Saved trained model
├── data/
│   └── raw/                  # Raw data storage
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Dependencies

- scikit-learn >= 1.3.0
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- joblib >= 1.1.0
- jupyter >= 1.0.0

# Loan Default Prediction - Logistic Regression from Scratch

A complete implementation of logistic regression for credit risk modeling, built from scratch using only NumPy and Pandas (no sklearn). This project demonstrates end-to-end machine learning including data preprocessing, model training, and comprehensive evaluation.

## Features

- **Custom Logistic Regression**: Gradient descent implementation with numerically stable sigmoid function
- **Robust Preprocessing**:
  - Automatic normalization of numeric features
  - One-hot encoding for categorical variables
  - Binary encoding for yes/no columns
  - Missing value imputation using median
- **Comprehensive Evaluation**:
  - ROC curve with AUC calculation
  - Precision-Recall curves
  - Confusion matrix (TP, FP, TN, FN)
  - Accuracy, Precision, Recall, F1 score
  - Brier score for probability calibration
  - Cost-based threshold optimization
- **Visualizations**: ROC and PR curves with threshold annotations
- **Progress Tracking**: Training progress bar with tqdm

## Requirements

- Python 3.8+
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- tqdm >= 4.62.0

## Installation

```bash
pip install numpy pandas matplotlib tqdm
```

Or create a `requirements.txt` and run:
```bash
pip install -r requirements.txt
```

## Data

This project uses the **Loan Default Dataset** from Kaggle:

**[Download Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default)**

**Setup:**
1. Download the CSV file from Kaggle
2. Place it in the project root directory as `loan_default.csv`
3. The dataset is already in `.gitignore` and will not be committed to version control

## Usage

1. Ensure you have downloaded `loan_default.csv` (see Data section above)
2. Run the main script:
```bash
python main.py
```

The script will:
- Load and preprocess the data
- Train a logistic regression model using gradient descent
- Evaluate performance on a test set
- Display ROC and Precision-Recall curves
- Print confusion matrix and performance metrics

**Alternative:** You can also import and use individual modules:
```python
from preprocessing import load_and_preprocess_data
from model import sigmoid, descend
from evaluation import find_stats, brier_score

# Load data
x_train, x_test, y_train, y_test = load_and_preprocess_data('loan_default.csv')

# Train model
beta, losses = descend(x_train, y_train, alpha=0.1, iterations=2000)

# Evaluate
# ... your code here
```

## Data Format

The script expects a CSV file with the following columns:

**Numeric features:**
- Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio

**Binary features (yes/no):**
- HasMortgage, HasDependents, HasCoSigner

**Categorical features:**
- Education, EmploymentType, MaritalStatus, LoanPurpose

**Target variable:**
- Default (0 or 1)

**ID column:**
- LoanID (will be dropped during preprocessing)

## Methodology

### Preprocessing
1. Numeric columns are standardized using z-score normalization: `(x - mean) / std`
2. Binary columns are converted from yes/no to 1/0
3. Categorical columns are one-hot encoded (with baseline removal to avoid multicollinearity)
4. Missing values are filled with column median
5. An intercept column is added for the bias term

### Model Training
- Algorithm: Gradient descent with vectorized operations
- Loss function: Binary cross-entropy (log loss)
- Learning rate: 0.1 (default)
- Iterations: 2000 (default)
- Train/test split: 80/20 with random seed 42

### Evaluation
- Multiple classification thresholds evaluated (focused around 0.05-0.15 range)
- Cost-based optimization using configurable false positive and false negative costs
- Default costs: FP = $1,000, FN = $10,000 (reflecting credit risk business logic)

## Example Output

```
Confusion Matrix (threshold = 0.08)
TP: 450 FP: 120
FN: 50 TN: 380
Accuracy:  83.00%
Precision: 78.95%
Recall:    90.00%
F1:        0.8421
Brier:     0.1234
ROC AUC: 0.8756
Optimal Cost: 125000.0000
Optimal Threshold: 0.0895
```

## Project Structure

```
misc-coding/
├── main.py              # Main script - orchestrates the full pipeline
├── preprocessing.py     # Data loading and preprocessing functions
├── model.py             # Logistic regression model (sigmoid, gradient descent)
├── evaluation.py        # Evaluation metrics and plotting functions
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
├── cd-core.py           # Original monolithic script (kept for reference)
└── loan_default.csv     # Data file (not committed - download from Kaggle)
```

### Module Breakdown

**preprocessing.py:**
- `to_binary()` - Convert yes/no strings to 1/0
- `load_and_preprocess_data()` - Complete data pipeline with normalization, encoding, and train/test split

**model.py:**
- `sigmoid()` - Numerically stable sigmoid activation
- `find_log_loss()` - Binary cross-entropy loss function
- `descend()` - Gradient descent training with progress bar

**evaluation.py:**
- `find_tf_fp()` - Confusion matrix (TP, FP, TN, FN)
- `find_stats()` - Calculate accuracy, precision, recall, F1
- `brier_score()` - Probability calibration metric
- `find_optimal_threshold()` - Cost-based threshold optimization
- `plot_roc_and_pr_curves()` - Visualization of model performance

## Future Improvements

- Regularization (L1/L2) to prevent overfitting
- Cross-validation for more robust evaluation
- Feature importance analysis (coefficient magnitudes)
- Calibration curves for probability reliability
- Command-line arguments for hyperparameters
- Model persistence (save/load trained models)

## Notes

- The sigmoid function uses separate calculations for positive and negative values to avoid numerical overflow
- The log loss function clips probabilities to [1e-15, 1-1e-15] to avoid log(0) errors
- Train/test split uses a fixed random seed (42) for reproducibility

## Author

Built as a learning project to understand logistic regression fundamentals without relying on high-level libraries.

## License

MIT License (or specify your preferred license)

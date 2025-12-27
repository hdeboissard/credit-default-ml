import pandas as pd
import numpy as np


def to_binary(column):
    # checking if the data is numeric
    if pd.api.types.is_numeric_dtype(column):
        return column.astype(int)

    s = column.astype(str).str.strip().str.lower() # making all data in the column a lowercase string and removing whitespace

    # setting up the mapping from str to binary
    mapper = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0
    }

    edited_column = s.map(mapper)

    return edited_column


def load_and_preprocess_data(filepath='loan_default.csv', test_size=0.2, random_seed=42):
    df = pd.read_csv(filepath)

    # labelling the y data as the default column and subsequently removing it from df
    y = df["Default"].astype(int)
    x = df.drop(columns=["Default"])
    # also removing loan id
    x = x.drop(columns=["LoanID"])



    rng = np.random.default_rng(random_seed) # numpy random number generator object - using seed 42 so we get the same shuffle each time
    n = x.shape[0] # setting n equal to the number of measurements
    idx = rng.permutation(n) # creates a list of numbers 0 - (n-1), distributed randomly
    split = int((1 - test_size) * n) # creating an 80 20 training testing split
    train_idx, test_idx = idx[:split], idx[split:]
    # setting x and y train and test
    x_train = x.iloc[train_idx].copy()
    x_test = x.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].to_numpy()
    y_test = y.iloc[test_idx].to_numpy()
    
    # identifying all the columns with numeric data to normalise them
    numeric_cols = [
        "Age", "Income", "LoanAmount", "CreditScore",
        "MonthsEmployed", "NumCreditLines",
        "InterestRate", "LoanTerm", "DTIRatio"
    ]

    # storing mean and std for later use
    train_numeric_cols_mean = []
    train_numeric_cols_std = []

    for col in numeric_cols:
        # making all data numeric and replacing errors with NaN
        x_train[col] = pd.to_numeric(x_train[col], errors='coerce')
        x_test[col] = pd.to_numeric(x_test[col], errors='coerce') 
        # defining the mean, std and median of the training data set to normalise the training and test set
        train_mean = x_train[col].mean()
        train_std = x_train[col].std()
        train_median = x_train[col].median()
        # filling errors with the median
        x_train[col] = x_train[col].fillna(train_median)
        x_test[col] = x_test[col].fillna(train_median)
        # numeric data is now both a) entirely numerial b) null values replaced with the median of training set

        train_numeric_cols_mean.append(train_mean)
        train_numeric_cols_std.append(train_std)
        if train_std == 0:
            train_std = 1
        # setting std to 1 in case of std = 0 (admittedly v unlikely)
        x_train[col] = ( x_train[col] - train_mean ) / train_std
        x_test[col] = ( x_test[col] - train_mean ) / train_std

    # defining columns which are filled as 'yes' or 'no' and changing them to 1 or 0
    binary_cols = [
        "HasMortgage",
        "HasDependents",
        "HasCoSigner"
    ]

    # using a for loop to change each column to a binary for verbal binary columns (yes/no etc)
    for col in binary_cols:
        x_train[col] = to_binary(x_train[col])
        x_test[col] = to_binary(x_test[col])

    # now tackling columns with multiple categories (e.g Education: Bachelors, Masters, None etc)
    cat_cols = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"]

    for col in cat_cols:
        x_train[col] = x_train[col].fillna("unknown").astype(str).str.strip() #removing white space and filling unknowns
        x_test[col] = x_test[col].fillna("unknown").astype(str).str.strip() #removing white space and filling unknowns
        categories = sorted(x_train[col].unique()) # finding the unique values within each column for x_train
        baseline = categories[0] # need to create a baseline to remove one column to remove interdependencies

        for category in categories:
            if category == baseline:
                continue

            new_col_name = f"{col}_{category}"
            # checks if the parent column (e.g education) has the correct sub category (e.g bachelors) and then gives a 1 or 0
            x_train[new_col_name] = (x_train[col] == category).astype(int)
            x_test[new_col_name] = (x_test[col] == category).astype(int)

        # dropping the 'parent' column
        x_train = x_train.drop(columns=[col])
        x_test = x_test.drop(columns=[col])
        

    # adding an intercept column
    x_train.insert(0, "Intercept", 1.0)
    x_test.insert(0, "Intercept", 1.0)

    # No string columns left
    print(x_train.select_dtypes(include=["object"]).columns)
    print(x_test.select_dtypes(include=["object"]).columns)

    # making both x and y into numpy arrays
    x_train = x_train.to_numpy(dtype=float)
    x_test = x_test.to_numpy(dtype=float)

    return x_train, x_test, y_train, y_test

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

    # identifying all the columns with numeric data to normalise them
    numeric_cols = [
        "Age", "Income", "LoanAmount", "CreditScore",
        "MonthsEmployed", "NumCreditLines",
        "InterestRate", "LoanTerm", "DTIRatio"
    ]

    # storing mean and std for later use
    numeric_cols_mean = []
    numeric_cols_std = []

    for col in numeric_cols:
        # making all data numeric and replacing errors with NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # filling errors with the median
        df[col] = df[col].fillna(df[col].median())
        # numeric data is now both a) entirely numerial b) null values replaced with the median

        # defining the mean and std per column for normalisation (x_norm = (x - mean) / std)
        mean = df[col].mean()
        std = df[col].std()
        numeric_cols_mean.append(mean)
        numeric_cols_std.append(std)
        if std == 0:
            std = 1
        # setting std to 1 in case of std = 0 (admittedly v unlikely)
        df[col] = ( df[col] - mean ) / std

    # dropping the LoanID column which has no predictive power
    df = df.drop(columns=['LoanID'])

    # labelling the y data as the default column and subsequently removing it from df
    y = df["Default"].astype(int)
    df = df.drop(columns=["Default"])

    # defining columns which are filled as 'yes' or 'no' and changing them to 1 or 0
    binary_cols = [
        "HasMortgage",
        "HasDependents",
        "HasCoSigner"
    ]

    # using a for loop to change each column to a binary for verbal binary columns (yes/no etc)
    for col in binary_cols:
        df[col] = to_binary(df[col])

    # now tackling columns with multiple categories (e.g Education: Bachelors, Masters, None etc)
    cat_cols = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"]

    for col in cat_cols:
        df[col] = df[col].fillna("unknown").astype(str).str.strip() #removing white space and filling unknowns
        categories = sorted(df[col].unique()) # finding the unique values within each column
        baseline = categories[0] # need to create a baseline to remove one column to remove interdependencies

        for category in categories:
            if category == baseline:
                continue

            new_col_name = f"{col}_{category}"
            # checks if the parent column (e.g education) has the correct sub category (e.g bachelors) and then gives a 1 or 0
            df[new_col_name] = (df[col] == category).astype(int)

        # dropping the 'parent' column
        df = df.drop(columns=[col])

    # adding an intercept column
    df.insert(0, "Intercept", 1.0)

    # No string columns left
    print(df.select_dtypes(include=["object"]).columns)

    # All values numeric
    print(df.dtypes.value_counts())

    # making both x and y into numpy arrays
    x = df.to_numpy(dtype=float)
    y = y.to_numpy(dtype=float)

    rng = np.random.default_rng(random_seed) # numpy random number generator object - using seed 42 so we get the same shuffle each time
    n = x.shape[0] # setting n equal to the number of measurements
    idx = rng.permutation(n) # creates a list of numbers 0 - (n-1), distributed randomly
    split = int((1 - test_size) * n) # creating an 80 20 training testing split
    train_idx, test_idx = idx[:split], idx[split:]
    # setting x and y train and test
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    return x_train, x_test, y_train, y_test

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import resample


def initialize_loan_approval_data(verbose=False):
    # Load dataset
    df = pd.read_csv("data/loan_data.csv")

    if verbose:
        print("loan_data dataset shape:", df.shape)
        print("First 5 rows:\n", df.head())
        print("Column types:\n", df.dtypes)

    # Separate features and target
    y = df["loan_status"]
    df = df.drop(
        columns=[
            "loan_status"
        ]
    )

    # Move credit_score and loan_amount to the start of the dataframe, as they are the least important numerical features based on correlation analysis
    cols = df.columns.tolist()
    cols.remove("loan_int_rate")
    cols.remove("loan_percent_income")
    cols = ["loan_int_rate", "loan_percent_income"] + cols
    df = df[cols]

    if verbose:
        # Count unique classes in target
        class_counts = y.value_counts()
        print("Class distribution in 'loan_status':\n", class_counts)

    # Split dataset into train, test, and validation sets (70% train, 15% val, 15% test)
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )

    # Undersample majority class in train set to balance the dataset
    df_train = pd.concat([X_train, y_train], axis=1)
    df_train_majority = df_train[df_train["loan_status"] == 0]
    df_train_minority = df_train[df_train["loan_status"] == 1]

    target_majority_count = min(len(df_train_majority), 2 * len(df_train_minority))

    df_train_majority_downsampled = resample(
        df_train_majority,
        replace=False,
        n_samples=target_majority_count,
        random_state=42,
    )
    df_train_balanced = pd.concat([df_train_majority_downsampled, df_train_minority])
    X_train = df_train_balanced.drop(columns=["loan_status"])
    y_train = df_train_balanced["loan_status"]

    if verbose:
        class_counts = y_train.value_counts()
        print(
            "Class distribution in 'loan_status' in train set after undersampling:\n",
            class_counts,
        )

    # One hot encode categorical features and scale numerical features. Fit on train set, transform on all sets.
    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    if verbose:
        print("Categorical features:", categorical_features)
        print("Numerical features:", numerical_features)

    encoder = OneHotEncoder()
    scaler = StandardScaler()

    encoder.fit(X_train[categorical_features])
    scaler.fit(X_train[numerical_features])

    X_train = encode_and_scale_loan_dataframe(X_train, encoder, scaler)
    X_test = encode_and_scale_loan_dataframe(X_test, encoder, scaler)
    X_val = encode_and_scale_loan_dataframe(X_val, encoder, scaler)

    eps = 0.1  # margin above zero

    # Compute shift on TRAIN SET ONLY
    min_vals = X_train[numerical_features].min(axis=0)  # Series
    shift = -min_vals + eps  # Series
    hash_features = ["loan_int_rate", "loan_percent_income"]
    shift[hash_features] = shift[hash_features] + 1  # To account for later hashing shift of +1

    # Apply same shift to train/val/test
    for df_ in (X_train, X_val, X_test):
        df_[numerical_features] = df_[numerical_features] + shift

    if verbose:
        print("Shift applied to hash features:", shift.to_dict())

    print("Statistics of hash features from dataset after positive normalization:\n",
          X_train[["loan_int_rate", "loan_percent_income"]].describe())

    if verbose:
        print("Features after encoding and scaling:\n", X_train.head())
        print("Feature names after encoding and scaling:\n", X_train.columns.tolist())
        print("Numerical feature statistics after scaling:\n", X_train[numerical_features].describe())

    # Print correlation of each feature with target
    correlation_with_target = X_train.copy()
    correlation_with_target["loan_status"] = y_train
    corr_matrix = correlation_with_target.corr()
    if verbose:
        print("Correlation of features with target 'loan_status':\n",
              corr_matrix["loan_status"].sort_values(ascending=False))

    if verbose:
        # plot correlation matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(12, 10))
        corr = X_train.copy()
        corr["loan_status"] = y_train
        correlation_matrix = corr.corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
        plt.title("Feature Correlation Matrix")
        plt.show()

    if verbose:
        print(
            X_train.shape,
            X_test.shape,
            X_val.shape,
            y_train.shape,
            y_test.shape,
            y_val.shape,
        )
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

    if verbose:
        print(
            "Shapes of tensors (X_train, Y_train):",
            X_train_tensor.shape,
            y_train_tensor.shape,
        )

    # Split into DataLoader for batching
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

    return train_dataset, test_dataset, val_dataset, encoder, scaler


def encode_and_scale_loan_dataframe(
        df: pd.DataFrame,
        encoder: OneHotEncoder,
        scaler: StandardScaler,
) -> pd.DataFrame:
    categorical_features = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    X_cat = encoder.transform(df[categorical_features]).toarray()
    X_num = scaler.transform(df[numerical_features])
    X = pd.DataFrame(
        np.hstack((X_num, X_cat)),
        columns=numerical_features
                + encoder.get_feature_names_out(categorical_features).tolist(),
    )

    return X


if __name__ == "__main__":
    train_dataset, test_dataset, val_dataset, encoder, scaler = initialize_loan_approval_data(
        verbose=True
    )
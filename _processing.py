# useful data processing functions
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer


# process dataframe
def process_df(df, k=10, verbose=False):
    # Label encoding
    encoder = LabelEncoder()

    # Force all labels to string type
    df[' Label'] = df[' Label'].astype(str)
    df[' Label'] = encoder.fit_transform(df[' Label'])
    df[' Label'].value_counts().sum

    # Replace NaNs and infs
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)

    # Convert to integers
    df = df.astype(int)

    # Remove constant columns (excluding the label)
    label_col = ' Label'
    feature_cols = [col for col in df.columns if col != label_col]
    nunique = df[feature_cols].nunique()
    constant_cols = nunique[nunique <= 1].index
    # if verbose and len(constant_cols) > 0:
    #     print(f"Removed constant features: {list(constant_cols)}")
    df = df.drop(columns=constant_cols)

    # Separate features and label
    X = df.drop(' Label', axis=1)
    Y = df[' Label']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Impute (should be redundant after fillna, but keeping in case you change fillna)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_scaled)

    # Feature selection
    k = min(k, X.shape[1])
    k_best = SelectKBest(score_func=f_classif, k=k)
    X_new = k_best.fit_transform(X_imputed, Y)
    selected_features_mask = k_best.get_support()
    selected_feature_names = X.columns[selected_features_mask]

    # Construct new DataFrame
    df_new = X[selected_feature_names].copy()
    df_new['label'] = Y

    # Convert labels to binary: 0 = normal, 1 = outlier
    Y_binary = np.where(df_new['label'] != 0, 1, 0)
    X_final = df_new.iloc[:, :-1].values

    # Print label distribution if requested
    if verbose:
        unique, counts = np.unique(Y_binary, return_counts=True)
        label_counts = dict(zip(unique, counts))
        print("Label counts:", label_counts)

    return X_final, Y_binary, selected_feature_names

# apply features found in process_df above to another dataframe
def apply_features(df, selected_feature_names):
    # Label encoding
    encoder = LabelEncoder()

    # Force all labels to string type
    df[' Label'] = df[' Label'].astype(str)
    df[' Label'] = encoder.fit_transform(df[' Label'])
    df[' Label'].value_counts().sum

    # Replace NaNs and infs
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)

    # Convert to integers
    df = df.astype(int)

    # Remove constant columns (excluding the label)
    label_col = ' Label'
    feature_cols = [col for col in df.columns if col != label_col]
    nunique = df[feature_cols].nunique()
    constant_cols = nunique[nunique <= 1].index
    # if verbose and len(constant_cols) > 0:
    #     print(f"Removed constant features: {list(constant_cols)}")
    df = df.drop(columns=constant_cols)

    # Separate features and label
    X = df.drop(' Label', axis=1)
    Y = df[' Label']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Impute (should be redundant after fillna, but keeping in case you change fillna)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_scaled)

    # Construct new DataFrame
    df_new = X[selected_feature_names].copy()
    df_new['label'] = Y

    # Convert labels to binary: 0 = normal, 1 = outlier
    Y_binary = np.where(df_new['label'] != 0, 1, 0)
    X_final = df_new.iloc[:, :-1].values

    return X_final, Y_binary
    
def summarize_array(arr, name='Array'):
    print(f"--- {name} ---")
    print(f"Type: {type(arr)}")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    print(f"Min: {np.min(arr)}  Max: {np.max(arr)}")
    print(f"Mean: {np.mean(arr):.3f}  Std: {np.std(arr):.3f}")
    print(f"Contains NaNs: {np.isnan(arr).any()}")
    print()
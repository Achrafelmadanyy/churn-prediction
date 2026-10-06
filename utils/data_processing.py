import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from config import *

def load_and_preprocess_data():
    """Load and preprocess the churn dataset."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Normalize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Normalize string values
    string_columns = df.select_dtypes(include='str').columns
    for col in string_columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')
    
    # Fix numeric columns
    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(df.totalcharges.mean())
    
    # Encode target
    df.churn = (df.churn == 'yes').astype(int)
    
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Churn rate: {df.churn.mean():.2%}")
    
    return df

def prepare_features(df, fit_vectorizer=True, vectorizer=None, scaler=None):
    """Prepare features for modeling."""
    # Remove customerid if present
    if 'customerid' in df.columns:
        df = df.drop('customerid', axis=1)
    
    # Split features and target
    y = df['churn'].values if 'churn' in df.columns else None
    X_df = df.drop('churn', axis=1) if 'churn' in df.columns else df
    
    # Convert to dictionary format
    data_dict = X_df[CATEGORICAL_FEATURES + NUM_FEATURES].to_dict(orient='records')
    
    # Vectorize
    if fit_vectorizer:
        vectorizer = DictVectorizer(sparse=False)
        X_encoded = vectorizer.fit_transform(data_dict)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)
    else:
        X_encoded = vectorizer.transform(data_dict)
        X_scaled = scaler.transform(X_encoded)
    
    return X_encoded, X_scaled, y, vectorizer, scaler

def split_data(df):
    """Split data into train, validation, and test sets."""
    print("\nSplitting data...")
    
    df_train_full, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df.churn
    )
    df_train, df_val = train_test_split(
        df_train_full, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=df_train_full.churn
    )
    
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_train_full = df_train_full.reset_index(drop=True)
    
    print(f"Train set: {len(df_train)} samples")
    print(f"Validation set: {len(df_val)} samples")
    print(f"Test set: {len(df_test)} samples")
    
    return df_train, df_val, df_test, df_train_full
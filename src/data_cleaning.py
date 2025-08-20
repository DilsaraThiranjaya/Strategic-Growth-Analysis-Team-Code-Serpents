"""
Data Cleaning and Preprocessing Module
=====================================

This module contains functions for cleaning and preprocessing the Online Retail II dataset.
It handles missing values, duplicates, cancellations, and data type conversions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
    """
    Load the online retail dataset from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Raw dataset
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def initial_data_assessment(df):
    """
    Perform initial data quality assessment.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
        
    Returns:
    --------
    dict
        Dictionary containing assessment results
    """
    assessment = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    
    print("=== INITIAL DATA ASSESSMENT ===")
    print(f"Dataset Shape: {assessment['shape']}")
    print(f"Total Duplicates: {assessment['duplicates']}")
    print(f"Memory Usage: {assessment['memory_usage']:.2f} MB")
    print("\nMissing Values:")
    for col, missing in assessment['missing_values'].items():
        if missing > 0:
            print(f"  {col}: {missing} ({missing/len(df)*100:.2f}%)")
    
    return assessment


def remove_duplicates(df):
    """
    Remove duplicate rows from the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
        
    Returns:
    --------
    pd.DataFrame
        Dataset without duplicates
    """
    initial_shape = df.shape
    df_clean = df.drop_duplicates()
    removed = initial_shape[0] - df_clean.shape[0]
    
    print(f"Duplicates removed: {removed}")
    return df_clean


def handle_missing_customer_ids(df):
    """
    Handle missing Customer IDs by removing rows without customer information.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
        
    Returns:
    --------
    pd.DataFrame
        Dataset with valid Customer IDs only
    """
    initial_shape = df.shape
    df_clean = df.dropna(subset=['Customer ID'])
    removed = initial_shape[0] - df_clean.shape[0]
    
    print(f"Rows with missing Customer ID removed: {removed}")
    return df_clean


def remove_cancelled_orders(df):
    """
    Remove cancelled orders (Invoice numbers starting with 'C').
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
        
    Returns:
    --------
    pd.DataFrame
        Dataset without cancelled orders
    """
    initial_shape = df.shape
    df_clean = df[~df['Invoice'].str.startswith('C', na=False)]
    removed = initial_shape[0] - df_clean.shape[0]
    
    print(f"Cancelled orders removed: {removed}")
    return df_clean


def remove_zero_price_records(df):
    """
    Remove records with zero unit price.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
        
    Returns:
    --------
    pd.DataFrame
        Dataset without zero-price records
    """
    initial_shape = df.shape
    df_clean = df[df['Price'] > 0]
    removed = initial_shape[0] - df_clean.shape[0]
    
    print(f"Zero-price records removed: {removed}")
    return df_clean


def filter_non_product_codes(df):
    """
    Filter out non-product stock codes (like POST, M, etc.).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
        
    Returns:
    --------
    pd.DataFrame
        Dataset with product codes only
    """
    initial_shape = df.shape
    
    # Common non-product codes
    non_product_codes = ['POST', 'M', 'BANK CHARGES', 'PADS', 'DOT']
    
    # Filter out non-product codes
    df_clean = df[~df['StockCode'].isin(non_product_codes)]
    
    # Also filter out codes that are clearly non-products (manual adjustments, etc.)
    df_clean = df_clean[~df_clean['StockCode'].str.contains('ADJUST|MANUAL|DISCOUNT', na=False)]
    
    removed = initial_shape[0] - df_clean.shape[0]
    print(f"Non-product records removed: {removed}")
    
    return df_clean


def create_total_price_column(df):
    """
    Create TotalPrice column by multiplying Quantity and Price.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
        
    Returns:
    --------
    pd.DataFrame
        Dataset with TotalPrice column
    """
    df = df.copy()
    df['TotalPrice'] = df['Quantity'] * df['Price']
    print("TotalPrice column created")
    return df


def parse_invoice_date(df):
    """
    Parse InvoiceDate and create temporal features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
        
    Returns:
    --------
    pd.DataFrame
        Dataset with parsed date features
    """
    df = df.copy()
    
    # Convert to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Create temporal features
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['HourOfDay'] = df['InvoiceDate'].dt.hour
    df['DayName'] = df['InvoiceDate'].dt.day_name()
    df['MonthName'] = df['InvoiceDate'].dt.month_name()
    
    print("Temporal features created")
    return df


def convert_data_types(df):
    """
    Convert data types to appropriate formats.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
        
    Returns:
    --------
    pd.DataFrame
        Dataset with corrected data types
    """
    df = df.copy()
    
    # Convert Customer ID to integer
    df['Customer ID'] = df['Customer ID'].astype(int)
    
    # Ensure StockCode is string
    df['StockCode'] = df['StockCode'].astype(str)
    
    # Ensure Invoice is string
    df['Invoice'] = df['Invoice'].astype(str)
    
    print("Data types converted")
    return df


def clean_dataset(file_path):
    """
    Complete data cleaning pipeline.
    
    Parameters:
    -----------
    file_path : str
        Path to the raw CSV file
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataset ready for analysis
    """
    print("=== STARTING DATA CLEANING PIPELINE ===\n")
    
    # Load data
    df = load_data(file_path)
    if df is None:
        return None
    
    # Initial assessment
    initial_data_assessment(df)
    print("\n" + "="*50 + "\n")
    
    # Cleaning steps
    print("=== CLEANING STEPS ===")
    df = remove_duplicates(df)
    df = handle_missing_customer_ids(df)
    df = remove_cancelled_orders(df)
    df = remove_zero_price_records(df)
    df = filter_non_product_codes(df)
    
    # Feature engineering
    print("\n=== FEATURE ENGINEERING ===")
    df = create_total_price_column(df)
    df = parse_invoice_date(df)
    df = convert_data_types(df)
    
    print(f"\n=== CLEANING COMPLETE ===")
    print(f"Final dataset shape: {df.shape}")
    print(f"Data range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    
    return df


if __name__ == "__main__":
    # Example usage
    cleaned_data = clean_dataset("../data/online_retail.csv")
    if cleaned_data is not None:
        print("\nCleaned data sample:")
        print(cleaned_data.head())
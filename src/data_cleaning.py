"""
Data Sanitation and Preprocessing
=================================

A comprehensive function-based system for cleaning, preprocessing, and preparing 
raw e-commerce transaction data for advanced analytical modeling.

Team: Code Serpents  
Team Member: S. M. L.Lakshan jayawardhana
Project: Strategic Growth Analysis for UK E-Commerce Retailer
Phase: 1 - Data Sanitation and Preprocessing
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# 1. Data Loading & Initial Assessment
# ================================================================

def load_data(file_path):
    """Load and preprocess retail data"""
    
    d_types = {
        'Invoice': 'object',
        'StockCode': 'object',
        'Description': 'object',
        'Quantity': 'int64',
        'InvoiceDate': 'object',
        'Price': 'float64',
        'Customer ID': 'float64',
        'Country': 'object'
    }

    try:
        df = pd.read_csv(file_path, dtype=d_types, encoding="ISO-8859-1")
        print(f"Data Loaded successfully! \n {df.head()}")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except ValueError as ve:
        print(f"Data type mismatch : {ve}")
        return None
    except EncodingWarning as ew:
        print(f"Encoding warning : {ew}")
        return None
    except Exception as e:
        print(f"Unexpected error : {e}")
        return None

def assess_data(df):
    """Perform initial data assessment including info, statistics, and missing values"""
    
    if df is None:
        print("No data to assess.")
        return None

    print(f"\n=== DataFrame Information ===")
    print(df.info())
    
    print(f"\n=== Statistical Summary ===")
    print(df.describe(include='all'))
    
    print(f"\n=== Missing Values ===")
    print(df.isnull().sum())
    
    return df


# ================================================================
# 2. Data Cleaning & Quality Control
# ================================================================

def handle_duplicates(df):
    """Identify and remove duplicate rows from the dataset"""
    
    if df is None:
        print(f"No data to handle duplicates.")
        return None

    duplicates_count = df.duplicated().sum()
    print(f"\nNumber of duplicate rows are: {duplicates_count}")
    df = df.drop_duplicates()  
    print(f"Duplicate values removed. New shape is: {df.shape}")
    return df

def handle_missing_customer_id(df):
    """Remove rows with missing Customer ID values"""
    
    if df is None:
        print(f"No data to handle missing Customer ID.")
        return None

    missing_cusid = df['Customer ID'].isnull().sum()
    print(f"Number of rows missing customer id : {missing_cusid}")
    df = df.dropna(subset=['Customer ID'])
    print(f"Remove misssing customer id rows. New shape is : {df.shape}")

    print(df.isnull().sum())
    return df

def remove_cancelled_orders(df):
    """Filter out cancelled orders (invoices starting with 'C')"""
    
    if df is None:
        print("No data to remove cancelled orders.")
        return None

    cancellsd_order_count = df[df['Invoice'].str.startswith('C', na=False)].shape[0]
    print(f"\nNumber of cancelled orders: {cancellsd_order_count}")
    df = df[~df['Invoice'].str.startswith('C', na=False)]
    print(f"Remove cancelled orders. New shape is : {df.shape}")
    return df

def remove_zero_price(df):
    """Remove rows with zero or negative prices"""
    
    if df is None:
        print(f"No data to remove zero price rows.")
        return None

    zero_count = df[df['Price'] == 0].shape[0]
    print(f"\nNumber of zero price rows: {zero_count}")
    df = df[df['Price'] > 0]
    print(f"Zero price rows removed. New shape: {df.shape}")
    return df

def remove_non_product_codes(df):
    """Remove non-product stock codes (postage, bank charges, etc.)"""
    
    if df is None:
        print(f"No data to remove non-product codes.")
        return None

    non_product_codes = ['POST', 'M', 'BANK CHARGES', 'C2', 'DOT', 'CRUK'] 
    non_product_count = df[df['StockCode'].isin(non_product_codes)].shape[0]
    print(f"\nNumber of non product rows: {non_product_count}")
    df = df[~df['StockCode'].isin(non_product_codes)]
    print(f"Non product rows removed. New shape: {df.shape}")
    return df

def remove_negative_quantity(df):
    """Remove rows with negative or zero quantities"""
    
    if df is None:
        print(f"No data to remove negative quantity.")
        return None

    zero_qty = df[df['Quantity'] <= 0].shape[0]
    print(f"\nNumber of negative or zero Quantity: {zero_qty}")
    df = df[df['Quantity'] > 0]
    print(f"Negative or zero quantity rows removed. New shape: {df.shape}")
    return df


# ================================================================
# 3. Feature Engineering
# ================================================================

def add_total_price(df):
    """Create TotalPrice column by multiplying Quantity and Price"""
    
    if df is None:
        print(f"No data to add TotalPrice.")
        return None

    df['TotalPrice'] = df['Quantity'] * df['Price']
    print(f"\nTotalPrice column added. New shape: {df.shape}")
    print(f"{df[['Quantity', 'Price', 'TotalPrice']].head()}")
    return df

def add_date_columns(df):
    """Extract temporal features from InvoiceDate column"""
    
    if df is None:
        print(f"No data to add date columns.")
        return None

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='ISO8601')
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek  
    df['HourOfDay'] = df['InvoiceDate'].dt.hour
    print(f"\nDate columns added. New shape: {df.shape}")
    print(f"{df[['InvoiceDate', 'Year', 'Month', 'DayOfWeek', 'HourOfDay']].head()}")
    return df


# ================================================================
# 4. Data Type Conversion & Final Processing
# ================================================================

def convert_data_types(df):
    """Convert columns to appropriate data types for analysis"""
    
    if df is None:
        print(f"No data to convert types.")
        return None

    df['Customer ID'] = df['Customer ID'].astype(int)
    df['StockCode'] = df['StockCode'].astype(str)
    df['Quantity'] = df['Quantity'].astype(int)
    df['Price'] = df['Price'].astype('float32')
    df['TotalPrice'] = df['TotalPrice'].astype('float32')

    print(f"\nFinal Data Types:")
    print(f"{df.dtypes}")
    print(f"\nFinal Shape: {df.shape}")
    return df

def save_cleaned_data(df, save_path):
    """Save the cleaned dataset to a CSV file"""
    
    if df is None:
        print(f"No data to save.")
        return None

    df.to_csv(save_path, index=False)
    print("Cleaned data saved to 'online_retail_clean.csv'.")
    return None
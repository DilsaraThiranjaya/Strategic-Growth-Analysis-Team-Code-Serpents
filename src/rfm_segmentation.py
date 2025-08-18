"""
RFM Customer Segmentation Analyzer
==================================

A comprehensive class-based system for conducting RFM (Recency, Frequency, Monetary) 
analysis on e-commerce transaction data for customer segmentation.

RFM analysis is a method used for analyzing customer value by examining three dimensions:
- Recency: How recently a customer has made a purchase
- Frequency: How often a customer makes purchases  
- Monetary: How much money a customer spends

Team: Code Serpents
Team Member: G.A.Dilsara Thiranjaya 
Project: Strategic Growth Analysis for UK E-Commerce Retailer
Phase: 3 - Advanced Analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RFMDataProcessor:
    ## A class to handles data preprocessing and RFM metrics calculation.
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.validate_data()
        
    def validate_data(self):
        ## Validate that the input data has all required columns.
        
        # Check for required columns
        required_columns = ['Customer ID', 'Invoice', 'InvoiceDate', 'TotalPrice']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        else:
            print("Data validation passed. All required columns are present.")
        
        
        # Check and Convert data types if needed
        if self.data['Customer ID'].dtype != 'int64':
            self.data['Customer ID'] = self.data['Customer ID'].astype(int)
        
        if not pd.api.types.is_datetime64_any_dtype(self.data['InvoiceDate']):
            self.data['InvoiceDate'] = pd.to_datetime(self.data['InvoiceDate'])
    
    def calculate_rfm(self):
        ## Calculate Recency, Frequency, Monetary value for each customer.
        
        # Set snapshot date as one day after the most recent transaction
        snapshot_date = self.data['InvoiceDate'].max() + timedelta(days=1)
        
        # Calculate Recency, Frequency, Monetary values
        rfm_data = (self.data.groupby('Customer ID').agg(
            Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days), # Recency - days to snapshot_date since last purchase
            Frequency=('Invoice', 'nunique'), # Frequency - number of unique invoices
            Monetary=('TotalPrice', 'sum') # Monetary - total amount spent
        ).reset_index())
        
        # Ensure Monetary values are non-negative
        rfm_data = rfm_data[rfm_data['Monetary'] > 0]
        
        return rfm_data

    def get_customer_summary_stats(self, rfm_data: pd.DataFrame):
        ## Generate summary statistics for the RFM data.
        
        stats = {
            'total_customers': len(rfm_data),
            'recency_stats': rfm_data['Recency'].describe(),
            'frequency_stats': rfm_data['Frequency'].describe(),
            'monetary_stats': rfm_data['Monetary'].describe(),
            'snapshot_date': self.data['InvoiceDate'].max() + timedelta(days=1),
            'data_period': {
                'start_date': self.data['InvoiceDate'].min(),
                'end_date': self.data['InvoiceDate'].max()
            }
        }
        
        return stats

def load_and_validate_data(filepath: str):
    ## Utility function to load and validate data for RFM analysis.
    
    try:
        data = pd.read_csv(filepath)
        print(f"✅ Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Basic validation
        required_columns = ['Customer ID', 'Invoice', 'InvoiceDate', 'TotalPrice']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {', '.join(required_columns)}")
        
        return data
        
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")
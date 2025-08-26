"""
Exploratory Data Analysis
=========================

A comprehensive function-based system for conducting exploratory data analysis 
on e-commerce transaction data to uncover business insights and patterns.

Team: Code Serpents
Team Member: A.M.Supun Madhuranga
Project: Strategic Growth Analysis for UK E-Commerce Retailer
Phase: 2 - Exploratory Data Analysis & Insight Generation
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set consistent style for all plots
plt.style.use('default')
sns.set_palette("husl")

# ================================================================
# 1. Data Loading & Preprocessing
# ================================================================
def load_and_validate_data(filepath):
    """Load and validate retail data with comprehensive checks"""
    
    df = pd.read_csv(filepath, parse_dates=['InvoiceDate'])
    
    # Data validation checks
    print("Data Validation Summary:")
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Total Revenue: £{df['TotalPrice'].sum():,.2f}")
    print(f"Total Transactions: {df['Invoice'].nunique():,}")
    print(f"Total Products: {df['StockCode'].nunique():,}")
    print(f"Total Customers: {df['Customer ID'].nunique():,}")
    
    return df


# ================================================================
# 2. Time-Based Sales Analysis
# ================================================================
def get_monthly_sales(df):
    """Calculate monthly sales revenue"""
    return df.set_index('InvoiceDate').resample('M')['TotalPrice'].sum()

def get_feb_apr_comparison(monthly_sales):
    """Extract February and April sales for comparison"""
    feb_apr = monthly_sales[monthly_sales.index.month.isin([2, 4])].reset_index()
    feb_apr['Year'] = feb_apr['InvoiceDate'].dt.year
    feb_apr['Month'] = feb_apr['InvoiceDate'].dt.month_name()
    return feb_apr

def get_daily_sales(df):
    """Calculate sales by day of week"""
    
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    
    return df.groupby('DayOfWeek')['TotalPrice'].sum().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])

def get_hourly_sales(df):
    """Calculate sales by hour of day"""
    
    df['HourOfDay'] = df['InvoiceDate'].dt.hour
    
    return df.groupby('HourOfDay')['TotalPrice'].sum()


# ================================================================
# 3. Geographical Sales Analysis
# ================================================================
def get_country_analysis(df):
    """Analyze revenue distribution by country"""
    
    country_revenue = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False)
    top_10 = country_revenue.head(10)
    
    # UK revenue contribution
    uk_revenue = country_revenue.get('United Kingdom', 0)
    uk_percentage = (uk_revenue / country_revenue.sum()) * 100
    
    return {
        'country_revenue': country_revenue,
        'top_10_countries': top_10,
        'uk_revenue': uk_revenue,
        'uk_percentage': uk_percentage,
        'international_revenue': country_revenue.sum() - uk_revenue
    }


# ================================================================
# 4. Product Performance Analysis
# ================================================================
def get_product_analysis(df):
    """Analyze top-selling products by quantity and revenue"""
    
    product_df = df.groupby(['StockCode', 'Description']).agg(
        TotalQuantity=('Quantity', 'sum'),
        TotalRevenue=('TotalPrice', 'sum'),
        TransactionCount=('Invoice', 'nunique')
    ).reset_index()
    
    top_qty = product_df.sort_values('TotalQuantity', ascending=False).head(10)
    top_rev = product_df.sort_values('TotalRevenue', ascending=False).head(10)
    
    # Comparison analysis
    comparison = pd.merge(
        top_qty[['Description']],
        top_rev[['Description']],
        on='Description',
        how='outer',
        indicator=True
    )
    
    return {
        'product_df': product_df,
        'top_quantity': top_qty,
        'top_revenue': top_rev,
        'comparison': comparison
    }


# ================================================================
# 5. Helper Functions for Plotting
# ================================================================
def save_plot(fig, filename, dpi=300):
    """Helper function to save plots with consistent settings"""
    
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.show()
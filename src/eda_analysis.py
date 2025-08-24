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
import seaborn as sns

# ================================================================
# 1. Data Loading & Preprocessing
# ================================================================
def load_data(filepath):
    """Load and preprocess retail data"""
    
    df = pd.read_csv(filepath, parse_dates=['InvoiceDate'])
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    df['HourOfDay'] = df['InvoiceDate'].dt.hour
    return df


# ================================================================
# 2. Time-Based Sales Analysis
# ================================================================
def plot_monthly_sales(df):
    """Generate monthly sales revenue plot"""
    
    monthly_sales = df.set_index('InvoiceDate').resample('M')['TotalPrice'].sum()

    fig, ax = plt.subplots(figsize=(14, 6))
    monthly_sales.plot(kind='line', marker='o', linewidth=2, ax=ax)
    ax.set_title('Monthly Sales Revenue (Dec 2009 - Dec 2011)', fontsize=15)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Revenue (£)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig, monthly_sales


def plot_feb_apr_comparison(monthly_sales):
    """Compare February and April sales across years"""
    
    feb_apr = monthly_sales[monthly_sales.index.month.isin([2, 4])].reset_index()
    feb_apr['Year'] = feb_apr['InvoiceDate'].dt.year
    feb_apr['Month'] = feb_apr['InvoiceDate'].dt.month_name()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Year', y='TotalPrice', hue='Month', data=feb_apr, ax=ax)
    ax.set_title('February vs April Sales Comparison', fontsize=14)
    ax.set_ylabel('Revenue (£)', fontsize=12)
    plt.tight_layout()
    return fig, feb_apr


def plot_daily_sales(df):
    """Plot sales revenue by day of week"""
    
    daily_sales = df.groupby('DayOfWeek')['TotalPrice'].sum().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])

    fig, ax = plt.subplots(figsize=(12, 6))
    daily_sales.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Sales Revenue by Day of Week', fontsize=14)
    ax.set_xlabel('Day of Week', fontsize=12)
    ax.set_ylabel('Revenue (£)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig, daily_sales


def plot_hourly_sales(df):
    """Plot sales revenue by hour of day"""
    
    hourly_sales = df.groupby('HourOfDay')['TotalPrice'].sum()

    fig, ax = plt.subplots(figsize=(12, 6))
    hourly_sales.plot(kind='bar', color='salmon', ax=ax)
    ax.set_title('Sales Revenue by Hour of Day', fontsize=14)
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Revenue (£)', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig, hourly_sales


# ================================================================
# 3. Geographical Sales Analysis
# ================================================================
def analyze_countries(df):
    """Analyze revenue distribution by country"""
    
    country_revenue = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False)
    top_10 = country_revenue.head(10)

    # UK revenue contribution
    uk_revenue = country_revenue.get('United Kingdom', 0)
    uk_percentage = (uk_revenue / country_revenue.sum()) * 100

    # Top 10 countries plot
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    top_10.plot(kind='bar', color='teal', ax=ax1)
    ax1.set_title('Top 10 Countries by Sales Revenue', fontsize=15)
    ax1.set_xlabel('Country', fontsize=12)
    ax1.set_ylabel('Revenue (£)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Pie chart (UK vs International)
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.pie([uk_percentage, 100 - uk_percentage],
            labels=['UK', 'International'],
            autopct='%1.1f%%',
            colors=['#66b3ff', '#99ff99'],
            startangle=90,
            explode=(0.1, 0))
    ax2.set_title('UK vs International Revenue Contribution', fontsize=14)

    return {
        'figures': [fig1, fig2],
        'top_10': top_10,
        'uk_percentage': uk_percentage,
        'total_revenue': country_revenue.sum()
    }


# ================================================================
# 4. Product Performance Analysis
# ================================================================
def analyze_products(df):
    """Analyze top-selling products by quantity and revenue"""
    
    product_df = df.groupby(['StockCode', 'Description']).agg(
        TotalQuantity=('Quantity', 'sum'),
        TotalRevenue=('TotalPrice', 'sum')
    ).reset_index()

    top_qty = product_df.sort_values('TotalQuantity', ascending=False).head(10)
    top_rev = product_df.sort_values('TotalRevenue', ascending=False).head(10)

    # Visualization
    fig, ax = plt.subplots(2, 1, figsize=(14, 12))

    # Top by quantity
    sns.barplot(x='TotalQuantity', y='Description', data=top_qty,
                ax=ax[0], palette='Blues_d')
    ax[0].set_title('Top 10 Products by Quantity Sold', fontsize=14)
    ax[0].set_xlabel('Total Units Sold', fontsize=12)
    ax[0].set_ylabel('Product Description', fontsize=12)

    # Top by revenue
    sns.barplot(x='TotalRevenue', y='Description', data=top_rev,
                ax=ax[1], palette='Greens_d')
    ax[1].set_title('Top 10 Products by Revenue Generated', fontsize=14)
    ax[1].set_xlabel('Total Revenue (£)', fontsize=12)
    ax[1].set_ylabel('Product Description', fontsize=12)

    plt.tight_layout()

    # Comparison (did any product appear in both?)
    comparison = pd.merge(
        top_qty[['Description']],
        top_rev[['Description']],
        on='Description',
        how='outer',
        indicator=True
    )

    return {
        'figure': fig,
        'top_quantity': top_qty,
        'top_revenue': top_rev,
        'comparison': comparison
    }
"""
Strategic Recommendations
=========================

A comprehensive function-based system for translating analytical findings into 
actionable business strategies and marketing recommendations.

Team: Code Serpents
Team Member: K. D. Vihanga Heshan Bandara
Project: Strategic Growth Analysis for UK E-Commerce Retailer  
Phase: 4 - Strategic Recommendations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

# ================================================================
# 1. Data Loading & Validation
# ================================================================
def load_rfm_data(data_path):
    """Load the RFM customer segments data"""
    
    try:
        df = pd.read_csv(data_path)
        if 'Monetary' not in df.columns:
            raise ValueError("CSV must contain 'Monetary' column")
        print(f"Data loaded successfully: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# ================================================================
# 2. Monetary Distribution Analysis
# ================================================================
def analyze_monetary_distribution(df):
    """Analyze the distribution of monetary values to investigate wholesaler hypothesis."""
    
    monetary_values = df['Monetary']
    
    # Calculate key statistics
    stats = {
        'total_customers': len(monetary_values),
        'mean_spending': monetary_values.mean(),
        'median_spending': monetary_values.median(),
        'std_spending': monetary_values.std(),
        'min_spending': monetary_values.min(),
        'max_spending': monetary_values.max(),
        'skewness': monetary_values.skew(),
        'percentiles': {
            '25th': monetary_values.quantile(0.25),
            '75th': monetary_values.quantile(0.75),
            '90th': monetary_values.quantile(0.90),
            '95th': monetary_values.quantile(0.95),
            '99th': monetary_values.quantile(0.99)
        }
    }
    
    # Wholesaler analysis (top 5% spenders)
    wholesale_threshold = stats['percentiles']['95th']
    potential_wholesalers = monetary_values[monetary_values >= wholesale_threshold]
    
    stats['wholesale_analysis'] = {
        'threshold_95th': wholesale_threshold,
        'potential_wholesaler_count': len(potential_wholesalers),
        'potential_wholesaler_percentage': (len(potential_wholesalers) / len(monetary_values)) * 100,
        'wholesaler_revenue_share': (potential_wholesalers.sum() / monetary_values.sum()) * 100
    }
    
    return stats

# ================================================================
# 3. Visualization Functions
# ================================================================
def create_monetary_histogram(df, stats, bins=50, figsize=(15, 10)):
    """Create comprehensive histogram to visualize monetary distribution"""
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Customer Monetary Value Distribution Analysis\n(Investigating Wholesaler Hypothesis)', 
                 fontsize=16, fontweight='bold')
    
    monetary_values = df['Monetary']
    
    # 1. Standard histogram with key statistics
    axes[0, 0].hist(monetary_values, bins=bins, color='skyblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Distribution of Customer Spending')
    axes[0, 0].set_xlabel('Total Spending (£)')
    axes[0, 0].set_ylabel('Number of Customers')
    axes[0, 0].axvline(stats['mean_spending'], color='red', linestyle='--', 
                      label=f'Mean: £{stats["mean_spending"]:.0f}')
    axes[0, 0].axvline(stats['median_spending'], color='orange', linestyle='--', 
                      label=f'Median: £{stats["median_spending"]:.0f}')
    axes[0, 0].axvline(stats['percentiles']['95th'], color='purple', linestyle='--', 
                      label=f'95th percentile: £{stats["percentiles"]["95th"]:.0f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Log scale histogram
    log_monetary = np.log10(monetary_values + 1)
    axes[0, 1].hist(log_monetary, bins=bins, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Distribution (Log10 Scale)')
    axes[0, 1].set_xlabel('Log10(Total Spending + 1)')
    axes[0, 1].set_ylabel('Number of Customers')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Zoomed histogram (exclude top 1%)
    percentile_99 = stats['percentiles']['99th']
    main_distribution = monetary_values[monetary_values <= percentile_99]
    axes[1, 0].hist(main_distribution, bins=bins, color='gold', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Distribution (Excluding Top 1% Spenders)')
    axes[1, 0].set_xlabel('Total Spending (£)')
    axes[1, 0].set_ylabel('Number of Customers')
    axes[1, 0].axvline(main_distribution.mean(), color='red', linestyle='--', 
                      label=f'Mean (excl. top 1%): £{main_distribution.mean():.0f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Box plot
    box = axes[1, 1].boxplot(monetary_values, vert=True, patch_artist=True,
                            boxprops=dict(facecolor='lightcoral', alpha=0.7))
    axes[1, 1].set_title('Box Plot of Customer Spending\n(Shows Outliers)')
    axes[1, 1].set_ylabel('Total Spending (£)')
    axes[1, 1].set_xlabel('All Customers')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    return fig

# ================================================================
# 4. Customer Segmentation
# ================================================================
def identify_customer_segments(df):
    """Identify potential retail vs wholesale customers based on spending patterns."""
    
    p90 = df['Monetary'].quantile(0.90)
    p95 = df['Monetary'].quantile(0.95)
    
    def categorize_customer(monetary_value):
        if monetary_value >= p95:
            return 'High-Value Wholesale'
        elif monetary_value >= p90:
            return 'Potential Wholesale'
        else:
            return 'Retail Customer'
    
    df['Customer_Type'] = df['Monetary'].apply(categorize_customer)
    return df

# ================================================================
# 5. Insights Generation
# ================================================================
def generate_insights_summary(stats):
    """Generate text summary of wholesaler hypothesis findings."""
    
    summary = f"""
            WHOLESALER HYPOTHESIS INVESTIGATION SUMMARY
            ===========================================

            Dataset Overview:
            - Total Customers Analyzed: {stats['total_customers']:,}
            - Average Customer Spending: £{stats['mean_spending']:.2f}
            - Median Customer Spending: £{stats['median_spending']:.2f}
            - Distribution Skewness: {stats['skewness']:.2f} (Positive = Right-skewed)

            Key Findings:
            1. Distribution Shape: {'Highly right-skewed' if stats['skewness'] > 2 else 'Moderately skewed' if stats['skewness'] > 1 else 'Relatively normal'}
            2. Mean vs Median Gap: £{stats['mean_spending'] - stats['median_spending']:.2f}
            (Large gap indicates presence of high-spending outliers)

            Potential Wholesaler Analysis:
            - 95th Percentile Threshold: £{stats['wholesale_analysis']['threshold_95th']:.2f}
            - Potential Wholesalers (Top 5%): {stats['wholesale_analysis']['potential_wholesaler_count']} customers
            - Revenue Share of Top 5%: {stats['wholesale_analysis']['wholesaler_revenue_share']:.1f}%

            Interpretation:
            {'Strong evidence of two distinct customer groups - likely retail vs wholesale customers.' if stats['wholesale_analysis']['wholesaler_revenue_share'] > 30 else 'Some evidence of high-value customers, but distribution may not clearly show wholesale segment.'}
            {'The high skewness and significant revenue concentration suggest wholesale presence.' if stats['skewness'] > 2 else 'Distribution suggests more uniform customer base.'}
    """
    
    print(summary)
    return summary
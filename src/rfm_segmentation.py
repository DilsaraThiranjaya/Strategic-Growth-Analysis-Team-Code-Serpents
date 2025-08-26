"""
RFM Customer Segmentation Analyzer
==================================

A comprehensive function-based system for conducting RFM (Recency, Frequency, Monetary) 
analysis on e-commerce transaction data for customer segmentation.

Team: Code Serpents
Team Member: G.A.Dilsara Thiranjaya 
Project: Strategic Growth Analysis for UK E-Commerce Retailer
Phase: 3 - Advanced Analytics
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# 1. Data Preprocessing & RFM Calculation
# ================================================================

def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and preprocess the dataset"""
    
    required_columns = ['Customer ID', 'Invoice', 'InvoiceDate', 'TotalPrice']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    else:
        print("Data validation passed. All required columns are present.")

    # Fix data types
    if data['Customer ID'].dtype != 'int64':
        data['Customer ID'] = data['Customer ID'].astype(int)
    if not pd.api.types.is_datetime64_any_dtype(data['InvoiceDate']):
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    
    return data


def calculate_rfm(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate Recency, Frequency, and Monetary values for each customer"""
    
    snapshot_date = data['InvoiceDate'].max() + timedelta(days=1)
    
    rfm_data = (data.groupby('Customer ID').agg(
        Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency=('Invoice', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index())
    
    # Remove invalid rows
    rfm_data = rfm_data[rfm_data['Monetary'] > 0]
    return rfm_data


def get_customer_summary_stats(data: pd.DataFrame, rfm_data: pd.DataFrame) -> dict:
    """Generate summary statistics for the RFM data"""
    
    stats = {
        'total_customers': len(rfm_data),
        'recency_stats': rfm_data['Recency'].describe(),
        'frequency_stats': rfm_data['Frequency'].describe(),
        'monetary_stats': rfm_data['Monetary'].describe(),
        'snapshot_date': data['InvoiceDate'].max() + timedelta(days=1),
        'data_period': {
            'start_date': data['InvoiceDate'].min(),
            'end_date': data['InvoiceDate'].max()
        }
    }
    return stats


# ================================================================
# 2. RFM Scoring & Segmentation
# ================================================================

def create_segment_map() -> dict:
    """Create mapping from RFM scores to segment names"""
    
    segment_map = {}
    for r in range(1, 6):
        for f in range(1, 6):
            for m in range(1, 6):
                key = f"{r}{f}{m}"
                if r >= 4 and f >= 4 and m >= 4:
                    segment_map[key] = "Champions"
                elif r >= 3 and f >= 4:
                    segment_map[key] = "Loyal Customers"
                elif r >= 4 and f <= 2:
                    segment_map[key] = "Potential Loyalists"
                elif r == 5 and f <= 2:
                    segment_map[key] = "New Customers"
                elif r <= 2 and f >= 3:
                    segment_map[key] = "At-Risk Customers"
                elif r <= 2 and f <= 2:
                    segment_map[key] = "Hibernating"
    return segment_map


def assign_rfm_scores(rfm_data: pd.DataFrame) -> pd.DataFrame:
    """Assign RFM scores (1–5) for each metric"""
    
    rfm_scored = rfm_data.copy()
    
    rfm_scored['R_Score'] = pd.qcut(rfm_scored['Recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
    rfm_scored['F_Score'] = pd.qcut(rfm_scored['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
    rfm_scored['M_Score'] = pd.qcut(rfm_scored['Monetary'], 5, labels=[1,2,3,4,5], duplicates='drop')
    
    rfm_scored = rfm_scored.astype({'R_Score': int, 'F_Score': int, 'M_Score': int})
    
    rfm_scored['RFM_Segment'] = (
        rfm_scored['R_Score'].astype(str) +
        rfm_scored['F_Score'].astype(str) +
        rfm_scored['M_Score'].astype(str)
    )
    
    return rfm_scored


def assign_segment_labels(rfm_scored: pd.DataFrame, segment_map: dict) -> pd.DataFrame:
    """Attach segment labels to RFM scored data"""
    
    rfm_final = rfm_scored.copy()
    rfm_final['Segment'] = rfm_final['RFM_Segment'].map(segment_map)
    return rfm_final


def get_segment_summary(rfm_final: pd.DataFrame) -> pd.DataFrame:
    """Generate aggregated statistics by customer segment"""
    
    segment_summary = rfm_final.groupby('Segment').agg({
        'Customer ID': 'count',
        'Recency': ['mean', 'median'],
        'Frequency': ['mean', 'median'],
        'Monetary': ['mean', 'median', 'sum']
    }).round(2)

    segment_summary.columns = [
        'Customer_Count',
        'Avg_Recency', 'Median_Recency',
        'Avg_Frequency', 'Median_Frequency',
        'Avg_Monetary', 'Median_Monetary', 'Total_Revenue'
    ]
    
    segment_summary['Customer_Percentage'] = (
        segment_summary['Customer_Count'] / segment_summary['Customer_Count'].sum() * 100
    ).round(2)
    segment_summary['Revenue_Percentage'] = (
        segment_summary['Total_Revenue'] / segment_summary['Total_Revenue'].sum() * 100
    ).round(2)
    
    return segment_summary.sort_values('Customer_Count', ascending=False)


# ================================================================
# 3. Visualizations
# ================================================================

def plot_rfm_distributions(rfm_data: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)):
    """Plot histograms of RFM metrics"""
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('RFM Metrics Distribution Analysis', fontsize=16, fontweight='bold')
    
    axes[0, 0].hist(rfm_data['Recency'], bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Recency Distribution')
    axes[0, 0].axvline(rfm_data['Recency'].mean(), color='red', linestyle='--')

    axes[0, 1].hist(rfm_data['Frequency'], bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Frequency Distribution')
    axes[0, 1].axvline(rfm_data['Frequency'].mean(), color='red', linestyle='--')

    axes[1, 0].hist(np.log10(rfm_data['Monetary']), bins=50, color='gold', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Monetary Distribution (Log10 Scale)')
    axes[1, 0].axvline(np.log10(rfm_data['Monetary'].mean()), color='red', linestyle='--')
    
    plt.tight_layout()
    return fig


def plot_segment_analysis(segment_summary: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)):
    """Plot bar/pie charts for customer segments"""
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Customer Segment Analysis', fontsize=16, fontweight='bold')
    
    segment_summary.plot(kind='bar', y='Customer_Count', ax=axes[0, 0], color='steelblue', alpha=0.8)
    axes[0, 0].set_title('Number of Customers by Segment')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].legend().remove()
    
    axes[0, 1].pie(segment_summary['Revenue_Percentage'], labels=segment_summary.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Revenue Distribution by Segment')
    
    segment_summary.plot(kind='bar', y='Avg_Monetary', ax=axes[1, 0], color='darkorange', alpha=0.8)
    axes[1, 0].set_title('Average Monetary Value by Segment')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].legend().remove()
    
    plt.tight_layout()
    return fig


# ================================================================
# 4. Orchestrator
# ================================================================

def run_complete_analysis(data: pd.DataFrame):
    """Run the complete RFM analysis"""

    print("Starting RFM Analysis...")

    data = validate_data(data)
    rfm_data = calculate_rfm(data)
    summary_stats = get_customer_summary_stats(data, rfm_data)

    rfm_scored = assign_rfm_scores(rfm_data)
    segment_map = create_segment_map()
    rfm_final = assign_segment_labels(rfm_scored, segment_map)
    segment_summary = get_segment_summary(rfm_final)

    print("RFM Analysis Complete!")
    return {
        'rfm_data': rfm_data,
        'rfm_scored': rfm_scored,
        'rfm_final': rfm_final,
        'segment_summary': segment_summary,
        'summary_stats': summary_stats
    }
    
def generate_visualizations(rfm_final: pd.DataFrame, segment_summary: pd.DataFrame) -> dict[str, plt.Figure]: 
    """Generate RFM analysis visualizations"""
        
    if rfm_final is None:
        raise ValueError("Analysis must be run before generating visualizations")
        
    figures = {}
        
    print("Creating RFM distribution plots...")
    figures['distributions'] = plot_rfm_distributions(rfm_final)
        
    print("Creating segment analysis plots...")
    figures['segments'] = plot_segment_analysis(segment_summary)

    return figures


def export_results_phase_3(rfm_final: pd.DataFrame, segment_summary: pd.DataFrame,
                   filepath: str = "../results/phase_3/rfm_analysis_results.xlsx",
                   export_charts: bool = True, chart_dir: str = "../figures/rfm_charts"):
    """Export RFM analysis results and visualizations"""
    
    figures = None
    if export_charts:
        figures = generate_visualizations(rfm_final, segment_summary)
        os.makedirs(chart_dir, exist_ok=True)
        for name, fig in figures.items():
            fig.savefig(os.path.join(chart_dir, f"{name}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Chart saved: {chart_dir}/{name}.png")
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        rfm_final.to_excel(writer, sheet_name='RFM_Analysis', index=False)
        segment_summary.to_excel(writer, sheet_name='Segment_Summary')
    
    print(f"Results exported to {filepath}")

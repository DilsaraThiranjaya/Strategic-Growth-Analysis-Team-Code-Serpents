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

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from typing import Tuple
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

class RFMScorer:
    ## Handles RFM scoring and customer segmentation.
    
    def __init__(self):
        self.segment_map = self._create_segment_map()
    
    def assign_rfm_scores(self, rfm_data: pd.DataFrame):
        ## Assign RFM scores (1-5) to each customer based on quintiles.
        
        rfm_scored_data = rfm_data.copy()
        
        # Assign scores using quintiles
        # For Recency: lower values (recent) get higher scores
        rfm_scored_data['R_Score'] = pd.qcut(rfm_scored_data['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        
        # For Frequency: higher values (frequent) get higher scores
        rfm_scored_data['F_Score'] = pd.qcut(rfm_scored_data['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        
        # For Monetary: higher values (spend more) get higher scores
        rfm_scored_data['M_Score'] = pd.qcut(rfm_scored_data['Monetary'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        
        # Convert scores to integers
        rfm_scored_data['R_Score'] = rfm_scored_data['R_Score'].astype(int)
        rfm_scored_data['F_Score'] = rfm_scored_data['F_Score'].astype(int)
        rfm_scored_data['M_Score'] = rfm_scored_data['M_Score'].astype(int)
        
        # Create combined RFM segment
        rfm_scored_data['RFM_Segment'] = (
            rfm_scored_data['R_Score'].astype(str) +
            rfm_scored_data['F_Score'].astype(str) +
            rfm_scored_data['M_Score'].astype(str)
        )
        
        return rfm_scored_data
    
    def _create_segment_map(self):
        ## Create a mapping from RFM scores to descriptive segment names.
        
        segment_map = {}

        for r in range(1, 6): # R_Score 1-5
            for f in range(1, 6): # F_Score 1-5
                for m in range(1, 6): # M_Score 1-5
                    key = f"{r}{f}{m}"

                    if r >= 4 and f >= 4 and m >= 4: # Champions: High RFM scores (4-5 range)
                        segment_map[key] = "Champions"
                    elif r >= 3 and f >= 4: # Loyal Customers: High RF, varying M
                        segment_map[key] = "Loyal Customers"
                    elif r >= 4 and f <= 2: # Potential Loyalists: Recent customers with average frequency
                        segment_map[key] = "Potential Loyalists"
                    elif r == 5 and f <= 2: # New Customers: High recency, low frequency
                        segment_map[key] = "New Customers"
                    elif r <= 2 and f >= 3: # At-Risk Customers: Used to be frequent buyers
                        segment_map[key] = "At-Risk Customers"
                    elif r <= 2 and f <= 2: # Hibernating: Low recency, low frequency
                        segment_map[key] = "Hibernating"
                    
        return segment_map

    def assign_segment_labels(self, rfm_scored_data: pd.DataFrame):
        ## Assign descriptive segment labels to customers.
        
        rfm_final = rfm_scored_data.copy()
        
        # Map RFM segments to descriptive names
        rfm_final['Segment'] = rfm_final['RFM_Segment'].map(self.segment_map)
        
        return rfm_final
    
    def get_segment_summary(self, rfm_final: pd.DataFrame):
        ## Generate summary statistics for each customer segment.
        
        # Customer ID → count → How many customers in that segment.
        # Recency → mean, min, max → On average how recently they bought, and range.
        # Frequency → mean, min, max → On average how often they buy, and range.
        # Monetary → mean, min, max → On average how much they spend, and range.
        segment_summary = rfm_final.groupby('Segment').agg({
            'Customer ID': 'count',
            'Recency': ['mean', 'median'],
            'Frequency': ['mean', 'median'],
            'Monetary': ['mean', 'median', 'sum']
        }).round(2)
    
        # Flatten column names
        segment_summary.columns = [
            'Customer_Count',
            'Avg_Recency', 'Median_Recency',
            'Avg_Frequency', 'Median_Frequency',
            'Avg_Monetary', 'Median_Monetary', 'Total_Revenue'
        ]
        
        # Calculate percentage of customers
        segment_summary['Customer_Percentage'] = (
            segment_summary['Customer_Count'] / segment_summary['Customer_Count'].sum() * 100
        ).round(2)
        
        # Calculate revenue percentage
        segment_summary['Revenue_Percentage'] = (
            segment_summary['Total_Revenue'] / segment_summary['Total_Revenue'].sum() * 100
        ).round(2)
        
        return segment_summary.sort_values('Customer_Count', ascending=False)
    
class RFMVisualizer:
    ## Creates visualizations for RFM analysis results.
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")

    def plot_rfm_distributions(self, rfm_data: pd.DataFrame):
        ## Create distribution plots for RFM metrics.
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('RFM Metrics Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Recency distribution
        axes[0, 0].hist(rfm_data['Recency'], bins=50, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Recency Distribution (Days Since Last Purchase)')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel('Number of Customers')
        axes[0, 0].axvline(rfm_data['Recency'].mean(), color='red', linestyle='--', label=f'Mean: {rfm_data["Recency"].mean():.1f}')
        axes[0, 0].legend()
        
        # Frequency distribution
        axes[0, 1].hist(rfm_data['Frequency'], bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Frequency Distribution (Number of Transactions)')
        axes[0, 1].set_xlabel('Number of Transactions')
        axes[0, 1].set_ylabel('Number of Customers')
        axes[0, 1].axvline(rfm_data['Frequency'].mean(), color='red', linestyle='--', label=f'Mean: {rfm_data["Frequency"].mean():.1f}')
        axes[0, 1].legend()
        
        # Monetary distribution (log scale due to skewness)
        axes[1, 0].hist(np.log10(rfm_data['Monetary']), bins=50, color='gold', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Monetary Distribution (Log10 Scale)')
        axes[1, 0].set_xlabel('Log10(Total Spent)')
        axes[1, 0].set_ylabel('Number of Customers')
        axes[1, 0].axvline(np.log10(rfm_data['Monetary'].mean()), color='red', linestyle='--', label=f'Mean: £{rfm_data["Monetary"].mean():.0f}')
        axes[1, 0].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_segment_analysis(self, segment_summary: pd.DataFrame):
        ## Create comprehensive segment analysis visualizations.
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Customer Segment Analysis', fontsize=16, fontweight='bold')
        
        # Customer count by segment
        segment_summary.plot(kind='bar', y='Customer_Count', ax=axes[0, 0], color='steelblue', alpha=0.8)
        axes[0, 0].set_title('Number of Customers by Segment')
        axes[0, 0].set_xlabel('Customer Segment')
        axes[0, 0].set_ylabel('Number of Customers')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend().remove()
        
        # Revenue percentage by segment
        axes[0, 1].pie(segment_summary['Revenue_Percentage'], labels=segment_summary.index, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Revenue Distribution by Segment')
        
        # Average monetary value by segment
        segment_summary.plot(kind='bar', y='Avg_Monetary', ax=axes[1, 0], color='darkorange', alpha=0.8)
        axes[1, 0].set_title('Average Monetary Value by Segment')
        axes[1, 0].set_xlabel('Customer Segment')
        axes[1, 0].set_ylabel('Average Spent (£)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend().remove()
        
        plt.tight_layout()
        return fig

class RFMAnalyzer:
    ## Main class that orchestrates the complete RFM analysis workflow.
    
    def __init__(self, data: pd.DataFrame):
        self.data_processor = RFMDataProcessor(data)
        self.scorer = RFMScorer()
        self.visualizer = RFMVisualizer()
        
        # Analysis results
        self.rfm_data = None
        self.rfm_scored = None
        self.rfm_final = None
        self.segment_summary = None
        self.summary_stats = None
        
    def run_complete_analysis(self):
        ## Execute the complete RFM analysis workflow.
        
        print("🔄 Starting RFM Analysis...")
        
        # Step 1: Calculate RFM metrics
        print("📊 Calculating RFM metrics...")
        self.rfm_data = self.data_processor.calculate_rfm()
        
        # Step 2: Generate summary statistics
        print("📈 Generating summary statistics...")
        self.summary_stats = self.data_processor.get_customer_summary_stats(self.rfm_data)
        
        # Step 3: Assign RFM scores
        print("🏷️ Assigning RFM scores...")
        self.rfm_scored = self.scorer.assign_rfm_scores(self.rfm_data)
        
        # Step 4: Assign segment labels
        print("🎯 Assigning segment labels...")
        self.rfm_final = self.scorer.assign_segment_labels(self.rfm_scored)
        
        # Step 5: Generate segment summary
        print("📋 Generating segment summary...")
        self.segment_summary = self.scorer.get_segment_summary(self.rfm_final)
        
        print("✅ RFM Analysis Complete!")
        
        return {
            'rfm_data': self.rfm_data,
            'rfm_scored': self.rfm_scored,
            'rfm_final': self.rfm_final,
            'segment_summary': self.segment_summary,
            'summary_stats': self.summary_stats
        }
    
    def generate_visualizations(self):
        ## Generate all RFM analysis visualizations.
        
        if self.rfm_data is None:
            raise ValueError("Analysis must be run before generating visualizations")
        
        figures = {}
        
        print("📊 Creating RFM distribution plots...")
        figures['distributions'] = self.visualizer.plot_rfm_distributions(self.rfm_data)
        
        print("🎯 Creating segment analysis plots...")
        figures['segments'] = self.visualizer.plot_segment_analysis(self.segment_summary)

        return figures
    
    def export_results(self, filepath: str = "../report/rfm_analysis_results.xlsx", export_charts: bool = True, chart_dir: str = "../figures/rfm_charts"):
    ## Export analysis results to Excel file.

        if self.rfm_final is None:
            raise ValueError("Analysis must be run before exporting results")
    
        # Create charts if requested
        figures = None
        if export_charts:
            figures = self.generate_visualizations()
            os.makedirs(chart_dir, exist_ok=True)
            for name, fig in figures.items():
                fig_path = os.path.join(chart_dir, f"{name}.png")
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                print(f"📁 Chart saved: {fig_path}")
        
        # Export data to Excel
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            self.rfm_final.to_excel(writer, sheet_name='RFM_Analysis', index=False)
            self.segment_summary.to_excel(writer, sheet_name='Segment_Summary')
        
        print(f"✅ Results exported to {filepath}")

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
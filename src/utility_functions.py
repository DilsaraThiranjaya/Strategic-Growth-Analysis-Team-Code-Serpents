import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# Utility Functions
# ================================================================

def load_and_validate_data(filepath: str):
        ## Utility function to load and validate data for RFM analysis.
        
        try:
            data = pd.read_csv(filepath, parse_dates=['InvoiceDate'])
            print(f"✅ Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
            
            # Define required columns
            required_columns = [
                'Invoice', 'StockCode', 'Description', 'Quantity',
                'InvoiceDate', 'Price', 'Customer ID', 'Country',
                'TotalPrice', 'Year', 'Month', 'DayOfWeek', 'HourOfDay'
            ]
            
            # Check missing columns
            missing = [col for col in required_columns if col not in data.columns]
            if missing:
                raise ValueError(f"❌ Missing required columns: {', '.join(missing)}")
            
            print("✅ All required columns are present")
            return data

        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
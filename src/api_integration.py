"""
Data Enrichment via API Integration
===================================

A comprehensive function-based system for acquiring and integrating external data
through web APIs to enhance primary dataset with currency conversion capabilities.

Team: Code Serpents
Team Member: K. Lahiru Chanaka
Project: Strategic Growth Analysis for UK E-Commerce Retailer
Phase: 5 - Data Enrichment via API Integration
"""

import requests
import time
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# API Configuration and Helper Functions
# ================================================================

# API Configuration

API_BASE_URL = 'https://v6.exchangerate-api.com/v6/679cea4bfe60efe64fe59613/latest/'
BASE_CURRENCY = 'GBP'
TARGET_CURRENCIES = ['USD', 'EUR']

def fetch_exchange_rates(base_currency='GBP', retries=3, delay=1):
    url = f"{API_BASE_URL}{base_currency}"

    for attempt in range(retries):
        try:
            print(f"Fetching exchange rates (attempt {attempt + 1}/{retries})...")

            response = requests.get(url)
            response.raise_for_status()

            data = response.json()

            # Validate response structure
            if 'rates' in data and 'date' in data:
                print(f"Successfully fetched rates for {data['data']}")
                return data
            else:
                print("Unexpected API response format.")
                return None

        except requests.exceptions.RequestException as e:
            print(f"API request failed (attempt {attempt + 1}): {str(e)}")

            if attempt < retries - 1:
                print(f"Waiting {delay} seconds before retrying...")
                time.sleep(delay)
            else:
                print("Max retries exceeded. Exiting...")
                return None

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {str(e)}")
            return None

    return None

def convert_currency(amount, from_currency, to_currency, exchange_rates):
    try:
        if from_currency == to_currency:
            return amount

        if to_currency in exchange_rates['rates']:
            converted_amount = amount * exchange_rates['rates'][to_currency]
            return round(converted_amount, 2)
        else:
            print(f"Exchange rate for {to_currency} not found in the API response.")
            return None

    except (KeyError, TypeError, ValueError) as e:
        print(f"Currency conversion error: {str(e)}")
        return None
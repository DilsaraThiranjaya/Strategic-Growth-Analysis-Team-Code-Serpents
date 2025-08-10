# Strategic Growth Analysis for Unique Gifts Ltd.  
**Data Science Capstone Project | ITS 2122: Python for Data Science & AI**  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-brightgreen)

## 📌 Project Overview
This project delivers data-driven strategic insights for **Unique Gifts Ltd.**, a UK-based e-commerce retailer, by analyzing 2 years of transactional data (2009-2011). The analysis covers:
- Sales performance and seasonality
- Product portfolio optimization  
- Customer segmentation (RFM model)  
- Geographic revenue distribution  
- Wholesaler vs. retail customer behavior  

**Business Impact**: Enables data-backed decisions for marketing, inventory management, and customer retention strategies.

---

## 🗂 Project Structure
```plaintext
strategic-growth-analysis/
├── data/
│   ├── online_retail.csv             # Raw dataset (1M+ records)
│   └── online_retail_clean.csv       # Processed data (Post Phase 1)
│
├── notebooks/
│   ├── 1_data_cleaning.ipynb         # Phase 1: Data sanitation
│   ├── 2_eda.ipynb                   # Phase 2: Exploratory analysis
│   ├── 3_rfm_segmentation.ipynb      # Phase 3: Customer modeling
│   └── 4_api_integration.ipynb       # Phase 5: Currency conversion
│
├── report/
│   ├── strategic_insights.pdf        # 3000-word business report
│   └── presentation_slides.pptx      # Executive summary deck
│
├── src/
│   ├── data_processor.py             # Modular cleaning pipeline
│   ├── visualizer.py                 # Custom plotting functions
│   └── rfm_engine.py                 # Segmentation logic
│
├── figures/                          # Exported visualizations
│   ├── temporal/
│   │   ├── monthly_sales_trend.png
│   │   └── hourly_sales_heatmap.png
│   ├── geographic/
│   │   ├── country_revenue.png
│   │   └── uk_vs_international.png
│   └── products/
│       ├── top10_by_quantity.png
│       └── top10_by_revenue.png
│
├── requirements.txt                  # Dependencies
└── README.md                         # This document"# Strategic-Growth-Analysis-Team-Code-Serpents" 

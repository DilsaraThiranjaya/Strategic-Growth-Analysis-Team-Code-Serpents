# 🚀 Strategic Growth Analysis – Team Code Serpents

### A Data Science & AI Capstone Project for **ITS 2122: Python for Data Science & AI**

---

## 📌 Project Overview

This project simulates the role of a **data science consulting team** engaged by the executive board of **Unique Gifts Ltd.**, a UK-based e-commerce retailer specializing in unique giftware.

The company has enjoyed steady organic growth but has relied heavily on intuition for decision-making. With access to **two years of transactional data**, our mission was to transform raw data into **actionable insights** and provide a **data-driven strategic roadmap** for sustainable growth.

---

## 🎯 Objectives

The project addresses five core business challenges:

1. **Sales Performance & Seasonality**

   * Identify monthly/yearly sales trends.
   * Detect seasonal patterns & peak shopping periods for promotions and inventory planning.

2. **Product Portfolio Optimization**

   * Distinguish top-performing vs. underperforming products.
   * Differentiate “bread-and-butter” (high-volume) vs. “cash cow” (high-value) products.

3. **Geographic Footprint**

   * Analyze domestic vs. international sales.
   * Pinpoint top countries contributing to revenue & explore international opportunities.

4. **Customer Segmentation (RFM Analysis)**

   * Apply **Recency, Frequency, Monetary (RFM)** modeling.
   * Classify customers into actionable segments (Champions, Loyal, At-Risk, etc.).

5. **Wholesaler vs. Retail Analysis**

   * Compare purchasing behavior of wholesalers vs. retail customers.
   * Provide strategic recommendations tailored to each group.

---

## 🛠️ Tools & Technologies

* **Python 3.x**

* **Libraries**:

  * `pandas` – Data cleaning & preprocessing
  * `numpy` – Numerical operations
  * `matplotlib`, `seaborn` – Data visualization
  * `requests` – API integration (currency conversion)

* **Dataset**: [Online Retail II – UCI Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

---

## 📂 Repository Structure

```plaintext
Strategic-Growth-Analysis-Team-Code-Serpents/
├── data/
│   ├── online_retail.csv             # Raw dataset (1M+ records)
│   └── online_retail_clean.csv       # Processed data (Post Phase 1)
│
├── notebooks/
│   └── retail_analysis.ipynb         # Main Jupyter Notebook
│
├── report/
│   └── Strategic_Insights_Report.pdf # Final business report
│
├── src/                              # Support functions
│   ├── data_cleaning.py              # Phase 1
│   ├── eda_analysis.py               # Phase 2
│   ├── rfm_segmentation.py           # Phase 3
│   ├── strategic_recommendations.py  # Phase 4
│   └── api_integration.py            # Phase 5
│
├── figures/                          # Exported visualizations
│   ├── currency_conversions/
│   ├── rfm_charts/
│   ├── strategic_rec_charts/
│   ├── temporal/
│   ├── geographic/
│   └── products/
│
├── results/                          # Exported csv files
│   ├── phase_3/
│   └── phase_5/
│
├── requirements.txt                  # Dependencies
└── README.md                         # Project documentation (this file)
```

---

## 📊 Analytical Approach (Phased Workflow)

1. **Phase 1: Data Cleaning & Preprocessing**

   * Removed duplicates, cancellations, and invalid entries.
   * Handled missing Customer IDs.
   * Created new features (`TotalPrice`, Year, Month, DayOfWeek, Hour).

2. **Phase 2: Exploratory Data Analysis (EDA)**

   * Temporal trends (monthly, daily, hourly).
   * Geographic revenue distribution.
   * Top 10 products by sales volume & revenue.

3. **Phase 3: Advanced Analytics – RFM Segmentation**

   * Calculated Recency, Frequency, Monetary metrics.
   * Assigned quintile-based scores (1–5).
   * Created descriptive customer segments.

4. **Phase 4: Strategic Insights**

   * Compared wholesalers vs. retail customers.
   * Provided data-driven recommendations for customer retention & targeted marketing.

5. **Phase 5: API Integration**

   * Integrated a **currency conversion API**.
   * Converted top 100 transactions into **USD** & **EUR** for international reporting.

---

## 📈 Key Deliverables

* **📑 Strategic Insights Report (PDF)** – Business-focused report with recommendations.
* **📓 Jupyter Notebook (Technical Appendix)** – Complete code with documentation.
* **📊 Visualizations** – Clear and professional charts to support insights.

---

## ⚡ Getting Started

### 1. Clone Repository

```bash
git clone https://github.com/DilsaraThiranjaya/Strategic-Growth-Analysis-Team-Code-Serpents.git
cd Strategic-Growth-Analysis-Team-Code-Serpents
```

### 2. Install Dependencies

It’s recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Run Jupyter Notebook

```bash
notebooks/retail_analysis.ipynb
```

---

## 🤝 Team Code Serpents

* 👤 G. A. Dilsara Thiranjaya
* 👤 A. M. Supun Madhuranga
* 👤 S. M. L.Lakshan jayawardhana
* 👤 K. D. Vihanga Heshan Bandara
* 👤 K. Lahiru Chanaka

---

## 📜 License

This project is developed as part of **ITS 2122: Python for Data Science & AI** coursework.
For academic purposes only. Not intended for commercial use.

---

## 🌟 Acknowledgements

* Dataset: [UCI Machine Learning Repository – Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
* IJSE – Institute of Software Engineering (Sri Lanka)
* Lecturers & mentors for guidance

---


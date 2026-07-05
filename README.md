# 📊 Interim Report: Quantitative Financial Analysis & Time Series Forecasting

**Submission Date:** July 5, 2026  
**Repository Link:** [portfolio-optimization](https://github.com/meskerem-sys/portfolio-optimization)  
**Assets Evaluated:** TSLA, SPY, BND

---

## 1. Summary of Data Extraction and Cleaning Steps
* **Scope:** 11 years of historical daily asset pricing data (2015 to 2026) was extracted for TSLA, SPY, and BND.
* **Cleaning Pipeline:** Chronologically sorted all entries by a standardized datetime index, handled missing values, and isolated historical features.
* **Leakage Prevention:** Sequentially split the dataset into historical training (2015–2024) and out-of-sample forward validation (2025–2026) subsets.

---

## 2. Stationarity Test Results & Interpretation (ADF Test)
Based on the exploratory data analysis, raw prices are strictly non-stationary, but their daily returns are highly stationary:

* **TSLA:** Raw Price p-value = **0.7217** (Non-Stationary) ❌ | Daily Returns p-value = **0.0000** (Stationary)  
* **SPY:** Raw Price p-value = **0.9968** (Non-Stationary) ❌ | Daily Returns p-value = **4.5881e-30** (Stationary)  
* **BND:** Raw Price p-value = **0.6740** (Non-Stationary) ❌ | Daily Returns p-value = **7.4429e-28** (Stationary)  

**Interpretation:** All three raw price series contain strong time-dependent trends. Applying first-order differencing (daily returns) successfully eliminates the trend, achieving the stationarity required for stable mathematical modeling.

---

## 3. Volatility Analysis & Risk Metrics
From the risk and reward summary metrics:

* **Asset Focus: TSLA**
  * Annualized Sharpe Ratio: **0.7895**
  * 95% Daily Value at Risk (VaR): **-5.18%** *(On 95% of days, daily losses will not exceed 5.18%)*
* **Asset Focus: SPY**
  * Annualized Sharpe Ratio: **0.7095**
  * 95% Daily Value at Risk (VaR): **-1.67%** *(On 95% of days, daily losses will not exceed 1.67%)*
* **Asset Focus: BND**
  * Annualized Sharpe Ratio: **-0.0095**
  * 95% Daily Value at Risk (VaR): **-0.48%** *(On 95% of days, daily losses will not exceed 0.48%)*

---

## 4. Task 2: Model Implementation Progress
* **Baseline Framework:** An **ARIMA(0, 1, 0)** configuration acts as our classical statistical benchmark, interpreting the price movements as a Random Walk.
* **Deep Learning Framework:** A deep Multi-Layer Perceptron (MLP) mapping a rolling historical window over multiple hidden layers has been initiated for forecasting out-of-sample trends.
# Semester Project–III Logbook  
*TY CSE (Data Science)*  
*Academic Year: 2025–26*

---

## 1. Introduction  
*Date: 18/08/2025 to 30/08/2025*

### Problem Statement  
Stock price prediction is a challenging task due to the volatile, non-linear, and dynamic nature of financial markets.  
Traditional methods rely on basic statistical analysis and human judgment, which may not effectively capture complex patterns in historical and real-time stock market data.  
Accurate prediction of stock prices is crucial for investors, traders, and financial institutions to minimize risks and maximize returns.  
There is a need for an automated, reliable, and data-driven machine learning system capable of predicting stock prices using historical as well as live market data.

### Objectives  
- To collect historical and live stock market data using the yfinance library.  
- To preprocess and clean stock price data for analysis.  
- To perform exploratory data analysis (EDA) and feature engineering.  
- To apply machine learning algorithms for stock price prediction.  
- To evaluate model performance using appropriate regression metrics.  
- To develop a real-time prediction system for decision support.

### Application of Project  
- Assists investors in making informed buy/sell decisions.  
- Useful for financial analysts and portfolio managers.  
- Can be integrated into trading platforms as a decision-support tool.  
- Serves as an educational tool for understanding stock market trends using machine learning.

---

## 2. Literature Survey  
*Date: 01/09/2025 to 13/09/2025*

### Background  
Machine learning techniques are widely used in financial forecasting and time-series analysis.  
Stock market prediction models generally utilize historical price data, technical indicators, and volume information.  
With the availability of real-time data APIs, live prediction systems have become increasingly feasible.

### Existing Systems  
Based on various research papers and studies:

- Linear Regression, Decision Tree, Random Forest, and Support Vector Regression (SVR) ,GRU are commonly used models.  
- Deep learning models such as LSTM are effective for sequential time-series data.  
- Technical indicators like Moving Average, RSI, and MACD improve prediction accuracy.  
- Reported prediction accuracy ranges between 70% and 90% depending on the dataset and model used.

### Limitations of Existing Systems  
- High volatility and noise in stock market data.  
- Overfitting due to complex patterns.  
- Limited generalization on unseen data.  
- Real-time prediction requires efficient data processing.

---

## 3. Methodology  
*Date: 15/09/2025 to 27/09/2025*

### Hardware Requirements  
- Processor: Intel Core i5 or above  
- RAM: 8GB minimum  
- Storage: 20GB  
- GPU: Optional  

### Software Requirements  
- Python 3.x  
- NumPy, Pandas  
- Scikit-learn  
- Matplotlib, Seaborn  
- yfinance  
- TensorFlow / Keras (optional)  
- Jupyter Notebook / Google Colab  
- Power BI / Tableau (for visualization)

---

### System Design  

Live Data Fetching (yfinance) → Data Preprocessing → Feature Engineering → Model Training → Model Evaluation → Real-Time Prediction

---

### Dataset Used  

- *Data Source:* Yahoo Finance (via yfinance Python library)  
- *Nature of Data:*  
  - Historical stock price data for training and testing  
  - Live/latest stock data for real-time prediction  
- *Base Features:*  
  - Date  
  - Open  
  - High  
  - Low  
  - Close  
  - Adjusted Close  
  - Volume  
- *Stocks Used:* Selected NSE / BSE / NASDAQ listed companies  

---

### Feature Engineering  

To enhance model performance, additional features were engineered from raw yfinance data:

- *Price-Based Features:*  
  - Daily Returns  
  - Open–Close Price Difference  
  - High–Low Price Range  

- *Trend Indicators:*  
  - Simple Moving Average (SMA – 10, 20 days)  
  - Exponential Moving Average (EMA – 10 days)  

- *Momentum Indicators:*  
  - Relative Strength Index (RSI)  
  - Moving Average Convergence Divergence (MACD)  

- *Volatility Indicators:*  
  - Rolling Standard Deviation  
  - Bollinger Bands  

- *Volume-Based Features:*  
  - Volume Moving Average  
  - Volume Percentage Change  

- *Lag Features:*  
  - Previous day closing prices (Lag-1, Lag-2)  

- *Time-Based Features:*  
  - Day  
  - Month  
  - Day of the Week  

These features help capture market trends, momentum, volatility, and temporal dependencies.

---

### Algorithms Used  

- GRU 
- Random Forest Regressor  

---

## 4. Implementation Details  
*Date: 29/09/2025 to 18/10/2025*

### Module 1: Live Data Collection and Preprocessing  
- Fetching historical and live stock market data using yfinance.  
- Handling missing values and non-trading days.  
- Normalization and scaling of numerical features.  
- Splitting data into training and testing datasets.

### Module 2: Feature Engineering and EDA  
- Computing technical indicators using rolling windows.  
- Visualizing stock trends, volume, and indicators.  
- Correlation analysis between engineered features.

### Module 3: Model Training and Prediction  
- Training regression models on historical data.  
- Hyperparameter tuning for improved accuracy.  
- Predicting stock prices using live market data.

---

## 5. Results  
*Date: 27/10/2025 to 04/11/2025*

### Results Obtained  
- Linear Regression provided a strong baseline model.  
- Decision Tree showed moderate accuracy with some overfitting.  
- Random Forest Regressor achieved the best performance with minimal error.  
- GRU is suitable.  

### Performance Metrics  
- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R² Score  

### Model Evaluation (Graphs)  
- Actual vs Predicted price plots  
- Error comparison bar charts  
- Trend prediction visualizations  

---

## 6. Conclusion  

The project successfully implemented a stock price prediction system using machine learning and *live stock market data fetched via yfinance*.  
Feature engineering significantly improved prediction accuracy.  
Among all models, the Random Forest Regressor performed best on both historical and live data.  
The system demonstrates practical applicability for real-world financial analysis and decision-making.  
Future enhancements may include deep learning models and automated trading alerts.

---

## 7. References (IEEE Format)

1. Fama, E. F., “Efficient Capital Markets: A Review of Theory and Empirical Work,” Journal of Finance, 1970.  
2. Patel, J. et al., “Predicting Stock Market Index Using Machine Learning,” Expert Systems with Applications, 2015.  
3. Yahoo Finance, “Stock Market Data,” Available: https://finance.yahoo.com/  
4. Ran Aroussi, “yfinance: Yahoo! Finance market data downloader,” GitHub Repository, 2020.  

---
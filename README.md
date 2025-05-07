
# 📊 Customer Churn Prediction

This project analyzes customer behavior and predicts churn using a machine learning model based on [telecom service data]( https://www.kaggle.com/datasets/hassanelfattmi/why-do-customers-leave-can-you-spot-the-churners), It includes exploratory data analysis (EDA), preprocessing, and training a Random Forest classifier, with a focus on contract type, satisfaction, revenue, and customer demographics.

---

## 🚀 Project Overview

Customer churn prediction is critical for customer retention strategies. This project aims to:

- Merge and clean multiple datasets (Customer Info, Online Services, Payments, etc., excluding satisfactory parameter for bigger challenge)
- Explore patterns affecting churn (contracts, satisfaction, dependents, etc.)
- Build a machine learning model to predict churn
- Evaluate the model's performance and interpret insights

---

## 🧱 Project Structure

```
📁 data/
   ├── Customer_Info.csv
   ├── Online_Services.csv
   ├── Payment_Info.csv
   ├── Service_Options.csv
   └── Status_Analysis.csv

📄 churn_analysis.py     # Data loading, preprocessing, model training
📄 acc.csv               # Model predictions and classification accuracy
📄 Churn_EDA.pbix        # Explanatory Data analysis in PowerBI
```

---

## 🛠 Features

- **EDA Dashboard** (Power BI) with insights on:
  - Contract type vs churn
  - Age and satisfaction trends
  - Revenue distribution
  - Dependents and churn likelihood
- **Data Pipeline** with:
  - Merging & preprocessing of multiple sources
  - Categorical encoding & normalization
  - Churn classification using Random Forest

---

## 📈 Model Performance

- **Accuracy**: ~80.9% correct predictions on test split
- Balanced performance with insights into misclassifications (classified vs unclassified)
- Churn prediction visualized in pie charts and revenue/satisfaction plots

---

## 🔍 How to Use

1. **Clone the repo**:
   ```bash
   git clone https://github.com/JakubRybka/churn-prediction.git
   cd churn-prediction
   ```

2. **Place CSV data** into the `data/` folder as listed above.

3. **Run the script**:
   ```bash
   python churn_analysis.py
   ```

4. **View Outputs**:
   - `acc.csv`: Contains predicted vs actual churn and classification result.
   - `eda_dashboard.png`: Visual representation of patterns (optional, provided).

---

## 🔧 Requirements

- Python 3.7+
- pandas, numpy, scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ✅ Future Improvements

- Implement feature importance analysis
- Use precision/recall and ROC-AUC metrics
- Try hyperparameter tuning (GridSearchCV)


---

## 📄 License

This project is open-source and available under the MIT License.

---

## 👤 Authors

- [Jakub Rybka](https://github.com/JakubRybka)
- [Zofia Szczepaniak](https://github.com/ZofiaSzczepaniak)

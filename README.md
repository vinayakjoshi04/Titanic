# 🚢 Titanic Survivor Prediction

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A machine learning project that predicts passenger survival on the Titanic using various classification algorithms. This project demonstrates comprehensive data preprocessing, feature engineering, and model comparison techniques.

## 🎯 Project Overview

This project analyzes the famous Titanic disaster dataset to predict passenger survival based on features such as age, gender, passenger class, family relationships, and embarkation details. The implementation includes extensive data preprocessing, feature engineering, and comparison of multiple machine learning models.

**Dataset Source:** [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)

## 📊 Results Summary

| Model | Accuracy | Notes |
|-------|----------|-------|
| **Logistic Regression** | **91%** | Best performing model |
| Random Forest Classifier | 81% | Good ensemble performance |
| XGBoost Classifier | 82% | Competitive gradient boosting |

## 📁 Project Structure

```
titanic-survivor-prediction/
├── data/
│   ├── train.csv              # Training dataset
│   ├── test.csv               # Test dataset
│   └── test_Survived.csv      # Ground truth labels
├── notebooks/
│   └── titanic_analysis.ipynb # Jupyter notebook with analysis
├── README.md                 # Project documentation
```

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/titanic-survivor-prediction.git
cd titanic-survivor-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🧹 Data Preprocessing & Feature Engineering

### Data Cleaning
- **Missing Age values**: Filled with median age
- **Missing Embarked values**: Dropped the missing 0.2% values
- **Cabin feature**: Removed due to excessive missing values (>75%)

### Feature Engineering
- **Title Extraction**: Extracted titles from passenger names (Mr., Mrs., Miss, etc.)
- **Family Size**: Created `FamilySize = SibSp + Parch + 1`
- **Is Alone**: Binary feature indicating solo travelers (`FamilySize == 1`)
- **Name Length**: Character count of passenger names
- **Age Groups**: Binned ages into categories (Child, Adult, Senior)

### Data Transformation
- **Categorical Encoding**: Used LabelEncoder for categorical variables
- **Feature Scaling**: Applied StandardScaler for Logistic Regression
- **Train-Test Split**: 80-20 split with stratification

## 🤖 Model Implementation

### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_scaled, y)
```

### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100, random_state=42)
RFC.fit(x, y)
```

### XGBoost
```python
from xgboost import XGBClassifier
XGB = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
XGB.fit(x,y)
```

## 📈 Model Evaluation

### Metrics Used
- **Accuracy Score**: Primary evaluation metric
- **Confusion Matrix**: For detailed performance analysis
- **Classification Report**: Precision, recall, and F1-scores

### Evaluation Code
```python
from sklearn.metrics import accuracy_score

# Make predictions
y_test_predictLR = LR.predict(X_test_cleaned_Scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test_predictLR , y_test_cleaned)
print(f"Accuracy: {accuracy:.2%}")
```

## 🔍 Key Insights

1. **Feature Importance**: Gender (Sex) was the most predictive feature
2. **Passenger Class**: Higher class passengers had better survival rates
3. **Age Factor**: Children and women had higher survival probabilities
4. **Family Size**: Small families (2-4 members) had better survival rates than solo travelers or large families


## 🎯 Future Improvements

- [ ] Implement cross-validation for more robust evaluation
- [ ] Add hyperparameter tuning using GridSearchCV
- [ ] Explore ensemble methods (Voting, Stacking)
- [ ] Add feature selection techniques
- [ ] Implement deep learning approaches
- [ ] Create interactive web interface

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 👨‍💻 Author

**Vinayak Joshi**
- GitHub: [@Vinayak Joshi](https://github.com/vinayakjoshi04)
- LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/vinayak-joshi-99521528b/)

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the Titanic dataset
- The open-source community for the amazing libraries used in this project
- Fellow data scientists and researchers for inspiration and guidance

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

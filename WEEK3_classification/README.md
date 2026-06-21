# Customer Intelligence System using Machine Learning

## Project Overview

This project develops an end-to-end Customer Intelligence System using machine learning techniques to analyze customer behavior, identify meaningful customer segments, and predict campaign response. The system integrates customer segmentation, predictive analytics, feature importance analysis, and business recommendations to support data-driven marketing decisions.

Using the Customer Personality Analysis dataset, the project combines clustering algorithms and supervised machine learning models to help businesses understand customer value, improve targeting strategies, and optimize marketing investments.

---

# Business Problem

Organizations often invest substantial resources in marketing campaigns without knowing which customers are most likely to respond. As a result, marketing budgets may be spent inefficiently and customer engagement opportunities may be missed.

The objective of this project is to:

* Segment customers into meaningful groups.
* Predict campaign response behavior.
* Identify high-value customers.
* Improve marketing effectiveness.
* Support data-driven business decisions.

---

# Dataset

**Dataset:** Customer Personality Analysis

The dataset contains customer demographics, spending behavior, purchasing activity, and campaign response information.

### Customer Information

* ID
* Year_Birth
* Education
* Marital_Status
* Income
* Kidhome
* Teenhome
* Dt_Customer
* Recency
* Complain

### Product Spending

* MntWines
* MntFruits
* MntMeatProducts
* MntFishProducts
* MntSweetProducts
* MntGoldProds

### Marketing Campaign Information

* AcceptedCmp1
* AcceptedCmp2
* AcceptedCmp3
* AcceptedCmp4
* AcceptedCmp5
* Response

### Customer Engagement

* NumWebPurchases
* NumCatalogPurchases
* NumStorePurchases
* NumWebVisitsMonth
* NumDealsPurchases

---

# Project Objectives

* Perform customer behavior analysis.
* Engineer meaningful business features.
* Segment customers into actionable groups.
* Predict campaign response using machine learning.
* Compare multiple classification algorithms.
* Evaluate ensemble learning models.
* Identify important business drivers.
* Generate actionable business recommendations.

---

# Project Workflow

## 1. Data Understanding

Initial dataset exploration included:

* Dataset shape
* Feature inspection
* Data types
* Missing value analysis
* Duplicate record detection

### Key Findings

* Dataset contains 2,240 customer records.
* Missing values were found in the Income feature.
* Customer demographics, spending patterns, and campaign history provide rich business insights.

---

## 2. Data Cleaning

Performed:

* Missing value treatment
* Outlier handling
* Duplicate verification
* Date conversion
* Data consistency validation

### Key Findings

* Missing Income values were imputed using median values.
* Unrealistic age records were removed.
* Extreme income outliers were treated.
* Data quality improved significantly after preprocessing.

---

## 3. Feature Engineering

Several business-focused features were created:

### Age

Age = Current Year − Year_Birth

### Total Children

Total_Children = Kidhome + Teenhome

### Total Spending

Total_Spending =
MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts + MntGoldProds

### Total Purchases

Total_Purchases =
NumWebPurchases + NumCatalogPurchases + NumStorePurchases

### Customer Tenure

Customer_Tenure =
Current Date − Dt_Customer

### Business Value

These engineered features provide a more comprehensive representation of customer value, engagement, and purchasing behavior.

---

## 4. Exploratory Data Analysis

### Spending Analysis

Key Findings:

* Customer spending follows a right-skewed distribution.
* Most customers belong to low-to-medium spending groups.
* A smaller segment contributes significantly higher revenue.

### Income vs Spending

Key Findings:

* Income shows a strong positive relationship with spending behavior.
* Higher-income customers generally spend more.

### Campaign Response Analysis

Key Findings:

* Campaign acceptance rates are relatively low.
* The target variable is imbalanced.

### Correlation Analysis

Strong Positive Correlations:

| Features                         | Correlation |
| -------------------------------- | ----------- |
| Total Spending ↔ Total Purchases | 0.82        |
| Income ↔ Total Spending          | 0.79        |
| Income ↔ Total Purchases         | 0.74        |

Strong Negative Correlations:

| Features                           | Correlation |
| ---------------------------------- | ----------- |
| Income ↔ NumWebVisitsMonth         | -0.65       |
| Total Spending ↔ NumWebVisitsMonth | -0.50       |
| Total Spending ↔ Total Children    | -0.50       |

### Insights

* Higher-income customers spend more.
* Spending behavior strongly predicts customer value.
* Customer activity and engagement are important indicators of campaign response.

---

# Customer Segmentation

## Features Used

* Income
* Age
* Recency
* Total Spending
* Total Purchases
* NumWebVisitsMonth
* Customer Tenure

## Feature Scaling

StandardScaler was applied before clustering.

### Why Scaling?

Scaling ensures all features contribute equally during distance-based clustering.

---

## K-Means Clustering

### Cluster Optimization

Methods Used:

* KneeLocator Method
* Silhouette Score Analysis

### Final Clusters

K-Means identified **6 customer segments**.

### Key Insights

* Distinct customer groups were identified based on spending behavior, purchasing activity, engagement, and income.
* High-value segments exhibited significantly higher spending and campaign response rates.
* Low-value segments showed lower spending and engagement behavior.

### Business Value

Customer segmentation enables targeted marketing campaigns, customer retention strategies, and personalized promotions.

---

## DBSCAN Clustering

DBSCAN was evaluated as an alternative density-based clustering approach.

### Purpose

* Detect unusual customer behavior.
* Identify potential outliers.
* Compare density-based segmentation with K-Means.

### Insights

* DBSCAN successfully identified customers with abnormal spending and purchasing behavior.
* K-Means produced more balanced and interpretable customer segments.
* K-Means was selected as the final segmentation approach.

---

# Campaign Response Prediction

## Target Variable

Response

### Objective

Predict whether a customer will accept a marketing campaign offer.

---

# Machine Learning Models Evaluated

## Baseline Models

### Logistic Regression

Linear classification model used as a baseline.

### Naive Bayes

Probabilistic classifier based on Bayes' theorem.

### K-Nearest Neighbors (KNN)

Classifies customers using nearest neighboring observations.

### Support Vector Machine (SVM)

Constructs an optimal decision boundary between response classes.

### Decision Tree

Creates interpretable decision rules for classification.

---

## Ensemble Learning Models

### Random Forest

Bagging-based ensemble model using multiple decision trees.

### AdaBoost

Boosting algorithm that focuses on previously misclassified observations.

### Gradient Boosting

Sequential ensemble learning approach that minimizes prediction errors.

### XGBoost

Highly optimized gradient boosting implementation with regularization.

### LightGBM

Efficient gradient boosting framework using leaf-wise tree growth.

---

# Model Evaluation Metrics

The following metrics were used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

# Hyperparameter Optimization

## RandomizedSearchCV

Hyperparameter tuning was performed on the LightGBM model using RandomizedSearchCV with 5-fold cross-validation.

### Result

Although tuning improved precision, the original LightGBM model achieved better overall performance in terms of Recall, F1 Score, and ROC-AUC.

Therefore, the original LightGBM configuration was selected as the final production model.

---

# Key Results

## Customer Segmentation

* K-Means successfully identified six meaningful customer segments.
* High-value customers demonstrated higher spending and campaign engagement.
* DBSCAN detected unusual customer behavior and outliers.

## Campaign Response Prediction

### Best Model: LightGBM

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 88.39% |
| Precision | 67.44% |
| Recall    | 43.28% |
| F1 Score  | 52.73% |
| ROC-AUC   | 91.01% |

---

# Feature Importance Analysis

Feature importance was extracted from the final LightGBM model.

## Most Important Features

1. Recency
2. Customer Tenure
3. Income
4. MntMeatProducts
5. MntGoldProds
6. MntWines

### Business Insights

* Recently active customers are more likely to respond to campaigns.
* Customer tenure influences campaign engagement.
* Spending behavior is a strong predictor of customer response.
* Income remains a major driver of customer value.

---

# Business Recommendations

### High-Value Customers

* Offer premium products and exclusive rewards.
* Implement loyalty and retention programs.

### Medium-Value Customers

* Increase engagement through targeted promotions.
* Encourage cross-selling opportunities.

### Low-Value Customers

* Focus on value-driven campaigns and discounts.
* Promote product bundles and special offers.

### At-Risk Customers

* Launch re-engagement campaigns.
* Provide retention incentives.

### High Response Probability Customers

* Prioritize marketing budget allocation.
* Deliver personalized campaign content.

---

# Conclusion

This project successfully developed an end-to-end Customer Intelligence System integrating customer segmentation and campaign response prediction.

K-Means clustering identified six actionable customer segments, while machine learning and ensemble learning models were evaluated for campaign response prediction. Among all models, LightGBM achieved the best overall performance with a ROC-AUC of 91.01%.

Feature importance analysis revealed that Recency, Customer Tenure, Income, and Spending Behavior are the primary drivers of campaign response. The resulting insights can help businesses improve customer targeting, optimize marketing expenditure, increase retention, and enhance overall marketing effectiveness through data-driven decision-making.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost
* LightGBM
* Jupyter Notebook

---

# Author

**Bhavani**

B.Tech – Artificial Intelligence & Machine Learning

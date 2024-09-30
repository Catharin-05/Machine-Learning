# **Heart Disease Prediction Model**

## **Overview**
This project focuses on building a machine learning model to predict the likelihood of heart disease based on various health-related features. Using datasets containing key features like Age, Cholesterol, Blood Pressure, and Exercise Hours, the model aims to assist in early detection and prevention of heart-related conditions.

## **Key Features**
- **Logistic Regression**, **Support Vector Machine (SVM)**, and **Linear Discriminant Analysis (LDA)** models are evaluated.
- The final model achieves a **validation accuracy of 93.5%** and an **AUC (Area Under the ROC Curve) score of 0.98**, indicating excellent performance.
- The project uses **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)** for model interpretation, making the model's predictions transparent and explainable.

## **Data**
The dataset contains the following features:
- **Age**: The age of the individual
- **Cholesterol**: Cholesterol level in mg/dL
- **Blood Pressure**: Blood pressure in mm Hg
- **Heart Rate**: Resting heart rate in beats per minute
- **Exercise Hours**: Number of hours of exercise per week
- **Smoking**: Whether the individual smokes or not
- **Alcohol Intake**: Amount of alcohol consumed per week
- **Family History**: Family history of heart disease
- **Diabetes**: Whether the individual has diabetes
- **Obesity**: Obesity level (BMI)
- **Stress Level**: Measured stress level
- **Blood Sugar**: Blood sugar level (fasting blood glucose)
- **Exercise Induced Angina**: Whether angina is triggered by exercise
- **Chest Pain Type**: Type of chest pain experienced
- **Heart Disease**: Target variable, indicating if the individual has heart disease (1) or not (0)

## **Model Evaluation**
- **Logistic Regression Accuracy**: 83.5%
- **SVM Accuracy**: 93.5%
- **LDA Accuracy**: 86%
  
### **ROC Curve**
The Receiver Operating Characteristic (ROC) curve for the model shows an **AUC of 0.98**, demonstrating excellent performance in distinguishing between individuals with and without heart disease.

### **SHAP Analysis**
Using SHAP, we identified that **Age**, **Cholesterol**, and **Blood Sugar** are the most important features in predicting heart disease. Higher values of these features increase the risk of heart disease.

### **LIME Interpretation**
LIME was also used to provide local explanations of the predictions, ensuring that individual predictions can be explained in terms of feature contributions.

## **Installation**

To set up this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/catharin-05/Machine-Learning.git
   cd heart-disease-prediction
2. Install the required python packages:
   pip install -r requirements.txt

## **Usage**
After running the project, you can:

Train the model using the provided dataset.
Evaluate the model's performance using ROC curves, SHAP, and LIME.
Modify the dataset and features to customize the prediction outcomes.
Model Performance
Training Accuracy: 93.5%
Validation Accuracy: 93.5%
AUC: 0.98
The model generalizes well, avoiding overfitting, as shown by the convergence of training and validation accuracies.

## **Contributors**
Catharin Nivitha
## **License**
This project is licensed under the MIT License - see the LICENSE file for details.
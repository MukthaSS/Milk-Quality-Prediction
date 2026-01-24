
# 🥛 Milk Quality Prediction #### Hoseted on https://milk-quality-prediction-j6wi.onrender.com/docs


A machine learning classifier that predicts milk quality (Low / Medium / High) based on physicochemical properties and sensory attributes using scikit-learn.

---

## 📊 Overview

- **Goal**: Build a robust model to classify milk quality using measurements such as pH, temperature, taste, odor, fat content, turbidity, and color.
- **Type**: Multiclass classification (Low, Medium, High).
- **Tech**: Python, Jupyter Notebook, pandas, scikit-learn, matplotlib & seaborn for EDA & visualization.

---

## 🚀 Features

1. **Exploratory Data Analysis**
   - Statistical summary & missing value checks.
   - Correlation heatmap to discover feature relationships.
2. **Data Preparation**
   - Cleaning, encoding, train-test split & scaling for model readiness.
3. **Model Building**
   - Trains **SVM (linear kernel)**, **Decision Tree**, and **Random Forest** classifiers.
   - Hyperparameter tuning via GridSearchCV for optimal performance.
   - It's a machine learning model used for classification tasks. The goal is to find a decision boundary (called a hyperplane) that best separates the data into different classes (e.g., spam vs non-spam).
   - Kernel: In SVM, a kernel is a function that transforms data into a higher-dimensional space where it might be easier to separate classes.
   - Linear Kernel: When you use a linear kernel, the model tries to separate the data using a straight line (in 2D) or a flat plane (in 3D). This kernel is suitable when the data is linearly separable, meaning it can be divided by a straight line or plane.

4. **Model Evaluation**
   - Accuracy, precision, recall, and F1‑score metrics.
   - Confusion matrices and ROC curves for in-depth comparison.
5. **User Input Prediction**
   - A function to predict milk quality from custom inputs.

---

## 📁 Project Files

- `Milk Quality Prediction.ipynb` – The main Jupyter notebook covering everything from EDA to model evaluation.
- `milknew.csv` – Dataset containing milk samples and corresponding quality labels.
- ⭐ *Your new README.md* – This file!

---

## 🛠 Usage

```bash
git clone https://github.com/yourusername/Milk-Quality-Prediction.git
cd Milk-Quality-Prediction
pip install -r requirements.txt
jupyter notebook
```
## 📈 Results
- Best performing model: e.g., SVM (linear kernel) with ~95% accuracy.

- Learn detailed model performance, including confusion matrix plots, by running the notebook.

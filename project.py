import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, multilabel_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import wilcoxon
import os
import graphviz
import gdown

# Streamlit App Title
st.title("Network Attack Detection ML App")






@st.cache_data
def load_data():
    file_id = "1lDRgc1TTKZH6LXT3AiTYrL-JEQUlHOT0"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "dataset.csv"  # Save file locally

    # Download the file using gdown
    gdown.download(url, output, quiet=False)

    # Load the dataset
    df = pd.read_csv(output)

    # Drop 'id' column if it exists
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # Convert categorical columns to numerical codes
    for col in ['proto', 'service', 'state']:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    if 'attack_cat' in df.columns:
        df['attack_cat'] = df['attack_cat'].astype('category')

    return df


df = load_data()

st.write("### Data Sample")
st.dataframe(df.head())

# Data Preprocessing
X = df.drop(columns=['attack_cat', 'label'])
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
feature_names = list(X.columns)

# Model Training
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox("Choose a Model", ["Decision Tree", "Random Forest", "XGBoost", "LightGBM"])

if model_choice == "Decision Tree":
    params = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 4], 'min_samples_split': [2, 4], 'min_samples_leaf': [1, 2]}
    model = GridSearchCV(DecisionTreeClassifier(), params, cv=5, scoring='recall')
elif model_choice == "Random Forest":
    model = RandomForestClassifier(random_state=11)
elif model_choice == "XGBoost":
    model = XGBClassifier()
elif model_choice == "LightGBM":
    model = LGBMClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
st.write(f"### Model Performance: {model_choice}")
st.write(f"Accuracy: {accuracy:.4f}")
st.write(f"Recall: {recall:.4f}")
st.write(f"Precision: {precision:.4f}")

# Confusion Matrix
st.write("### Confusion Matrix")
cross = pd.crosstab(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cross, annot=True, fmt='d', cmap="YlGnBu", ax=ax)
st.pyplot(fig)

# Feature Importance
if model_choice == "Random Forest":
    feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
    feature_imp = feature_imp.sort_values(by='Importance', ascending=False)
    st.write("### Feature Importance")
    st.bar_chart(feature_imp.set_index("Feature"))

# User Input for Prediction
st.sidebar.header("Predict New Sample")
input_values = [st.sidebar.number_input(f"{feature}", value=0.0) for feature in feature_names]
if st.sidebar.button("Predict"):
    prediction = model.predict([input_values])
    st.sidebar.write(f"### Prediction: {'Attack' if prediction[0] == 1 else 'Normal'}")

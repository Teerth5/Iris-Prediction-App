import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Load the Iris Dataset
@st.cache
def load_data():
    data = pd.read_csv("E:\\INTERNSHIPS\\ORISON TECHNOLOGY\\TASK 2\\iris - iris.csv")
    return data

data = load_data()

st.title('Iris Species Prediction App')
st.write('This app predicts the Iris species using Linear Regression, Decision Tree, and Random Forest Classifiers')

# Display the dataset
st.write("### Iris Dataset", data)

# Features and target selection
X = data.drop('species', axis=1)
y = data['species']

# Encode the categorical target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox("Choose Classifier", ("Linear Regression", "Decision Tree", "Random Forest"))

# Implement Linear Regression Model
def linear_regression_model(X_train, X_test, y_train, y_test):
    st.write("## Linear Regression Model")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Convert predictions to integer for evaluation
    y_pred_int = np.round(y_pred).astype(int)
    
    # Accuracy
    mse = metrics.mean_squared_error(y_test, y_pred)
    st.write(f"### Mean Squared Error: {mse:.2f}")
    
    # Residuals Plot
    residuals = y_test - y_pred
    st.write("### Residuals Plot")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='r', linestyles='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    st.pyplot(plt)
    
    # Distribution of Residuals
    st.write("### Distribution of Residuals")
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title('Distribution of Residuals')
    st.pyplot(plt)
    
    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    corr_matrix = X.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    st.pyplot(plt)
    
    # Visualization
    st.write("### Actual vs Predicted (Linear Regression)")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted - Linear Regression')
    st.pyplot(plt)
    
    return model

# Implement Decision Tree Model
def decision_tree_model(X_train, X_test, y_train, y_test):
    st.write("## Decision Tree Classifier")
    
    model = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=5, min_samples_leaf=1)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    st.write(f"### Accuracy: {accuracy * 100:.2f}%")
    
    # Confusion Matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(plt)
    
    # Plot Decision Tree (Pruned)
    st.write("### Decision Tree (Pruned) Visualization")
    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=X.columns, class_names=[str(c) for c in model.classes_])
    plt.title('Decision Tree Visualization')
    st.pyplot(plt)
    
    return model

# Implement Random Forest Model
def random_forest_model(X_train, X_test, y_train, y_test):
    st.write("## Random Forest Classifier")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    st.write(f"### Accuracy: {accuracy * 100:.2f}%")
    
    # Confusion Matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(plt)
    
    # Feature Importance
    st.write("### Feature Importance (Random Forest)")
    feature_importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=X.columns)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance (Random Forest)')
    st.pyplot(plt)
    
    return model

# Run the selected model
if model_choice == "Linear Regression":
    model = linear_regression_model(X_train, X_test, y_train, y_test)
elif model_choice == "Decision Tree":
    model = decision_tree_model(X_train, X_test, y_train, y_test)
elif model_choice == "Random Forest":
    model = random_forest_model(X_train, X_test, y_train, y_test)

# Sidebar for user input to make predictions
st.sidebar.title("Make a Prediction")
sepal_length = st.sidebar.slider("Sepal Length", min_value=float(X['sepal_length'].min()), max_value=float(X['sepal_length'].max()))
sepal_width = st.sidebar.slider("Sepal Width", min_value=float(X['sepal_width'].min()), max_value=float(X['sepal_width'].max()))
petal_length = st.sidebar.slider("Petal Length", min_value=float(X['petal_length'].min()), max_value=float(X['petal_length'].max()))
petal_width = st.sidebar.slider("Petal Width", min_value=float(X['petal_width'].min()), max_value=float(X['petal_width'].max()))

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Make prediction
if model_choice == "Linear Regression":
    model = linear_regression_model(X_train, X_test, y_train, y_test)
    prediction = model.predict(input_data)
    # Convert numeric predictions to class labels
    prediction = le.inverse_transform(np.round(prediction).astype(int))
elif model_choice == "Decision Tree":
    model = decision_tree_model(X_train, X_test, y_train, y_test)
    prediction = model.predict(input_data)
    prediction = le.inverse_transform(prediction)
elif model_choice == "Random Forest":
    model = random_forest_model(X_train, X_test, y_train, y_test)
    prediction = model.predict(input_data)
    prediction = le.inverse_transform(prediction)

st.write("### Prediction:", prediction[0])

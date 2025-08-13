# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# -------------------
# Page Config
# -------------------
st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")

# -------------------
# Load data & model
# -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/Titanic-Dataset.csv")
    # Handle missing values as done in training
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode(dropna=True)[0])
    return df


@st.cache_resource
def load_model():
    with open("notebooks/best_model.pkl", "rb") as f:
        return pickle.load(f)

# Load the list of features the model was trained on
with open("notebooks/model_features.pkl", "rb") as f:
    loaded_features = pickle.load(f)

# The model was actually trained on features excluding Name, Ticket, PassengerId
# This matches what was done in the training notebook: X_train_num = X_train.drop(columns=['Name', 'Ticket', 'PassengerId'])
excluded_columns = ['Name', 'Ticket', 'PassengerId']
model_features = [col for col in loaded_features if col not in excluded_columns]

# Display the features being used
st.sidebar.subheader("🔍 Model Features")
st.sidebar.write(f"**Total features:** {len(model_features)}")
st.sidebar.write("**Feature list:**")
for i, feature in enumerate(model_features, 1):
    st.sidebar.write(f"{i}. {feature}")

# Preprocessing function to match training exactly
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    
    

    if "Age" in data.columns:
        data["Age"] = data["Age"].fillna(data["Age"].median())
    if "Embarked" in data.columns:
        data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode(dropna=True)[0])


    # Convert categorical features exactly as in training
    if "Sex" in data.columns:
        data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

    if "Embarked" in data.columns:
        data = pd.get_dummies(data, columns=["Embarked"], drop_first=True)

    # Drop unused columns exactly as in training - CRITICAL: must be done before adding missing columns
    columns_to_drop = ["Name", "Ticket", "PassengerId", "Cabin"]
    for col in columns_to_drop:
        if col in data.columns:
            data.drop(columns=col, inplace=True)

    # Add any missing columns from training with 0 values
    for col in model_features:
        if col not in data.columns:
            data[col] = 0

    # Ensure correct column order
    data = data[model_features]

    # Convert to proper numeric types to avoid Arrow serialization issues
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        elif data[col].dtype == 'Int64':
            data[col] = data[col].astype('int64')
        elif data[col].dtype == 'Float64':
            data[col] = data[col].astype('float64')

    return data

df = load_data()
model = load_model()

# -------------------
# App Title & Description
# -------------------
st.title("🚢 Titanic Survival Prediction App")
st.markdown("""
### Predict survival chances of Titanic passengers
This app allows you to:
- **Explore the dataset**
- **Visualise patterns & relationships**
- **Make survival predictions**
- **View model performance metrics**
""")

# -------------------
# Sidebar Navigation
# -------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Data Exploration", "Visualisations", "Model Prediction", "Model Performance"]
)

# -------------------
# Data Exploration
# -------------------
if menu == "Data Exploration":
    st.header("🔍 Data Exploration")

    st.subheader("Dataset Overview")
    st.write(f"Shape: {df.shape}")


    st.subheader("Data Types")
    dtype_df = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str)
    })
    st.table(dtype_df)


    st.write("Missing Values:", df.isnull().sum())

    st.subheader("Sample Data")
    st.dataframe(df.head())

    st.subheader("Interactive Filter")
    sex_filter = st.selectbox("Filter by Gender", ["All"] + df["Sex"].unique().tolist())
    pclass_filter = st.selectbox("Filter by Pclass", ["All"] + sorted(df["Pclass"].unique().tolist()))

    filtered_df = df.copy()
    if sex_filter != "All":
        filtered_df = filtered_df[filtered_df["Sex"] == sex_filter]
    if pclass_filter != "All":
        filtered_df = filtered_df[filtered_df["Pclass"] == pclass_filter]

    st.dataframe(filtered_df)

# -------------------
# Visualisations
# -------------------
elif menu == "Visualisations":
    st.header("📊 Visualisations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Survival Count")
        fig1 = px.histogram(df, x="Survived", title="Survival Count")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Survival by Gender")
        fig2 = px.histogram(df, x="Sex", color="Survived", barmode="group", title="Survival by Gender")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Age Distribution")
    fig3 = px.histogram(df, x="Age", nbins=30, marginal="box", title="Age Distribution")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Correlation Heatmap")
    # Use only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# -------------------
# Model Prediction
# -------------------
elif menu == "Model Prediction":
    st.header("🎯 Model Prediction")

    with st.form("prediction_form"):
        st.write("Enter passenger details:")

        pclass = st.selectbox("Pclass", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.number_input("Age", min_value=0, max_value=100, value=25)
        sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
        parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
        fare = st.number_input("Fare", min_value=0.0, value=30.0)
        embarked = st.selectbox("Embarked", ["C", "Q", "S"])

        submit = st.form_submit_button("Predict")

    if submit:
        try:
            # Prepare input data
            input_data = pd.DataFrame({
                "Pclass": [pclass],
                "Sex": [sex],
                "Age": [age],
                "SibSp": [sibsp],
                "Parch": [parch],
                "Fare": [fare],
                "Embarked": [embarked]
            })
            input_data = preprocess_data(input_data)

            # Debug: Show columns being passed to model
            st.write("Model input columns:", input_data.columns.tolist())

            with st.spinner("Making prediction..."):
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1]

            st.success(f"Prediction: {'Survived' if prediction == 1 else 'Did Not Survive'}")
            st.info(f"Confidence: {probability*100:.2f}%")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# -------------------
# Model Performance
# -------------------
elif menu == "Model Performance":
    st.header("📈 Model Performance")

    # Debug: Show what features we're working with
    st.write("**Training features being used:**", model_features)
    
    # Preprocess features before splitting
    X = preprocess_data(df.drop(columns=["Survived"]))
    y = df["Survived"]
    
    # Keep only training features - this matches what was done during training
    X = X[model_features]
    
    st.write("**Processed features shape:**", X.shape)
    st.write("**Processed features columns:**", X.columns.tolist())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write(f"**Accuracy:** {acc:.4f}")
    st.write(f"**Precision:** {prec:.4f}")
    st.write(f"**Recall:** {rec:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Did Not Survive", "Survived"],
                yticklabels=["Did Not Survive", "Survived"], ax=ax)
    st.pyplot(fig)

    st.subheader("Model Comparison Results")
    st.markdown("- Logistic Regression CV Accuracy: 0.79")
    st.markdown("- Random Forest CV Accuracy: 0.80")
    st.markdown("**Best Model:** Random Forest (selected during training)")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import mysql.connector
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# MySQL Database Connection
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="alkhailgate",
    database="user_db"
)
cursor = conn.cursor()

# Load Data
file_path = 'C:\\Users\\theha\\OneDrive\\Desktop\\industry\\public\\dataset.csv'
df = pd.read_csv(file_path)

# Rename columns
rename_columns = {
    'Air temperature [K]': 'AirTemp',
    'Process temperature [K]': 'ProcessTemp',
    'Rotational speed [rpm]': 'RotSpeed',
    'Torque [Nm]': 'Torque',
    'Tool wear [min]': 'ToolWear',
    'Machine failure': 'MachineFailure'
}
df.rename(columns=rename_columns, inplace=True)

# Feature Engineering
feature_columns = ['AirTemp', 'ProcessTemp', 'RotSpeed', 'Torque', 'ToolWear']
X = df[feature_columns]
y = df['MachineFailure']

# Normalize features
df_normalized = (X - X.min()) / (X.max() - X.min())
df['FailureProbability'] = df_normalized.mean(axis=1).clip(0, 1)
df['FailureCause'] = df_normalized.idxmax(axis=1)
df['PredictedRUL'] = (1 - df['FailureProbability']) * (df['ToolWear'].max() / 2)
df['PredictedRUL'] = df['PredictedRUL'].clip(0, df['ToolWear'].max())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline for Logistic Regression
log_reg_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Cross Validation
cv_scores = cross_val_score(log_reg_pipeline, X, y, cv=5, scoring='accuracy')
cv_mean_accuracy = np.mean(cv_scores)

# Train Logistic Regression
log_reg_pipeline.fit(X_train, y_train)

y_pred = log_reg_pipeline.predict(X_test)
y_prob_train = log_reg_pipeline.predict_proba(X_train)[:, 1]
y_prob_test = log_reg_pipeline.predict_proba(X_test)[:, 1]

# Compute classification metrics
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, output_dict=True)

# Train Random Forest Regressor for RUL Prediction
X_train_rfr = X_train.copy()
X_train_rfr['FailureProbability'] = y_prob_train
X_test_rfr = X_test.copy()
X_test_rfr['FailureProbability'] = y_prob_test

rul_target = df['PredictedRUL']
rul_train = rul_target.loc[X_train.index]
rul_test = rul_target.loc[X_test.index]

rfr = RandomForestRegressor(n_estimators=50, random_state=42)
rfr.fit(X_train_rfr, rul_train)

# Function to store predictions in MySQL
def store_prediction(air_temp, process_temp, rot_speed, torque, tool_wear, failure_prob, predicted_rul, failure_cause):
    query = """
    INSERT INTO MachineFailurePredictions (AirTemp, ProcessTemp, RotSpeed, Torque, ToolWear, FailureProbability, PredictedRUL, FailureCause)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (air_temp, process_temp, rot_speed, torque, tool_wear, failure_prob, predicted_rul, failure_cause)
    cursor.execute(query, values)
    conn.commit()

# Prediction function
def predict_failure_and_rul(input_data):
    input_df = pd.DataFrame([input_data])
    input_df_normalized = (input_df - X.min()) / (X.max() - X.min())
    failure_prob = (0.7 * input_df_normalized.mean(axis=1).values[0]) + (0.3 * log_reg_pipeline.predict_proba(input_df)[:, 1][0])
    failure_prob = np.clip(failure_prob, 0, 1)
    input_df['FailureProbability'] = failure_prob
    input_df['FailureCause'] = input_df_normalized.idxmax(axis=1).values[0]
    predicted_rul = rfr.predict(input_df[feature_columns + ['FailureProbability']])[0]
    
    store_prediction(input_data['AirTemp'], input_data['ProcessTemp'], input_data['RotSpeed'],
                     input_data['Torque'], input_data['ToolWear'], failure_prob, predicted_rul,
                     input_df['FailureCause'].values[0])
    
    return failure_prob, predicted_rul, input_df['FailureCause'].values[0]

# Streamlit UI
st.title('Machine Failure Prediction & RUL Estimation')

st.sidebar.header('Input Machine Parameters')
air_temp = st.sidebar.number_input('Air Temperature (K)', value=300.0)
process_temp = st.sidebar.number_input('Process Temperature (K)', value=310.0)
rot_speed = st.sidebar.number_input('Rotational Speed (rpm)', value=1500.0)
torque = st.sidebar.number_input('Torque (Nm)', value=40.0)
tool_wear = st.sidebar.number_input('Tool Wear (min)', value=100.0)

if st.sidebar.button('Predict'):
    input_data = {'AirTemp': air_temp, 'ProcessTemp': process_temp, 'RotSpeed': rot_speed, 'Torque': torque, 'ToolWear': tool_wear}
    failure_prob, predicted_rul, failure_cause = predict_failure_and_rul(input_data)
    
    st.subheader('Prediction Results')
    st.write(f"**Failure Probability:** {failure_prob:.4f}")
    st.write(f"**Predicted Remaining Useful Life (RUL):** {predicted_rul:.2f} units")
    st.write(f"**Potential Cause of Failure:** {failure_cause}")
    
    if failure_prob > 0.8 or predicted_rul < 5:
        st.error('⚠️ Urgent Maintenance Alert: Immediate attention required!')
    elif failure_prob > 0.5 or predicted_rul < 15:
        st.warning('⚠️ Maintenance Recommended: Schedule maintenance soon.')
    else:
        st.success('✅ Machine is in good condition.')

    # Display Logistic Regression Metrics
    st.subheader("Logistic Regression Evaluation Metrics")
    st.write(f"**Accuracy (on test set):** {accuracy:.4f}")
    st.write(f"**Cross-Validation Accuracy (mean of 5-fold):** {cv_mean_accuracy:.4f}")
    
    st.write("**Classification Report:**")
    st.dataframe(pd.DataFrame(classification_rep).transpose())

    # Plot Actual vs Predicted RUL
    st.subheader("Actual vs Predicted RUL")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(rul_test, rfr.predict(X_test_rfr), alpha=0.7, color='blue', label="Predicted")
    ax.plot([rul_test.min(), rul_test.max()], [rul_test.min(), rul_test.max()], 'r--', lw=2, label="Ideal")
    ax.set_title("Actual vs Predicted Remaining Useful Life (RUL)")
    ax.set_xlabel("Actual RUL (Units)")
    ax.set_ylabel("Predicted RUL (Units)")
    ax.legend()
    st.pyplot(fig)

# Close MySQL connection
cursor.close()
conn.close()

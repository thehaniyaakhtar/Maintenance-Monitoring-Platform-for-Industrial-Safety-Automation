import streamlit as st
import json
from datetime import datetime
import os

# Path to store data
data_file_path = "data.json"

# Simulating user input form
st.title("Machine Failure Analysis")

# Collecting user inputs through Streamlit UI
air_temp = st.number_input("Enter Air Temperature (K)", min_value=0)
process_temp = st.number_input("Enter Process Temperature (K)", min_value=0)
rotational_speed = st.number_input("Enter Rotational Speed (rpm)", min_value=0)
torque = st.number_input("Enter Torque (Nm)", min_value=0)
tool_wear = st.number_input("Enter Tool Wear (min)", min_value=0)

# Button for submitting the data
if st.button("Submit"):
    # Collect data into a dictionary
    data = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "air_temp": air_temp,
        "process_temp": process_temp,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear,
    }

    # Try reading the existing data from the JSON file if it exists
    try:
        with open(data_file_path, "r") as file:
            historical_data = json.load(file)
    except FileNotFoundError:
        historical_data = []

    # Append new data to the existing historical data
    historical_data.append(data)

    # Write the updated data back to the JSON file
    with open(data_file_path, "w") as file:
        json.dump(historical_data, file, indent=4)

    st.success("Data successfully saved!")

# Button to view data in HTML page
if st.button("View Data"):
    # Redirect to the HTML page (index.html)
    st.markdown('<a href="index.html" target="_blank">Click here to view the data</a>', unsafe_allow_html=True)

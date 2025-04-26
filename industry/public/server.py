from flask import Flask, render_template
import json

app = Flask(__name__)

@app.route("/")
def index():
    # Read the data from data.json
    try:
        with open("C:\Users\theha\OneDrive\Desktop\industry\public\data.json", "r") as file:
            historical_data = json.load(file)
    except FileNotFoundError:
        historical_data = []

    # Pass the data to the HTML template
    return render_template("index.html", data=historical_data)

if __name__ == "__main__":
    app.run(debug=True)

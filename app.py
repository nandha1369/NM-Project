from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load model (assuming it's a pickle file)
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from form
        input_data = request.form["input_name"]  # Ensure "input_name" matches the HTML form
        input_df = pd.DataFrame([[float(input_data)]], columns=["feature"])
        
        # Predict result
        prediction = model.predict(input_df)[0]

        return render_template("result.html", prediction=round(prediction, 2))
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)

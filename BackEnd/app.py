from flask import Flask,request,jsonify
# Flask is to create an api end point 
# request is to get the data from the streamlit app
# jsonify is to covert the data into json format and return it to the streamlit app
import joblib
import pandas as pd

app = Flask(__name__) #creating ana object of the Flask class

# Load the trained model 
model = joblib.load("random_forest_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint to get predections from the trained model."""

    data = request.json # Get data from the request {"total_bill":16.99, "sex":"MAle"....}

    input_df = pd.DataFrame([{"total_bill": data["total_bill"],
                             "sex": data["sex"],
                             "smoker": data["smoker"],
                             "day": data["day"],
                             "time": data["time"],
                             "size": data["size"]}]) # Covert data to DataFrame
    # Covert data to DataFrame
    prediction = model.predict(input_df) # Make prediction # array[tip_value]
    return jsonify({"prediction": prediction[0]})


if __name__=="__main__":
    app.run(debug=True)
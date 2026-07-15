"""
Smart Lender — Flask application
Loads the trained RandomForest model (rdf.pkl) and serves three pages:
  /            -> home.html      (landing page)
  /predict     -> predict.html   (intake form)
  /submit      -> submit.html    (prediction result, POST target of the form)
"""

import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# ---- Load the trained model -------------------------------------------------
MODEL_PATH = "model/rdf.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---- Encoding maps (must match how the training data was encoded) ----------
GENDER_MAP = {"Male": 1, "Female": 0}
MARRIED_MAP = {"Yes": 1, "No": 0}
DEPENDENTS_MAP = {"0": 0, "1": 1, "2": 2, "3+": 3}
EDUCATION_MAP = {"Graduate": 1, "Not Graduate": 0}
SELF_EMPLOYED_MAP = {"Yes": 1, "No": 0}
PROPERTY_AREA_MAP = {"Rural": 0, "Semiurban": 1, "Urban": 2}


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/submit", methods=["POST"])
def submit():
    form = request.form

    gender = form.get("Gender")
    married = form.get("Married")
    dependents = form.get("Dependents")
    education = form.get("Education")
    self_employed = form.get("Self_Employed")
    applicant_income = float(form.get("ApplicantIncome", 0))
    coapplicant_income = float(form.get("CoapplicantIncome", 0))
    loan_amount = float(form.get("LoanAmount", 0))
    loan_amount_term = float(form.get("Loan_Amount_Term", 0))
    credit_history = float(form.get("Credit_History", 0))
    property_area = form.get("Property_Area")

    # Encode categorical fields exactly as they were encoded for training
    features = [
        GENDER_MAP.get(gender, 0),
        MARRIED_MAP.get(married, 0),
        DEPENDENTS_MAP.get(dependents, 0),
        EDUCATION_MAP.get(education, 0),
        SELF_EMPLOYED_MAP.get(self_employed, 0),
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history,
        PROPERTY_AREA_MAP.get(property_area, 0),
    ]

    prediction = model.predict(np.array(features).reshape(1, -1))
    # Note: In this specific model version, label 0 corresponds to Approval
    # and label 1 corresponds to Rejection.
    approved = int(prediction[0]) == 0

    summary = {
        "Gender": gender,
        "Marital Status": married,
        "Dependents": dependents,
        "Education": education,
        "Self Employed": self_employed,
        "Applicant Income": applicant_income,
        "Co-applicant Income": coapplicant_income,
        "Loan Amount (thousands)": loan_amount,
        "Loan Term (days)": loan_amount_term,
        "Credit History": "Yes" if credit_history == 1 else "No",
        "Property Area": property_area,
    }

    if approved:
        prediction_text = "Approved"
        headline = "This application clears underwriting."
        message = (
            "Based on the details provided, the model reads this application "
            "as a strong candidate for approval."
        )
    else:
        prediction_text = "Declined"
        headline = "This application does not clear underwriting."
        message = (
            "Based on the details provided, the model does not currently "
            "recommend approval. A manual review may still be warranted."
        )

    return render_template(
        "submit.html",
        prediction_text=prediction_text,
        approved=approved,
        headline=headline,
        message=message,
        summary=summary,
    )


if __name__ == "__main__":
    app.run(debug=True)

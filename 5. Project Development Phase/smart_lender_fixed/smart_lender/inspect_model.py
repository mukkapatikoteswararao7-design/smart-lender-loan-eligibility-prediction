import pickle
import numpy as np

def inspect():
    try:
        with open('model/rdf.pkl', 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model type: {type(model)}")
        print(f"Feature count: {model.n_features_in_}")
        print(f"Classes: {model.classes_}")
        
        # Test with a high-quality applicant vector
        # Features order in app.py:
        # 0: Gender (Male: 1, Female: 0)
        # 1: Married (Yes: 1, No: 0)
        # 2: Dependents (0: 0, 1: 1, 2: 2, 3+: 3)
        # 3: Education (Graduate: 1, Not Graduate: 0)
        # 4: Self_Employed (Yes: 1, No: 0)
        # 5: ApplicantIncome
        # 6: CoapplicantIncome
        # 7: LoanAmount
        # 8: Loan_Amount_Term
        # 9: Credit_History (1.0 or 0.0)
        # 10: Property_Area (Rural: 0, Semiurban: 1, Urban: 2)
        
        # Let's try a very obvious "good" case from common datasets (like Loan Prediction Dataset)
        # Often the order is: Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
        
        # Try a lower loan amount
        test_vector_low_loan = np.array([[1, 1, 0, 1, 0, 5000.0, 2000.0, 100.0, 360.0, 1.0, 1]])
        prediction_low_loan = model.predict(test_vector_low_loan)
        print(f"Test Prediction (Low Loan): {prediction_low_loan}")

        # Try a case that is usually rejected
        test_vector_bad = np.array([[0, 0, 3, 0, 1, 1000.0, 0.0, 500.0, 180.0, 0.0, 0]])
        prediction_bad = model.predict(test_vector_bad)
        print(f"Test Prediction (Bad Applicant): {prediction_bad}")

        # If both are 0, maybe 0 means Approved?
        # Let's check the probabilities
        probs_good = model.predict_proba(test_vector_low_loan)
        probs_bad = model.predict_proba(test_vector_bad)
        print(f"Probs (Good): {probs_good}")
        print(f"Probs (Bad): {probs_bad}")

        # Try switching the prediction logic
        # Maybe 0 is Approved and 1 is Declined?
        # Usually Credit_History=1.0 is the strongest predictor for Approval.
        # Let's see if Credit_History=0.0 makes it even more '0'.
        test_no_history = np.array([[1, 1, 0, 1, 0, 5000.0, 2000.0, 100.0, 360.0, 0.0, 1]])
        probs_no_history = model.predict_proba(test_no_history)
        print(f"Probs (No Credit History): {probs_no_history}")
        
        # Try a case that MUST be approved (1)
        # Often in this dataset, label 1 is Approved.
        # Let's try to find a combination that yields 1.
        test_perfect = np.array([[1, 1, 0, 1, 0, 10000.0, 5000.0, 50.0, 360.0, 1.0, 1]])
        probs_perfect = model.predict_proba(test_perfect)
        print(f"Probs (Perfect): {probs_perfect}")
        
        # Test a case with extreme income
        test_extreme = np.array([[1, 1, 0, 1, 0, 50000.0, 0.0, 10.0, 360.0, 1.0, 1]])
        probs_extreme = model.predict_proba(test_extreme)
        print(f"Probs (Extreme): {probs_extreme}")

        # Let's try to find label 1
        for i in range(1000):
            # Try more realistic values
            rand_vec = np.array([[
                np.random.choice([0, 1]), # Gender
                np.random.choice([0, 1]), # Married
                np.random.choice([0, 1, 2, 3]), # Dependents
                np.random.choice([0, 1]), # Education
                np.random.choice([0, 1]), # Self_Employed
                np.random.randint(1000, 20000), # ApplicantIncome
                np.random.randint(0, 10000), # CoapplicantIncome
                np.random.randint(10, 500), # LoanAmount
                np.random.choice([180, 360, 480]), # Loan_Amount_Term
                np.random.choice([0.0, 1.0]), # Credit_History
                np.random.choice([0, 1, 2]) # Property_Area
            ]])
            pred = model.predict(rand_vec)
            if pred[0] == 1:
                print(f"Found label 1 with vector: {rand_vec}")
                print(f"Probs: {model.predict_proba(rand_vec)}")
                break
        else:
            print("Did not find label 1 in 1000 random trials.")

        # Check if labels are swapped
        # Often in these datasets, 1 = Y (Approved), 0 = N (Rejected)
        # But let's verify.
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()

import gradio as gr
import pandas as pd
import joblib


try:
    best_model = joblib.load('loan_model.pkl')
except:
    print("Error: loan_model.pkl not found. Run rf-train.py first.")

def predict_loan(gender, married, dependents, education, self_employed, loan_amt, term, credit_hist, property_area, income):

    input_data = pd.DataFrame([[gender, married, dependents, education, self_employed, loan_amt, term, credit_hist, property_area, income]],
                              columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Total_Income'])
    
    # 3. Make Prediction
    prediction = best_model.predict(input_data)
    
    # 4. Return result
    return "✅ Loan Approved" if prediction[0] == 1 else "❌ Loan Rejected"


inputs = [
    gr.Dropdown(['Male', 'Female'], label="Gender"),
    gr.Dropdown(['Yes', 'No'], label="Married"),
    gr.Dropdown(['0', '1', '2', '3+'], label="Dependents"),
    gr.Dropdown(['Graduate', 'Not Graduate'], label="Education"),
    gr.Dropdown(['Yes', 'No'], label="Self Employed"),
    gr.Number(label="Loan Amount (e.g., 120)"),
    gr.Number(label="Loan Amount Term (e.g., 360)"),
    gr.Dropdown([1.0, 0.0], label="Credit History (1.0 for Good, 0.0 for Bad)"),
    gr.Dropdown(['Urban', 'Semiurban', 'Rural'], label="Property Area"),
    gr.Number(label="Total Monthly Income (Applicant + Co-applicant)")
]

interface = gr.Interface(
    fn=predict_loan, 
    inputs=inputs, 
    outputs="text", 
    title="Loan Approval Predictor",
    description="Enter customer details to predict loan approval status."
)

if __name__ == "__main__":
    interface.launch()
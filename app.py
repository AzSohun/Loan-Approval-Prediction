import gradio as gr
import numpy as np
import pandas as pd
import pickle



# with open("loan_model.pkl", "rb") as file:
#     model = pickle.load(file)


def predict_loan(gender, married, dependents, education, self_employed, loan_amt, term, credit_hist, property_area, income):
    
    # Create a dataframe for the input
    input_data = pd.DataFrame([[gender, married, dependents, education, self_employed, loan_amt, term, credit_hist, property_area, income]],
                              columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Total_Income'])
    
    prediction = best_model.predict(input_data)
    return "Approved" if prediction[0] == 1 else "Rejected"

inputs = [
    gr.Dropdown(['Male', 'Female'], label="Gender"),
    gr.Dropdown(['Yes', 'No'], label="Married"),
    gr.Dropdown(['0', '1', '2', '3+'], label="Dependents"),
    gr.Dropdown(['Graduate', 'Not Graduate'], label="Education"),
    gr.Dropdown(['Yes', 'No'], label="Self Employed"),
    gr.Number(label="Loan Amount"),
    gr.Number(label="Loan Amount Term"),
    gr.Dropdown([1.0, 0.0], label="Credit History"),
    gr.Dropdown(['Urban', 'Semiurban', 'Rural'], label="Property Area"),
    gr.Number(label="Total Monthly Income")
]

interface = gr.Interface(fn=predict_loan, inputs=inputs, outputs="text", title="Loan Approval Predictor")
interface.launch(share=True)

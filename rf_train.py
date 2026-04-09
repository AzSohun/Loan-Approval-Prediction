import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import gradio as gr


df = pd.read_csv("loan_predication.csv")

print(df)


print("Dataset Shape:", df.shape)
print(df.head())


df.drop('Loan_ID', axis=1, inplace=True)


df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1, inplace=True)

# Separate Features and Target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status'].map({'Y': 1, 'N': 0}) # Encoding target

# Identify column types
numeric_features = ['LoanAmount', 'Loan_Amount_Term', 'Total_Income']
categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()) # Step 4: Scaling
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # Step 5: Encoding
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# Justification: Random Forest is selected because it handles non-linear relationships well,
# is robust to outliers, and works effectively on tabular data with mixed feature types.
model = RandomForestClassifier(random_state=42)

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)
print("Model trained successfully.")




cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)


best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model for deployment
joblib.dump(best_model, 'loan_model.pkl')




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




import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.title('Heart Disease Prediction App')

image_path = 'images.jpg'  # Adjust the path accordingly
st.image(image_path, width=700)
# Get age from user
age = st.sidebar.number_input('Enter your age', min_value=1, max_value=150, step=1)
sex = st.sidebar.selectbox("Sex:",("Male", "Female"))
cp = st.sidebar.selectbox("Chest Pain: ",("Atypical Angina", "Non-anginal pain", "Typical angina","Asymptomatic"))
trestbps = st.sidebar.slider("Enter your resting blood sugar", 50,250)
chol = st.sidebar.slider("Enter your Cholestrol: ", 0,  250)
fbs = st.sidebar.selectbox("Blood Sugar: ",("Greater than 120mg/dl", "Lower than 120mg/dl"))
restecg   = st.sidebar.selectbox("Resting electrocardiographic result", ("Value 0: showing probable or definite left ventricular hypertrophy by Estes’ criteria", "Value 1: normal","Value 2: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)"))
thalach = st.sidebar.number_input("Enter maximum heart rate achieved",min_value = 50, max_value = 250)
exang = st.sidebar.selectbox("Exercise induced angina: ", ("Yes", "No"))
oldpeak = st.sidebar.number_input(" ST depression induced by exercise relative to rest", min_value=0.0, max_value = 10.0)
slope = st.sidebar.selectbox("Slope of the peak exercise ST segment: ", ("Downsloping", "Flat", "Upsloping"))
ca = st.sidebar.selectbox("Number of major vessels: ", ("0","1","2"))
thal = st.sidebar.selectbox("A blood disorder called thalassemia: ", ("Fixed defect", "Normal blood flow","Reversible defect"))

data = pd.read_csv("heart.csv")
x = data.drop("target",axis=1)
y = data["target"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=5)

from sklearn.naive_bayes import GaussianNB
dt= DecisionTreeClassifier()
dt.fit(x_train,y_train)
pred = dt.predict(x_test)
score_nb = round(accuracy_score(pred,y_test)*100,2)


# Function to make prediction based on user input
def predict_disease(user_data):
    # Convert user input into DataFrame to predict
    input_df = pd.DataFrame([user_data])
    prediction = nb.predict(input_df)
    probability = nb.predict_proba(input_df)[0][1]  # Probability of class 1
    return prediction, probability

# Mapping user input to the expected numerical or categorical codes
sex_mapping = {'Male': 1, 'Female': 0}
cp_mapping = {'Typical angina': 3, 'Atypical Angina': 1, 'Non-anginal pain': 2, 'Asymptomatic':0}
fbs_mapping = {'Greater than 120mg/dl': 1, 'Lower than 120mg/dl': 0}
restecg_mapping = {'Value 0: showing probable or definite left ventricular hypertrophy by Estes’ criteria': 0, 'Value 1: normal': 1, 'Value 2: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)': 2}
exang_mapping = {'Yes': 1, 'No': 0}
slope_mapping = {'Upsloping': 2, 'Flat': 1, 'Downsloping': 0}
ca_mapping = {'0': 0, '1': 1, '2': 2}
thal_mapping = {'Fixed defect': 0, 'Normal blood flow': 1, 'Reversible defect': 2}

# Collect and map user input data
user_input = [
    age,
    sex_mapping[sex],
    cp_mapping[cp],
    trestbps,
    chol,
    fbs_mapping[fbs],
    restecg_mapping[restecg],
    thalach,
    exang_mapping[exang],
    oldpeak,
    slope_mapping[slope],
    ca_mapping[ca],
    thal_mapping[thal]
]

# Make prediction
prediction, probability = predict_disease(user_input)
st.write("Prediction: ",prediction, "Probability: ", probability)
# # Display prediction
if prediction[0] == 1:
    st.write(f"Based on the inputs, there is a {probability*100:.2f}% probability that you have heart disease.")
else:
    st.write(f"Based on the inputs, there is a {100 - probability*100:.2f}% probability that you do not have heart disease.")

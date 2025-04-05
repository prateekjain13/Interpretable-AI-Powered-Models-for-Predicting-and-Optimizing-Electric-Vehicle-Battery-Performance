import streamlit as st
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost
import joblib
import plotly
import time
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
from email.mime.text import MIMEText
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from twilio.rest import Client 

st.set_page_config(layout="wide")

st.title("Battery RUL Prediction App")

# Load the saved models
reg_model_files = {
    "Model 1": "ExtraTreesRegressor_best_model.pkl",
    "Model 2": "XGBoost_best_model.pkl",
    "Model 3": "LightGBM_best_model.pkl",
    "Model 4": "LR_model.pkl"
}

reg_models = {}
# Load each model using joblib
for name, file in reg_model_files.items():
    reg_models[name] = joblib.load(file)  # Use joblib.load() instead of pickle.load()

clf_model = joblib.load("rul_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load dataset to determine input ranges
data = pd.read_csv("new_RUL_data.csv")
df = data.copy()
feature_names = df.drop("RUL", axis=1).columns.tolist()
print(df.shape)
missing = [feature for feature in df.columns if feature not in feature_names]
print("Missing features: ", missing)

scaler = MinMaxScaler()
scaler.fit(df[feature_names])


# mean_predictions!
def classify_battery_health(rul, max_rul):
    print("40% -> ", 0.4 * max_rul)
    print("70% -> ",0.7 * max_rul)
    if rul > 0.7 * max_rul:
        return "Healthy"
    elif rul >= 0.4 * max_rul:
        return "Moderate"
    else:
        return "Critical"

def send_email_alert(pred_health_label, recipient_email):
    message = Mail(
        from_email="prateekjain442@gmail.com",  # Must be verified in SendGrid
        to_emails="880prateekjain@gmail.com",
        subject="Battery Health Alert!",
        html_content=f"<strong>Alert:</strong> Battery health is at {pred_health_label}. Immediate attention required!"
    )
    try:
        sg = SendGridAPIClient("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")
        response = sg.send(message)
        print("Email sent successfully!", response.status_code)
    except Exception as e:
        print(f"Error: {e}")

# SMS Alert Function
def send_sms_alert(prediction, recipient_phone):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    message = client.messages.create(
        body=f"⚠️ ALERT! Battery Health is in Critical Condition: {prediction}. Take immediate action!",
        from_=TWILIO_PHONE_NUMBER,
        to=recipient_phone
    )

    print(f"✅ SMS Alert Sent: {message.sid}")

# Twilio Credentials
TWILIO_ACCOUNT_SID = "ACb2077d8be40a16594252978b8db056b9"
TWILIO_AUTH_TOKEN = "d01fccd179bf2c1cce9c34e7ed746608"
TWILIO_PHONE_NUMBER = "+13347814101"  # Your Twilio number

tab1, tab2 = st.tabs(["RUL Prediction", "Model explanability"])

with tab1:
    # Streamlit UI

    #  user input
    st.header("Enter Feature Values")
    user_input = {}

    cols = st.columns(3)  # Create 3 columns

    for idx, feature in enumerate(feature_names):
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())

        with cols[idx % 3]:  # Distribute inputs across 3 columns
            if min_val != max_val:
                user_input[feature] = st.number_input(feature, min_val, max_val)
            else:
                user_input[feature] = min_val        


    # ------------------------------- TEST -----------------------------------
    # user_input = [921.0,1145.02,314.6666666660458,3.857,3.669,1781.3509999997914,2492.35,7414.82,100,0.0,1106.9516666666666,7725.836666666666,6269.799999999999,3.8296666666666663,0.1880000000000001,1.5557378910410224,2.176686870098339,6.4757122146338055]
    # input_array = np.array([user_input]).reshape(1, -1)
    input_array = np.array([list(user_input.values())]).reshape(1, -1)
    scaled_input = scaler.transform(input_array)

    # ------------------------------------------------------------------------

    mean_prediction = 0
    if st.button("Predict"):
        # st.session_state.first_button_clicked = True

        predictions = {}
        for name, model in reg_models.items():
            predicted_value = model.predict(scaled_input)[0]
            predictions[name] = predicted_value
            mean_prediction += predicted_value
        mean_prediction /= 4 
        print(" Mean prediction : ", mean_prediction)
        st.subheader("Model Predictions Comparison")

        # Create columns for side-by-side comparison
        cols = st.columns(len(predictions))

        # Display Model Names
        for idx, (name, _) in enumerate(predictions.items()):
            with cols[idx]:
                st.write(f"**{name}**")

        # Display Predictions
        for idx, (_, pred) in enumerate(predictions.items()):
            with cols[idx]:
                st.write(f"🔢 **{pred:.2f} RUL**") 
        
        # Store results in session state
        st.session_state.predictions = predictions
        st.session_state.mean_prediction = mean_prediction

        colx, coly = st.columns(2)
        # if st.session_state.predictions:
        with colx:
            # 📈 **Line Chart for Model Predictions**
            
            st.subheader("Predictions Visualization - Line Chart")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(list(predictions.keys()), list(predictions.values()), marker='o', linestyle='-', color='b', linewidth=2)

            ax.set_ylabel("Predicted RUL")
            ax.set_xlabel("Models")
            ax.set_title("Comparison of Model Predictions")

            # Show grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)

            st.pyplot(fig)

        with coly:
            # Optional: Scatter Plot with Trendline
            st.subheader("Scatter Plot of Predictions")

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.regplot(x=np.arange(len(predictions)), y=list(predictions.values()), ax=ax, scatter_kws={"s": 100}, color="red", marker="o", line_kws={"color": "blue"})

            ax.set_xticks(np.arange(len(predictions)))
            ax.set_xticklabels(list(predictions.keys()))
            ax.set_ylabel("Predicted RUL")
            ax.set_xlabel("Models")
            ax.set_title("Scatter Plot with Trendline for Predictions")

            st.pyplot(fig)
        
        # ----------------------------- BATTERY HEALTH CLASSIFICATION -------------------------------------
        st.write("##")

        pred_health_label = classify_battery_health(mean_prediction, data["RUL"].max())

        st.write(f"Battery Health Status: **{pred_health_label}**")

        # st.subheader(f"🔋 Predicted Battery Health: **{pred_health_label}**")

        # 🚨 Send Alerts if Condition is Critical
        if pred_health_label == "Critical":
            send_email_alert("Critical", "880prateekjain@gmail.com")
            send_sms_alert("Critical", "+918824307693")
            # send_email_alert(pred_health_label, recipient_email="beingjarvis@gmail.com")
            # send_sms_alert(pred_health_label, recipient_phone="+11234567890")

            st.warning("⚠️ Alert Sent! Battery Health is in **Critical Condition**. Please check your system.")
    # else:
    #     st.write("Please generation the RUL predictions first!")
            # ---------------------------------------------------------------
with tab2:
    st.header("Model Explainability with SHAP")
    # Load images
    image1 = Image.open("SHAP/firstplot.png") 
    image2 = Image.open("SHAP/secondplot.png")  
    # Create two columns
    col1, col2 = st.columns(2)
    # Display images in respective columns
    with col1:
        st.image(image1)
    with col2:
        st.image(image2)





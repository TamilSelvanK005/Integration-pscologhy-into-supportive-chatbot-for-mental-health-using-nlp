import streamlit as st
import altair as alt
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table, create_health_metrics_table, add_health_metrics, view_all_health_metrics, IST  # Import IST from track_utils
import cv2
from deepface import DeepFace
import random
from PIL import Image
import matplotlib.pyplot as plt
import io

# Load Model
pipe_lr = joblib.load(open("./models/stress.pkl", "rb"))

# Functions for text emotion prediction
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Function for image emotion detection
def detect_emotion(img_array):
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    result = DeepFace.analyze(img, actions=['emotion'])
    st.write(result)  # Debug: check the structure of the result
    emotion = result['dominant_emotion']
    return emotion

# Function to select an image based on detected emotion
def select_image_based_on_emotion(emotion, emotion_images):
    if emotion in emotion_images:
        images = emotion_images[emotion]
        selected_image = random.choice(images)
        return selected_image
    else:
        return None

# Dictionary of images categorized by emotions
emotion_images = {
    'happy': ['happy_image1.jpg', 'happy_image2.jpg'],
    'sad': ['sad_image1.jpg', 'sad_image2.jpg'],
    'angry': ['angry_image1.jpg', 'angry_image2.jpg'],
    'surprise': ['surprise_image1.jpg', 'surprise_image2.jpg'],
    # Add more emotions and images as needed
}

emotions_emoji_dict = {"anger": "Recomendation:Take deep breaths, step away from the situation, or try light physical activity", "joy": "Recomendation:Maintain this positive vibe! Engage in activities you love to keep your spirits up", "surprise": "Recomendation:Enjoy the unexpected moment and adapt to new experiences positively", "love":"Recomendation:Maintain balance and mindfulness in your daily activities.", "sadness": "Recomendation:Consider talking to a friend, listening to uplifting music, or engaging in a creative hobby.", "fear": "Recomendation:Try relaxation techniques like deep breathing and remind yourself that you are safe"}

# Main Application
def main():
    st.title("Health Stress Classifier App")
    menu = ["Home", "Monitor", "Image Emotion Detection", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    create_page_visited_table()
    create_emotionclf_table()
    create_health_metrics_table()  # Create table for health metrics

    if choice == "Home":
        add_page_visited_details("Home", datetime.now(IST))
        st.subheader("Health Stress Classifier in Text")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            temperature = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0, step=0.1)
            blood_pressure = st.text_input("Blood Pressure (e.g., 120/80)")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            add_prediction_details(raw_text, prediction, np.max(probability), datetime.now(IST))
            add_health_metrics(raw_text, temperature, blood_pressure, datetime.now(IST))  # Save health metrics

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))

                st.success("Health Metrics")
                st.write(f"Temperature: {temperature} °C")
                st.write(f"Blood Pressure: {blood_pressure}")

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)

    elif choice == "Monitor":
        add_page_visited_details("Monitor", datetime.now(IST))
        st.subheader("Monitor App")

        with st.expander("Page Metrics"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Page Name', 'Time of Visit'])
            st.dataframe(page_visited_details)

            pg_count = page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x='Page Name', y='Counts', color='Page Name')
            st.altair_chart(c, use_container_width=True)

            p = px.pie(pg_count, values='Counts', names='Page Name')
            st.plotly_chart(p, use_container_width=True)

        with st.expander('Mental Health Stress Classifier Metrics'):
            df_emotions = pd.DataFrame(view_all_prediction_details(), columns=['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
            st.dataframe(df_emotions)

            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction', y='Counts', color='Prediction')
            st.altair_chart(pc, use_container_width=True)

        with st.expander('Health Metrics'):
            df_health_metrics = pd.DataFrame(view_all_health_metrics(), columns=['Rawtext', 'Temperature', 'Blood Pressure', 'Time_of_Visit'])
            st.dataframe(df_health_metrics)

            temp_chart = alt.Chart(df_health_metrics).mark_line().encode(x='Time_of_Visit', y='Temperature', color='Temperature')
            st.altair_chart(temp_chart, use_container_width=True)

            bp_chart = alt.Chart(df_health_metrics).mark_bar().encode(x='Time_of_Visit', y='Blood Pressure', color='Blood Pressure')
            st.altair_chart(bp_chart, use_container_width=True)

    elif choice == "Image Emotion Detection":
        add_page_visited_details("Image Emotion Detection", datetime.now(IST))
        st.subheader("Image Emotion Detection")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            with st.spinner('Analyzing the image...'):
                img_bytes = uploaded_file.read()
                img_array = np.frombuffer(img_bytes, np.uint8)
                
                try:
                    emotion = detect_emotion(img_array)
                    st.write(f"Detected emotion: {emotion}")
                    
                    # Display uploaded image
                    img = Image.open(io.BytesIO(img_bytes))
                    st.image(img, caption='Uploaded Image', use_column_width=True)
                    
                    # Display uploaded image with imshow
                    plt.imshow(img)
                    plt.title('Uploaded Image')
                    plt.axis('off')
                    st.pyplot()
                    
                    selected_image = select_image_based_on_emotion(emotion, emotion_images)
                    if selected_image:
                        st.image(selected_image, caption=f"Selected image for {emotion} emotion")
                    else:
                        st.write("No images available for the detected emotion.")
                except Exception as e:
                    st.error(f"Error detecting emotion: {e}")

    else:
        add_page_visited_details("About", datetime.now(IST))

        st.write("Welcome to the Mental Health Stress Classifier in Text App! This application utilizes the power of natural language processing and machine learning to analyze and identify stress in textual data.")

        st.subheader("Our Mission")

        st.write("At stress Detection in Text, our mission is to provide a user-friendly and efficient tool that helps individuals and organizations understand the emotional content hidden within text. We believe that stress play a crucial role in communication, and by uncovering these emotions, we can gain valuable insights into the underlying sentiments and attitudes expressed in written text.")

        st.subheader("How It Works")

        st.write("When you input text into the app, our system processes it and applies advanced natural language processing algorithms to extract meaningful features from the text. These features are then fed into the trained model, which predicts the stress associated with the input text. The app displays the detected stress, along with a confidence score, providing you with valuable insights into the emotional content of your text.")

        st.subheader("Key Features:")

        st.markdown("##### 1. Real-time Mental Health Stress Classifier ")

        st.write("Our app offers real-time Mental Health Stress Classifier, allowing you to instantly analyze the stress expressed in any given text. Whether you're analyzing customer feedback, social media posts, or any other form of text, our app provides you with immediate insights into the stress underlying the text.")

        st.markdown("##### 2. Confidence Score")

        st.write("Alongside the detected stress, our app provides a confidence score, indicating the model's certainty in its predictions. This score helps you gauge the reliability of the stress detection results and make more informed decisions based on the analysis.")

        st.markdown("##### 3. User-friendly Interface")

        st.write("We've designed our app with simplicity and usability in mind. The intuitive user interface allows you to effortlessly input text, view the results, and interpret the Mental health stress detected. Whether you're a seasoned data scientist or someone with limited technical expertise, our app is accessible to all.")

        st.subheader("Applications")

        st.markdown("""
          The Emotion Detection in Text App has a wide range of applications across various industries and domains. Some common use cases include:
          - Social media sentiment analysis
          - Customer feedback analysis
          - Market research and consumer insights
          - Brand monitoring and reputation management
          - Content analysis and recommendation systems
          """)

if __name__ == '__main__':
    main()

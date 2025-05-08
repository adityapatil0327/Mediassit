import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Sample Data
data = {
    'Symptom': ['fever', 'headache', 'cough', 'nausea', 'fatigue'],
    'Medicine': ['Paracetamol', 'Ibuprofen', 'Cough Syrup', 'Anti-nausea drugs', 'Vitamin B12']
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
symptom_matrix = vectorizer.fit_transform(df['Symptom'])

prayas_store_data = {
    'store_name': 'Prayas Medical',
    'location': 'Ashta, Maharashtra',
    'latitude': 16.9430558,
    'longitude': 74.415467,
    'available_medicines': ['Paracetamol', 'Ibuprofen', 'Cough Syrup', 'Vitamin B12'],
    'google_maps_link': 'https://www.google.com/maps?q=16.9430558,74.415467'
}

def recommend_medicine(input_symptom):
    input_vector = vectorizer.transform([input_symptom])
    similarity_scores = cosine_similarity(input_vector, symptom_matrix)
    most_similar_idx = similarity_scores.argmax()
    return df.iloc[most_similar_idx]['Medicine']

def check_prayas_medical_availability(medicine):
    if medicine in prayas_store_data['available_medicines']:
        return f"✅ {medicine} is available at Prayas Medical."
    else:
        return f"❌ {medicine} is NOT available at Prayas Medical."

# Streamlit UI
st.title("💊 Medicine Recommender App")

symptom = st.text_input("Enter your symptom:")

if symptom:
    recommended_medicine = recommend_medicine(symptom.lower())
    st.subheader(f"Recommended Medicine: {recommended_medicine}")
    
    availability_msg = check_prayas_medical_availability(recommended_medicine)
    st.write(availability_msg)
    
    st.map({'lat': [prayas_store_data['latitude']], 'lon': [prayas_store_data['longitude']]})
    st.markdown(f"[🗺️ Click here to view Prayas Medical on Google Maps]({prayas_store_data['google_maps_link']})")

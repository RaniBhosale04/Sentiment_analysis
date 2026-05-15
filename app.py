import streamlit as st
import pickle
import numpy as np

# Load the trained model
# Using st.cache_resource ensures the model is only loaded once, saving memory
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("🔮 Positive/Negative Classifier")
st.write("Enter the values for the 8 features to get a prediction.")

# Create input fields for the 8 features
st.header("Input Features")

# Organizing inputs into two columns for a cleaner UI
col1, col2 = st.columns(2)

with col1:
    f1 = st.number_input("Feature 1", value=0.0)
    f2 = st.number_input("Feature 2", value=0.0)
    f3 = st.number_input("Feature 3", value=0.0)
    f4 = st.number_input("Feature 4", value=0.0)

with col2:
    f5 = st.number_input("Feature 5", value=0.0)
    f6 = st.number_input("Feature 6", value=0.0)
    f7 = st.number_input("Feature 7", value=0.0)
    f8 = st.number_input("Feature 8", value=0.0)

# Prediction execution
if st.button("Predict"):
    # Group inputs into a 2D numpy array (1 sample, 8 features)
    input_data = np.array([[f1, f2, f3, f4, f5, f6, f7, f8]])
    
    # Make the prediction
    prediction = model.predict(input_data)
    
    # Display the result (matching your model's 'positive'/'negative' classes)
    st.subheader("Result:")
    if prediction[0] == 'positive':
        st.success("🌟 The predicted class is: **Positive**")
    else:
        st.error("📉 The predicted class is: **Negative**")

# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np

def load_model():
    """Load the trained model and its metadata"""
    with open('zeolite_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def main():
    st.set_page_config(page_title="Zeolite Synthesis Predictor", layout="wide")
    
    # Add title and description
    st.title("🔮 Zeolite Synthesis Prediction")
    st.markdown("""
    This app predicts the success of zeolite synthesis experiments based on input parameters.
    
    **Prediction Classes:**
    - **Class 0**: Failed experiment (amorphous, mixed, dense, or layered phases)
    - **Class 1**: Successful experiment (pure zeolite phase)
    """)
    
    try:
        # Load model and metadata
        model_data = load_model()
        pipeline = model_data['pipeline']
        numerical_columns = model_data['numerical_columns']
        categorical_columns = model_data['categorical_columns']
        
        # Create two columns for input parameters
        col1, col2 = st.columns(2)
        
        # Dictionary to store user inputs
        input_data = {}
        
        with col1:
            st.subheader("Composition Parameters")
            input_data['SiO2'] = st.number_input(
                'SiO2 Amount',
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                help='Silicon dioxide content'
            )
            
            input_data['NaOH'] = st.number_input(
                'NaOH Amount',
                min_value=0.0,
                max_value=5.0,
                value=0.5,
                help='Sodium hydroxide content'
            )
            
            input_data['SDA'] = st.number_input(
                'SDA Amount',
                min_value=0.0,
                max_value=5.0,
                value=0.2,
                help='Structure-Directing Agent content'
            )
            
            input_data['B2O3'] = st.number_input(
                'B2O3 Amount',
                min_value=0.0,
                max_value=2.0,
                value=0.1,
                help='Boron oxide content'
            )
            
            input_data['H2O'] = st.number_input(
                'H2O Amount',
                min_value=10.0,
                max_value=100.0,
                value=30.0,
                help='Water content'
            )
        
        with col2:
            st.subheader("Process Parameters")
            input_data['temperature\n(°C)'] = st.slider(
                'Temperature (°C)',
                min_value=100,
                max_value=200,
                value=150,
                help='Synthesis temperature'
            )
            
            input_data['seed \namount'] = st.number_input(
                'Seed Amount',
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                help='Amount of seed crystals'
            )
            
            input_data['fd'] = st.number_input(
                'Framework Density (FD)',
                min_value=10.0,
                max_value=25.0,
                value=18.0,
                help='Framework density of the target zeolite'
            )
            
            input_data['seed'] = st.selectbox(
                'Seed Type',
                options=['Type A', 'Type B', 'None'],
                help='Type of seed crystals used'
            )
            
            input_data['si/al\n(ICP-AES)'] = st.selectbox(
                'Si/Al Ratio',
                options=['infy', '10', '30', '29', '15'],
                help='Silicon to Aluminum ratio (infy represents infinite ratio)'
            )
        
        # Add a predict button
        if st.button('Predict Synthesis Outcome'):
            # Create DataFrame from input
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = pipeline.predict(input_df)
            probability = pipeline.predict_proba(input_df)
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            # Create three columns for the results
            result_col1, result_col2, result_col3 = st.columns([2,2,1])
            
            with result_col1:
                if prediction[0] == 1:
                    st.success("🎯 Predicted Outcome: SUCCESSFUL")
                    st.write("The synthesis is predicted to result in a pure zeolite phase.")
                else:
                    st.error("❌ Predicted Outcome: FAILED")
                    st.write("The synthesis is predicted to result in amorphous, mixed, dense, or layered phases.")
            
            with result_col2:
                st.write("### Prediction Confidence")
                confidence = probability[0][prediction[0]]
                st.progress(confidence)
                st.write(f"Confidence: {confidence:.2%}")
            
            with result_col3:
                st.write("### Probabilities")
                st.write("Failure (0):", f"{probability[0][0]:.2%}")
                st.write("Success (1):", f"{probability[0][1]:.2%}")
        
    except FileNotFoundError:
        st.error("Error: Model file 'zeolite_model.pkl' not found!")
        st.write("Please ensure the model file is in the same directory as this application.")
    
    # Add footer with additional information
    st.markdown("---")
    st.markdown("""
    **Note:** This is a machine learning model prediction and should be used as a guide only. 
    The actual synthesis outcome may vary based on other experimental conditions and factors 
    not included in this prediction model.
    """)

if __name__ == '__main__':
    main()
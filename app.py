import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="NHANES CVD Longitudinal Study", layout="wide")

# 2. Load Assets
@st.cache_resource
def load_assets():
    # Ensure v2.pkl exists from the step above
    df = pd.read_csv('NHANES_1999_2023_Master_Clean.csv')
    model = joblib.load('cvd_prediction_model_v2.pkl')
    return df, model

df, model = load_assets()

# 3. Sidebar Navigation
st.sidebar.title("Study Navigation")
view = st.sidebar.radio("Go to", ["Trend Analysis", "Risk Predictor"])

if view == "Trend Analysis":
    st.title("Cardiovascular Health Trends (1999-2023)")
    st.write("Analysis of physiological shifts in the U.S. population across 20 years.")
    
    df['Phase'] = df['Cycle_Year'].apply(
        lambda x: 'Post-Pandemic (2021-23)' if '2021' in str(x) else 'Pre-Pandemic (1999-2018)'
    )
    
    feature = st.selectbox("Select Bio-marker", ['feat_bmi', 'feat_systolic_bp', 'feat_glyco_hemoglobin'])
    fig = px.histogram(df, x=feature, color="Phase", barmode="overlay", marginal="box",
                       title=f"Distribution Shift in {feature}")
    st.plotly_chart(fig, use_container_width=True)

elif view == "Risk Predictor":
    st.title("Personalized CVD Risk Assessment")
    st.markdown("Clinical risk calculation based on longitudinal NHANES data and advanced XGBoost modeling.")

    with st.form("clinical_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Demographics")
            age = st.slider("Age", 20, 85, 45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            bmi = st.number_input("BMI (kg/m2)", 15.0, 50.0, 25.0)
            smoking = st.selectbox("Smoking History (100+ cigarettes)", ["No", "Yes"])
        with col2:
            st.subheader("Clinical Markers")
            sys_bp = st.number_input("Systolic BP", 90, 200, 120)
            dia_bp = st.number_input("Diastolic BP", 60, 120, 80)
            a1c = st.number_input("A1c %", 4.0, 15.0, 5.5)
            tot_chol = st.number_input("Total Cholesterol", 100, 400, 190)
            hdl = st.number_input("HDL Cholesterol", 20, 100, 50)
            crp = st.number_input("C-Reactive Protein (CRP)", 0.0, 10.0, 0.5)

        submit = st.form_submit_button("Calculate Risk Score")

    if submit:
        # EXACT column order required by model v2
        feature_order = [
            'demo_age', 'demo_gender', 'feat_bmi', 'feat_crp', 'feat_diastolic_bp', 
            'feat_ever_smoked', 'feat_glyco_hemoglobin', 'feat_hdl', 'feat_systolic_bp', 
            'feat_total_cholesterol', 'feat_pulse_pressure', 'feat_metabolic_index', 
            'feat_age_bmi'
        ]

        # 1. Map Inputs
        user_df = pd.DataFrame({
            'demo_age': [age], 'demo_gender': [1 if gender == "Male" else 0],
            'feat_bmi': [bmi], 'feat_crp': [crp], 'feat_diastolic_bp': [dia_bp],
            'feat_ever_smoked': [1 if smoking == "Yes" else 2],
            'feat_glyco_hemoglobin': [a1c], 'feat_hdl': [hdl],
            'feat_systolic_bp': [sys_bp], 'feat_total_cholesterol': [tot_chol]
        })

        # 2. Add Engineered Features
        user_df['feat_pulse_pressure'] = user_df['feat_systolic_bp'] - user_df['feat_diastolic_bp']
        user_df['feat_metabolic_index'] = user_df['feat_bmi'] * user_df['feat_glyco_hemoglobin']
        user_df['feat_age_bmi'] = user_df['demo_age'] * user_df['feat_bmi']
        user_df['feat_lipid_ratio'] = user_df['feat_total_cholesterol'] / user_df['feat_hdl']

        # 3. Align and Predict
        user_df = user_df[feature_order]
        risk_prob = model.predict_proba(user_df)[:, 1][0]
        
        st.divider()
        st.subheader(f"Calculated Risk Probability: {risk_prob*100:.1f}%")

        # 4. What-If Scenario Analysis
        st.subheader("Scenario Analysis")
        scenario_df = user_df.copy()
        scenario_df['feat_systolic_bp'] = 120
        scenario_df['feat_pulse_pressure'] = 120 - scenario_df['feat_diastolic_bp']
        
        scenario_risk = model.predict_proba(scenario_df)[:, 1][0]
        reduction = (risk_prob - scenario_risk) * 100
        
        c1, c2 = st.columns(2)
        c1.metric("Current Risk", f"{risk_prob*100:.1f}%")
        c2.metric("Target Risk (BP 120)", f"{scenario_risk*100:.1f}%", delta=f"-{reduction:.1f}%")
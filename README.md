# Cardiovascular Disease Risk Prediction: A 24-Year Longitudinal Study (1999-2023)

## Project Overview
This project analyzes the evolution of cardiovascular health in the United States over two decades, utilizing the CDC's **National Health and Nutrition Examination Survey (NHANES)** data. The study harmonizes data from 12 distinct survey cycles to build a robust machine learning pipeline capable of predicting CVD risk in a post-pandemic landscape.



## Key Technical Achievements
* **Large-Scale Data Integration:** Harmonized and merged over 100 individual files, representing **66,378 adult participants** from 1999 to 2023.
* **Feature Engineering:** Developed advanced physiological interaction markers, including **Metabolic Index**, **Pulse Pressure**, and **Age-BMI interaction terms**.
* **High-Performance Modeling:** Utilized **XGBoost** to achieve an **AUC-ROC of 0.80**, demonstrating strong predictive power on imbalanced medical data.
* **Temporal Validation:** Validated the model using a chronological split, training on pre-2018 data and testing on the **2021-2023 post-pandemic cohort** to ensure model stability across shifting population health profiles.

## Dataset & Harmonization
The study integrates four primary data domains:
1. **Demographics:** Age, Gender.
2. **Examinations:** BMI, Systolic and Diastolic Blood Pressure.
3. **Laboratory:** Total Cholesterol, HDL, Glycohemoglobin (A1c), and C-Reactive Protein (CRP).
4. **Questionnaires:** Smoking history and self-reported medical history (Heart Attack/Stroke).



## Predictive Dashboard
The project includes a **Streamlit Dashboard** that features:
* **Trend Analysis:** Visualizing "Dataset Drift" and health distribution shifts between pre- and post-pandemic eras.
* **Risk Predictor:** An interactive clinical calculator for individual risk assessment.
* **What-If Analysis:** Real-time modeling of how reducing specific risk factors (e.g., Blood Pressure) impacts overall CVD probability.

## Technologies Used
* **Python** (Pandas, NumPy, Scikit-Learn, XGBoost)
* **Visualization:** Plotly, Seaborn, Matplotlib
* **Deployment:** Streamlit
* **Data Source:** CDC NHANES (1999-2023)

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the dashboard: `streamlit run app.py`.

---


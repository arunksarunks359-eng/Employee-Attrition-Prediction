import streamlit as st
import pandas as pd
import pickle

# Load Model
with open("employee_attrition_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Employee Attrition Predictor")

st.title("Employee Attrition Prediction")

# Education Level
education_options = {
    "High School": 1,
    "Diploma": 2,
    "Bachelor": 3,
    "Master": 4,
    "PhD": 5
}

education_level = education_options[
    st.selectbox(
        "Education Level",
        list(education_options.keys())
    )
]

# Ratings
rating_map = {
    "Very Low": 1,
    "Low": 2,
    "Medium": 3,
    "High": 4,
    "Very High": 5
}

job_satisfaction = rating_map[
    st.selectbox(
        "Job Satisfaction",
        list(rating_map.keys())
    )
]

work_life_balance = rating_map[
    st.selectbox(
        "Work Life Balance",
        list(rating_map.keys())
    )
]

# Numeric Inputs
monthly_income = st.number_input(
    "Monthly Income",
    min_value=0.0
)

monthly_expenses = st.number_input(
    "Monthly Expenses",
    min_value=0.0
)

performance_rating = st.number_input(
    "Performance Rating",
    min_value=0.0
)

overtime_hours = st.number_input(
    "Overtime Hours",
    min_value=0.0
)

manager_rating = st.slider(
    "Manager Rating",
    1.0,
    5.0,
    3.0
)

absenteeism_days = st.number_input(
    "Absenteeism Days",
    min_value=0
)

job_hopping_count = st.number_input(
    "Job Hopping Count",
    min_value=0
)

salary_hike_percentage = st.number_input(
    "Salary Hike Percentage",
    min_value=0.0
)

employee_engagement_score = st.number_input(
    "Employee Engagement Score",
    min_value=0.0
)

work_stress_score = st.slider(
    "Work Stress Score",
    1,
    10,
    5
)

# Prediction
if st.button("Predict Attrition"):

    input_data = pd.DataFrame({
        'Education_Level': [education_level],
        'Monthly_Income': [monthly_income],
        'Monthly_Expenses': [monthly_expenses],
        'Job_Satisfaction': [job_satisfaction],
        'Work_Life_Balance': [work_life_balance],
        'Performance_Rating': [performance_rating],
        'Overtime_Hours': [overtime_hours],
        'Manager_Rating': [manager_rating],
        'Absenteeism_Days': [absenteeism_days],
        'Job_Hopping_Count': [job_hopping_count],
        'Salary_Hike_Percentage': [salary_hike_percentage],
        'Employee_Engagement_Score': [employee_engagement_score],
        'Work_Stress_Score': [work_stress_score]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ Employee Likely To Leave ({probability:.2%})"
        )
    else:
        st.success(
            f"✅ Employee Likely To Stay ({1-probability:.2%})"
        )

    st.write(
        f"Attrition Probability: {probability:.2%}"
    )


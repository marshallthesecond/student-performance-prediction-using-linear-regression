from pathlib import Path

import streamlit as st

from src.data_preprocessing import load_and_clean_data
from src.predict import grade_letter, predict_gpa, train_app_model, what_if

DATA_PATH = Path(__file__).resolve().parent / "data" / "Student_performance_data.csv"

PARENTAL_SUPPORT_LABELS = {0: "None", 1: "Low", 2: "Moderate", 3: "High", 4: "Very high"}

st.set_page_config(page_title="Student GPA Predictor", page_icon="📚", layout="wide")

# the model is trained once when the app starts and reused for every visitor
@st.cache_resource
def load_model():
    df = load_and_clean_data(DATA_PATH)
    return train_app_model(df)

model, rmse = load_model()

st.title("📚 Student GPA Predictor")
st.write("Enter your study habits and activities to get a predicted GPA. The prediction updates as you change the inputs.")

inputs_column, results_column = st.columns(2, gap="large")

with inputs_column:
    st.subheader("Your details")

    study_time = st.slider("Study time per week (hours)", 0.0, 20.0, 10.0, step=0.5)
    absences = st.slider("Absences this school year", 0, 30, 5)
    parental_support = st.select_slider(
        "Parental support",
        options=list(PARENTAL_SUPPORT_LABELS),
        value=2,
        format_func=PARENTAL_SUPPORT_LABELS.get,
    )

    st.write("Do you take part in...")
    tutoring = st.toggle("Tutoring")
    extracurricular = st.toggle("Extracurricular activities")
    sports = st.toggle("Sports")
    music = st.toggle("Music")

inputs = {
    "StudyTimeWeekly": study_time,
    "Absences": absences,
    "Tutoring": int(tutoring),
    "ParentalSupport": parental_support,
    "Extracurricular": int(extracurricular),
    "Sports": int(sports),
    "Music": int(music),
}

gpa = predict_gpa(model, inputs)

with results_column:
    st.subheader("Your prediction")

    with st.container(border=True):
        gpa_column, grade_column = st.columns(2)
        gpa_column.metric("Predicted GPA", f"{gpa:.2f}")
        grade_column.metric("Grade", grade_letter(gpa))
        st.progress(gpa / 4.0)
        st.caption(
            f"Likely range: {max(gpa - rmse, 0.0):.2f} – {min(gpa + rmse, 4.0):.2f} "
            f"(the model is typically off by about {rmse:.2f} GPA points)"
        )

    st.subheader("What could raise it")
    suggestions = what_if(model, inputs)
    if suggestions:
        for description, new_gpa, gain in suggestions:
            st.write(f"**{description}** → {new_gpa:.2f} (+{gain:.2f})")
    elif gpa >= 4.0:
        st.write("Your predicted GPA is already at the top of the scale. 🎉")
    else:
        st.write("No single small change moves this prediction much. Cutting absences has by far the biggest effect.")

with st.expander("About this model"):
    st.write(
        "This is a linear regression trained on the *Students Performance Dataset* by Rabie El Kharoua "
        f"(2,392 students). It explains about 95% of the variance in GPA in that dataset. "
        "Absences have by far the biggest effect: each one lowers the predicted GPA by about 0.1."
    )
    st.write(
        "The dataset appears to be generated from a near-linear formula, so the prediction shows the "
        "patterns in that data, not a guarantee for a real student. Effects such as joining a sport "
        "are associations in the data, not proven causes."
    )
    st.write("Nothing you enter is stored.")

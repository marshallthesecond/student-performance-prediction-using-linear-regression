import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

from src.train_evaluate_model import RANDOM_STATE

# the app only asks for the features that actually affect GPA
# Age, Gender, Ethnicity, ParentalEducation and Volunteering have practically
# zero effect (see README), and leaving them out gives the same cross validated R²,
# so there is no reason to ask students for demographic data
APP_FEATURES = [
    "StudyTimeWeekly",
    "Absences",
    "Tutoring",
    "ParentalSupport",
    "Extracurricular",
    "Sports",
    "Music",
]

# GradeClass bins from the dataset description
GRADE_BINS = [(3.5, "A"), (3.0, "B"), (2.5, "C"), (2.0, "D"), (0.0, "F")]

def train_app_model(df, target_column="GPA"):
    X = df[APP_FEATURES]
    y = df[target_column]

    # cross validated RMSE is used to show how far off a prediction typically is
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rmse = -cross_val_score(LinearRegression(), X, y, cv=cv, scoring="neg_root_mean_squared_error").mean()

    # the final model is trained on all the data, since it was already evaluated above
    model = LinearRegression()
    model.fit(X, y)

    return model, rmse

def predict_gpa(model, inputs):
    X = pd.DataFrame([inputs])[APP_FEATURES]
    # GPA cannot go outside 0.0 - 4.0, even if the linear formula does
    return float(min(max(model.predict(X)[0], 0.0), 4.0))

def grade_letter(gpa):
    for threshold, letter in GRADE_BINS:
        if gpa >= threshold:
            return letter
    return "F"

# small changes a student could make, and the predicted GPA after each one
def what_if(model, inputs):
    scenarios = []

    if inputs["Absences"] > 0:
        fewer = min(5, inputs["Absences"])
        scenarios.append((f"{fewer} fewer absences", {"Absences": inputs["Absences"] - fewer}))
    if inputs["StudyTimeWeekly"] < 20:
        more = min(5, 20 - inputs["StudyTimeWeekly"])
        scenarios.append((f"{more:g} more hours of study per week", {"StudyTimeWeekly": inputs["StudyTimeWeekly"] + more}))
    if not inputs["Tutoring"]:
        scenarios.append(("Getting tutoring", {"Tutoring": 1}))
    for feature, label in [("Extracurricular", "an extracurricular activity"), ("Sports", "a sport"), ("Music", "music")]:
        if not inputs[feature]:
            scenarios.append((f"Joining {label}", {feature: 1}))

    current = predict_gpa(model, inputs)
    results = []
    for description, change in scenarios:
        new_gpa = predict_gpa(model, {**inputs, **change})
        if new_gpa - current >= 0.01:
            results.append((description, new_gpa, new_gpa - current))

    return sorted(results, key=lambda r: r[2], reverse=True)

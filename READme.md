This project consists of predictive models that try to estimate the grade/GPA of students from the dataset 📚 Students Performance Dataset 📚 by Rabie El Kharoua (Kaggle).


--- How to run ---:

```
pip install -r requirements.txt
python main.py            # train, evaluate, compare models, save results/metadata.json
python main.py --plots    # same, and also save the plots to results/plots/outputs
```

The script can be run from any directory.


--- Models ---:

1. Baseline (always predicts the mean GPA) - every model should beat it
2. Linear Regression - the main model
3. Ridge/Lasso (features standardized first)
4. Random Forest Regression

The Linear Regression model is evaluated on a 75/25 train/test split (random_state = 42).
All models are also compared with 5-fold cross validation, so the scores do not depend on one particular split.


-- Main metrics --:

1. RMSE
2. R²
3. MSE
4. MAE


--- Data ---:

data/Student_performance_data.csv - 2392 students, no missing values, no duplicates

Properties:
- **Student ID**
    - ranging from 1001 to 3392
- **Age**
    - from 15 to 18
- **Gender**
    - 0 = male
    - 1 = female
- **Ethnicity**
    - 0: Caucasian
    - 1: African American
    - 2: Asian
    - 3: Other
- **ParentalEducation**
    The education level of the parents, coded as follows:
    - 0: None
    - 1: High School
    - 2: Some College
    - 3: Bachelor's
    - 4: Higher
- **StudyTimeWeekly**
    - ranging from 0 to 20
- **Absences**
    - ranging from 0 to 30, during the school year
- **Tutoring**
    - 0 = NO
    - 1 = YES
- **ParentalSupport**
    The level of parental support, coded as follows:
    - 0: None
    - 1: Low
    - 2: Moderate
    - 3: High
    - 4: Very High
- **Extracurricular**
    - 0: No
    - 1: Yes
- **Sports**
    - 0: No
    - 1: Yes
- **Music**
    - 0: No
    - 1: Yes
- **Volunteering**
    - 0: No
    - 1: Yes
- **GPA  <— target variable**
    - ranging from 0.0 to 4.0
- **GradeClass**
    Classification of students' grades based on GPA:

    - 0: 'A' (GPA >= 3.5)
    - 1: 'B' (3.0 <= GPA < 3.5)
    - 2: 'C' (2.5 <= GPA < 3.0)
    - 3: 'D' (2.0 <= GPA < 2.5)
    - 4: 'F' (GPA < 2.0)

StudentID and GradeClass are dropped before training. GradeClass is calculated from GPA, so keeping it would leak the target into the features.


--- Results ---:
The results are recorded in results/metadata.json, and the plots inside the *results/plots/outputs* folder.

Linear Regression on the test set: R² = 0.953, RMSE = 0.195, MAE = 0.155

5-fold cross validation:

| Model             | RMSE  | MAE   | R²     |
|-------------------|-------|-------|--------|
| Baseline (mean)   | 0.915 | 0.772 | -0.001 |
| Linear Regression | 0.197 | 0.159 | 0.953  |
| Ridge             | 0.197 | 0.159 | 0.953  |
| Lasso             | 0.199 | 0.160 | 0.953  |
| Random Forest     | 0.240 | 0.191 | 0.931  |

Feature importance is reported as standardized coefficients (coefficient × feature standard deviation), i.e. the GPA change for a one-standard-deviation increase of the feature.
Raw coefficients are not comparable, because the features are measured on different ranges.
Absences is by far the most important feature (-0.85), followed by StudyTimeWeekly, ParentalSupport, Tutoring and the extracurricular activities. Age, Gender, Ethnicity, ParentalEducation and Volunteering have practically no effect.


--- Is the R² too high? ---:
R² - coefficient of determination - is the share of the variance in GPA that the model explains (1 = perfect fit, 0 = no better than predicting the mean). It is not an accuracy percentage: R² = 0.953 means the model explains 95.3% of the variance, and its typical error is about 0.2 GPA points (RMSE).

A score this high is unusual for predicting human behaviour, so the model was checked for leakage and split problems:
- 5-fold cross validation gives the same R² (0.948 - 0.960 across folds), so it is not a lucky train/test split
- no feature is calculated from GPA once GradeClass is dropped, so there is no target leakage
- Absences alone already gives R² ≈ 0.84, and 7 features give the full 0.953
- Linear Regression beats Random Forest, which means the relationship is almost exactly linear

The most likely explanation is that GPA in this dataset was generated as a (nearly) linear formula of the features - roughly -0.10 GPA per absence and +0.25 for tutoring. The model is recovering that formula, so its score should not be taken as how well GPA can be predicted for real students.

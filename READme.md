This project consists of predictive model that try to estimate the grade/GPA of students from the dataset 📚 Students Performance Dataset 📚 by Rabie El Kharoua at https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset


--- Models ---:

1. Linear Regression
2. Ridge/Lasso
3. Random Forest Regression


-- Main metrics --:

1. RMSE
2. R²
3. MSE
4. MAE


--- Data ---:

Student_performance_data_.csv

Porperties: 
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
- **Sport**
    - 0: No
    - 1: Yes
- **Music**
    - 0: No
    - 1: Yes
- **Volunteering**
    - 0: Yes
    - 1: No
- **GPA  <— target variable**
    - ranging from 0.0 to 4.0
- **GradeClass**
    Classification of students' grades based on GPA:
    
    - 0: 'A' (GPA >= 3.5)
    - 1: 'B' (3.0 <= GPA < 3.5)
    - 2: 'C' (2.5 <= GPA < 3.0)
    - 3: 'D' (2.0 <= GPA < 2.5)
    - 4: 'F' (GPA < 2.0)



--- Results ---:
The results of the prediction is recorded in metadate.json file, and visualls inside *results/plots* folder.
Overall, model performs very well with the R^2 score of 0.9531


--- Future works to be done ---:
Even though the dataset is very clean and no outlier were detected in the dataset, model performs in a slightly suspicious way, which is R^2 score being too high for the human behaviour prediction model. 
In general, R^2 score - coefficient of determination - measures how well a linear regression model fits a dataset, where 1 determines a perfect fit and 0 meaning the model is performing poorly.
So in this case, if a student provides all the info required to preict for this model, the model will determine the student's GPA with 95.31% accuracy. It is highly suspicious, since the GPA depends on many things, within and without the ones considered in this very model.
That is why, the behaviour of the model and the dataset should be reanalyzed for data leakage, target leakage, overfitting, train-test split mistakes and maybe other components too
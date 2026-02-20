import pandas as pd

def load_and_clean_data(path):

    # reading the data and deleting unnecessary columns - StudentID and GradeClass
    df = pd.read_csv(path)

    df = df.drop(columns=["StudentID", "GradeClass"])

    return df

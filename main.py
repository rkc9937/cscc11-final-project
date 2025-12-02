import pandas as pd
import numpy as np
import pickle
import os
from data_format.format import format_dataframe, save_formatted_data, load_formatted_data
from data_format.severity import create_severity, load_severity_weights


if __name__ == "__main__":

    #init the dataset
    df = pd.read_csv('./data/mci.csv') 

    if not os.path.exists('./data/weights/severity_weights.pkl'):
        create_severity(df)

    severity_weights = load_severity_weights()

    formatted_df = format_dataframe(df)
    #saves in data folder
    save_formatted_data(formatted_df)





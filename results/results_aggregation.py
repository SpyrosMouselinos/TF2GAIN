import os
import pandas as pd



def main():
    # Get all files but this
    files_here = os.listdir('.')
    files_here = [f for f in files_here if '.csv' in f]

    for file in files_here:
        new_name = file.split('_')[0]
        new_prefix = file.split('_')[1]
        df = pd.read_csv(f'{file}')
        df1 = df[(df.index + 1) % 2 != 0]
        df1 = df1.describe()['Val_ROC']

        df1.to_csv(f'{new_name}_{new_prefix}'+'_results_gain.csv')


main()
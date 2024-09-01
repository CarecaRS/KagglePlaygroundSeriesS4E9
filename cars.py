# Kaggle Playground Challenges S4E9
#
# Objective: predict the sale values
# Metric: RMSE
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
%autoindent OFF

# Load the datasets
treino = pd.read_csv('Datasets/train.csv')
teste = pd.read_csv('Datasets/test.csv')

# Check for the NaN percentages (if any)
(treino.isnull().sum()/treino.shape[0])*100
(teste.isnull().sum()/teste.shape[0])*100


# Check for duplicated observations (if any)
treino.duplicated().sum()
teste.duplicated().sum()

###
# DATA WRANGLING
###
to_drop = ['id', 'clean_title', 'cylinders']
#
for df in [treino, teste]:  # same wrangling for both datasets
#
    objs = df.select_dtypes('object').columns.values
    df[objs] = df[objs].astype('category')
    df['model_year'] = df['model_year'].astype('category')
# About the 'engine' feature:
    # First, extracting the engine power
    motor_hp = []
    for i in df['engine']:
        if len(i.split('HP')) > 1:
            motor_hp.append(i.split('HP')[0])
        else:
            motor_hp.append(np.nan)
    df['motor_hp'] = motor_hp
    df['motor_hp'] = df['motor_hp'].fillna('Other')
#
# Second, extracting the engine volume, gathering the information
# based on the 'L' character
    motor_volume_temp = []
    for i in df['engine']:
        motor_volume_temp.append(i.split('L')[0])
#
        motor_volume = []
    for i in motor_volume_temp:
        if len(i.split('HP ')) > 1:  # Some obs return two items, just filtering
            motor_volume.append(i.split('HP ')[1])
        else:
            motor_volume.append(i.split('HP ')[0])
#
    df['engine_vol'] = motor_volume
# Third, extracting cylinder information
    cylinders_temp = []
    for i in df['engine']:
        if len(i.split('L ')) > 1:  # Some obs with two items again
            cylinders_temp.append(i.split('L ')[1])
        else:
            cylinders_temp.append(i.split('L ')[0])
    cylinders_temp = pd.DataFrame(cylinders_temp)
#
    cylinders = []
    for i in cylinders_temp[0]:
        cylinders.append(i.split(' Cylinder ')[0]  # First split, through 'Cylinder'
                         .split('Straight ')[-1]  # Second split, removes 'Straight'
                         .split('Flat ')[-1]  # Third split, removes 'Flat'
                         .split('HP')[-1]  # Fourth split, removes 'HP'
                         .split()[0])  # Last split, removes composite names
#
    df['cylinders'] = cylinders
#
# 'accident' feature to dummies
    df['accident'] = df['accident'].map(
            {'None reported': 0,
             'At least 1 accident or damage reported': 1}
            )
#

###
# FIRST APPROACH: dropping NaNs
###
treino = treino.dropna()
teste = teste.dropna()

treino.drop(to_drop, axis=1, inplace=True)
teste.drop(to_drop, axis=1, inplace=True)

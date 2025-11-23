# The goal of this file is to define severity weights for different crime categories,
# and different Jupyter notebooks will import this file to access the severity weights.


# To do this, we will look at Statiscs Canada dataset, we will only look at Ontario since we
# cannot detail down to city.

# Formula = average length of incarceration * rate of incarceration per crime

# Note : all rate of incarerations come from https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3510002701

import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('./data/mci.csv')


# Count distinct crime offences called MCI_CATEGORY, and print them, these will be the categories we 
# need to define severity weights for.
mci_categories = df['MCI_CATEGORY'].unique().tolist()
print(mci_categories)

SEVERITY_WEIGHTS = {
    'Assault': None,
    'Robbery': None,
    'Breaking and Enter': None,
    'Auto Theft': None,
    'Theft Over': None,
}



# Calculate the average length of incarceration for each crime category
avg_length_assault = np.array([])

# StatsCan has three types of assault, we want to get the average length of incarceration per type 
# and then average over the three types.

for i in range(3):
    adf = pd.read_csv(f'./data/weights/assault_{i}.csv')
    # We average the columns, and multiply by 30 to get days
    temp = np.array([])
    for col in adf.columns[1:]:
        temp = np.append(temp, adf[col].mean() * 30)
    avg_length_assault = np.append(avg_length_assault, temp.mean()) 

print('Average length of incarceration for the three types of assault over 2014-2024:', avg_length_assault.mean())

a_0_percentage_guilty = np.array([34, 37, 37, 34, 33, 32, 29, 29, 26, 28]).mean() / 100

a_1_percentage_guilty = np.array([50, 51, 50, 47, 47, 47, 42, 39, 38, 38]).mean() / 100

a_2_percentage_guilty = np.array([48, 47, 46, 46, 43, 43, 37, 36, 35, 35]).mean() / 100

a_percentage_guilty = np.array([a_0_percentage_guilty, a_1_percentage_guilty, a_2_percentage_guilty]).mean()

print('Average percentage of guilty verdicts for the three types of assault over 2014-2024:', a_percentage_guilty)
SEVERITY_WEIGHTS['Assault'] = avg_length_assault.mean() * a_percentage_guilty
print('Severity weight for assault:', SEVERITY_WEIGHTS['Assault'])
print('-' * 8)


##################################################

avg_length_robbery = np.array([])
rdf = pd.read_csv(f'./data/weights/robbery.csv')
for col in rdf.columns[1:]:
    avg_length_robbery = np.append(avg_length_robbery, rdf[col].mean() * 30)
print('Average length of incarceration for robbery over 2014-2024:', avg_length_robbery.mean())
r_percentage_guilty = np.array([50, 52, 52, 53, 53, 54, 50, 47, 50, 52]).mean() / 100
print('Average percentage of guilty verdicts for robbery over 2014-2024:', r_percentage_guilty)

SEVERITY_WEIGHTS['Robbery'] = avg_length_robbery.mean() * r_percentage_guilty
print('Severity weight for robbery:', SEVERITY_WEIGHTS['Robbery'])

##################################################
avg_length_burglary = np.array([])
bdf = pd.read_csv(f'./data/weights/bne.csv')
for col in bdf.columns[1:]:
    avg_length_burglary = np.append(avg_length_burglary, bdf[col].mean() * 30)
print('Average length of incarceration for brekaing and entering over 2014-2024:', avg_length_burglary.mean())
# Straight from StatsCan website, the percentage of guilty verdicts for breaking and entering, only one row copied
bdf_percentage_guilty = np.array([66, 69, 66, 65, 67, 67, 65, 63, 60, 61]).mean() / 100
print('Average percentage of guilty verdicts for breaking and entering over 2014-2024:', bdf_percentage_guilty)

SEVERITY_WEIGHTS['Breaking and Enter'] = avg_length_burglary.mean() * bdf_percentage_guilty

print('Severity weight for breaking and entering:', SEVERITY_WEIGHTS['Breaking and Enter'])


print('-' * 8)

##################################################

avg_length_motor_vehicle_theft = np.array([])
mvdf = pd.read_csv(f'./data/weights/motor_theft.csv')
for col in mvdf.columns[1:]:
    avg_length_motor_vehicle_theft = np.append(avg_length_motor_vehicle_theft, mvdf[col].mean() * 30)
print('Average length of incarceration for motor vehicle theft over 2014-2024:', avg_length_motor_vehicle_theft.mean())
mvt_percentage_guilty = np.array([57, 62, 65, 60, 60, 58, 54, 51, 55, 53]).mean() / 100
print('Average percentage of guilty verdicts for motor vehicle theft over 2014-2024:', mvt_percentage_guilty)

SEVERITY_WEIGHTS['Auto Theft'] = avg_length_motor_vehicle_theft.mean() * mvt_percentage_guilty

print('Severity weight for motor vehicle theft:', SEVERITY_WEIGHTS['Auto Theft'])
print('-' * 8)

##################################################

avg_length_theft = np.array([])
tdf = pd.read_csv(f'./data/weights/theft.csv')
for col in tdf.columns[1:]:
    avg_length_theft = np.append(avg_length_theft, tdf[col].mean() * 30)
print('Average length of incarceration for theft over 2014-2024:', avg_length_theft.mean())
t_percentage_guilty = np.array([44, 47, 49, 47, 47, 46, 29, 34, 30, 28]).mean() / 100
print('Average percentage of guilty verdicts for theft over 2014-2024:', t_percentage_guilty)
SEVERITY_WEIGHTS['Theft Over'] = avg_length_theft.mean() * t_percentage_guilty
print('Severity weight for theft over:', SEVERITY_WEIGHTS['Theft Over'])

###################################################

# Export the severity weights as a pickle file
with open('./data/weights/severity_weights.pkl', 'wb') as f:
    pickle.dump(SEVERITY_WEIGHTS, f)
print("Severity weights saved to './data/weights/severity_weights.pkl'")






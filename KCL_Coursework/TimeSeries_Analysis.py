import pandas as pd
import matplotlib.pyplot as plt

raw = pd.read_csv('datasets/traffic-data.csv', encoding='ISO-8859-1')
row_count = len(raw)
print('Number of instances ' + str(row_count))
new_data = raw.drop(columns=['Unnamed: 0'])
attributes = new_data.keys()
print(attributes)

values = new_data['CAMERA.ID'].unique()
print('Cameras : ' + str(values))

total_violations = 0
for value in values:
    result = new_data[new_data['CAMERA.ID'] == value]
    violations = result['VIOLATIONS'].sum()
    print ('Camera ' + str(value) + ' caught ' + str(violations) + ' violations')
    total_violations += violations
print('Total violations is ' + str(total_violations))

import time
def preprocess(x):
    (yyyy, mm, dd) = x.split('-')
    tm = time.struct_time((int(yyyy), int(mm), int(dd), 0, 0, 0, 0, 0, 0))
    ts = time.mktime(tm)
    return ts

new_data['new_date'] = new_data['date'].apply(preprocess)

camera = new_data[new_data['CAMERA.ID'] == 'CHI003']
plt.plot(camera['new_date'], camera['VIOLATIONS'], alpha=0.5, label='CHI003')
plt.legend(loc='upper left')
plt.show()

values = new_data['CAMERA.ID'].unique()
for value in values:
    camera = new_data[new_data['CAMERA.ID'] == value]
    plt.plot(camera['new_date'], camera['VIOLATIONS'], alpha=0.5, label=value)
plt.legend(loc='upper left')
plt.show()
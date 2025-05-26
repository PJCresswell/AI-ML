import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.random.randn(5)
my_series = pd.Series(data, index=['a', 'b', 'c', 'd', 'e'])
#my_series.plot()
#plt.show()

# Creating a series from a dict
d = {'a': 0., 'b': 1., 'c': 2.}
my_series = pd.Series(d)
#print(my_series.b)
#print(my_series[['b', 'c']])
squared_values = my_series ** 2

dates = pd.date_range('1/1/2000', periods=5)
time_series = pd.Series(data, index=dates)
#ax=time_series.plot()
#plt.show()

series_dict = {
    'x' : pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
    'y' : pd.Series([4., 5., 6., 7.], index=['a', 'b', 'c', 'd']),
    'z' : pd.Series([0.1, 0.2, 0.3, 0.4], index=['a', 'b', 'c', 'd'])
}
df = pd.DataFrame(series_dict)
#ax = df.plot()
#plt.show()
#print(df['x'])
#print(df['x']['b'])
#print(df.x.b)

# Slicing the data in a dataframe
slice = df['x'][['b', 'c']]
#print(slice)
#print(type(slice))
slice = df[['x', 'y']]
#print(slice)
#print(type(slice))
slice = df.loc[:, ['x', 'y']]
#print(slice)
slice = df.loc['b':'d', ['x', 'y']]
#print(slice)
slice = df.iloc[1:, 0:2]
#print(slice)
slice = df[df['x'] >= 2]
#print(slice)

result = df['x'] + df['y']
#print(result)

print(df.describe())
specific_result = df.describe()['x']['mean']

# Table joins
student = pd.DataFrame({
    'name': ['Smith', 'Brown', 'Phelps'],
    'student_number': [17, 8, 666]
})
grade_report = pd.DataFrame({
    'student_number': [17, 17, 8, 8, 8, 8],
    'section_identifier': [112, 119, 85, 92, 102, 135],
    'grade': ['B', 'A', 'A', 'A', 'B', 'A']
})
inner_join = student.merge(grade_report, on='student_number')
#print(inner_join)
left_outer_join = student.merge(grade_report, on='student_number', how='left')
#print(left_outer_join)

# Working with financial data
import yfinance as yf
data = yf.Ticker('GOOGL').history(period='1y')
print(data.head())
print(type(data.index[0]))
print(data.columns)
#ax = data['Volume']['2023-12-01':'2024-6-01'].plot()
#plt.show()
weekly_prices = data['Close'].resample('W').last()
print(weekly_prices.head())

weekly_rets = np.diff(np.log(weekly_prices))
print(type(weekly_rets))
weekly_rets_series = pd.Series(weekly_rets, index=weekly_prices.index[1:])
print(weekly_rets_series.head())
plt.plot(weekly_rets_series)
plt.show()
weekly_rets_series.hist()
plt.show()


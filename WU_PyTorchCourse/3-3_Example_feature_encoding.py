import pandas as pd

# Load the data
# The target is going to be the product column - what someone bought
df = pd.read_csv('https://data.heatonresearch.com/data/t81-558/jh-simple-dataset.csv', na_values=["NA", "?"],)
print(df)

# Start by converting the job code into dummy variables
# 33 different job codes, 33 dummy variables
dummies = pd.get_dummies(df["job"], prefix="job")
print(dummies.shape)
print(dummies)

# Now we drop the original job field and merge back into the original dataframe
df = pd.concat([df, dummies], axis=1)
df.drop("job", axis=1, inplace=True)
print(df)

# Same process for the area column
df = pd.concat([df, pd.get_dummies(df["area"], prefix="area")], axis=1)
df.drop("area", axis=1, inplace=True)
print(df)

# Finally fill in missing income values
med = df["income"].median()
df["income"] = df["income"].fillna(med)

# Now ready to be converted to Numpy for NN training
# We need to know the list of columns for x and y
print(list(df.columns))

# Drop product as this is the target. Also remove id as not useful
x_columns = df.columns.drop("product").drop("id")
print(list(x_columns))

# Generate the x and y
from sklearn import preprocessing

# Convert to numpy for Classification
x_columns = df.columns.drop("product").drop("id")
x = df[x_columns].values
le = preprocessing.LabelEncoder()
y = le.fit_transform(df["product"])
products = le.classes_
y = dummies.values

# Now put into a NN ...
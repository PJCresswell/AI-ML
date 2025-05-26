# Make use of a GPU or MPS (Apple) if one is available.  (see module 3.2)
import torch
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

import pandas as pd

df_sales_train = pd.read_csv("https://data.heatonresearch.com/wustl/CABI/demand-forecast/sales_train.csv")
df_items = pd.read_csv("https://data.heatonresearch.com/wustl/CABI/demand-forecast/items.csv")
df_resturant = pd.read_csv("https://data.heatonresearch.com/wustl/CABI/demand-forecast/resturants.csv")
df_sales_train.date = pd.to_datetime(df_sales_train.date, errors='coerce')

# Show the original data
import plotly.express as px
df_plot = df_sales_train[['date','item_count']].groupby(['date']).mean().reset_index()
fig = px.line(df_plot, x="date", y="item_count", title='RAW Sales by Date')
fig.show()

# Apply de-trending
from scipy import signal
df_plot.item_count = signal.detrend(df_plot.item_count)
fig = px.line(df_plot, x="date", y="item_count", title='RAW Sales by Date')
fig.show()

# Remove seasonality
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
df_plot = df_sales_train[['date','item_count']].groupby(['date']).mean()
# extrapolate_trend='freq',
adjustment = seasonal_decompose(df_plot.item_count, model='multiplicative') # , model='additive', period=7
# multiplicative
# additive
adjustment.plot()
pyplot.show()
print(adjustment.trend)
print(adjustment.seasonal)

# View the dataset flattened
df_plot2 = df_plot.copy()
df_plot2.item_count = df_plot2.item_count / adjustment.seasonal / adjustment.trend
fig = px.line(df_plot2.reset_index(), x="date", y="item_count", title='RAW Sales by Date')
fig.show()
df_adjustment = pd.DataFrame(adjustment.seasonal)
df_adjustment['trend'] = adjustment.trend
print(df_adjustment)
# Save down the adjustment table
df_adjustment.to_pickle("adjustment.pkl")

# Now that we've estimated seasonality and trend for the average of all items, we must apply this to the individual items
df_sales_adj = df_sales_train.merge(df_adjustment,right_index=True,left_on="date")
df_sales_adj.dropna(inplace=True)
df_sales_adj['adjust'] = df_sales_adj.item_count / df_sales_adj.seasonal / df_sales_adj.trend

# Compare the individual item plot to the previous example with seasonality still embeded
SINGLE_YEAR = 2020
df_sales_single_year = df_sales_adj[df_sales_train['date'].dt.year == SINGLE_YEAR]
df_plot = df_sales_single_year[['date','item_id','adjust']].groupby(['date','item_id']).mean().reset_index()
df_plot = df_plot.merge(df_items,left_on="item_id",right_on="id")[['date','adjust','name']]
fig = px.bar(df_plot, x='date', y='adjust',color="name", title=f'Item Sales by Date - {SINGLE_YEAR}')
fig.update_layout(bargap=0.0,bargroupgap=0.0)
fig.show()
import pandas
from iexfinance.stocks import get_historical_data
import datetime
api_key = ""


start = datetime.date(2010, 1, 2)
end = datetime.date(2017, 1, 1)
df = get_historical_data("MSFT", output_format='pandas', token=api_key, start=start, end=end)
# df = df.reindex(index=df.index[::-1])
print(df.head())
df.to_csv("DataFrame1.csv")

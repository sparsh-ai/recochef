import dns
import pandas as pd
from pymongo import MongoClient

client = MongoClient("mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db = client.get_database("OnlineGroceryDB")

for col in db.collection_names():
  cursor = db[col].find()
  pd.DataFrame(list(cursor)).to_csv('{}.csv'.format(col))

!zip retail_data.zip ./*.csv

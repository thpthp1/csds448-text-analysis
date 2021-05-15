import pandas as pd

df = pd.read_json('scrape-geomdash.json')

df.to_csv('clean/clean-geomdash.csv')
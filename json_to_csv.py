import pandas as pd

df = pd.read_json('scrape-coc.json')

df.to_csv('clean/clean-coc.csv')
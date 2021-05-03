from google_play_scraper import app, reviews_all, reviews, Sort
import datetime
import json

def default(o):
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()

results = reviews_all(
    'com.plarium.raidlegends',
    lang='en', # defaults to 'en'
    country='us', # defaults to 'us'
)
#print(results)
with open('scrape-raid.json', 'w') as f:
    json.dump(results, f, default=default)



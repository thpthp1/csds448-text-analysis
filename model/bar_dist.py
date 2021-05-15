import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("./model/sample_pred_geomdash.csv")

tagged_df = df[df["is_aggresive_IAP"] == True]
rating_count = []
for i in range(5):
    rating_count.append(len(tagged_df[tagged_df["score"] == i + 1]))
plt.bar([1,2,3,4,5], np.array(rating_count)/len(tagged_df))
plt.title("Rating score aggresive IAP review count out of total = {} reviews".format(len(tagged_df[tagged_df["score"] <= 5])))
plt.xlabel("Score")
plt.ylabel("Distribution")
plt.xticks([1,2,3,4,5])
plt.show()
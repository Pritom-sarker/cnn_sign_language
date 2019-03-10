from sklearn.externals import joblib
import numpy as np

loaded_model = joblib.load('data/01.pkl')
print(loaded_model.shape)

import  pandas as pd
li=loaded_model[0]
x=0
df=pd.DataFrame()
for i in li:
    df[x]=i


print(df.head())
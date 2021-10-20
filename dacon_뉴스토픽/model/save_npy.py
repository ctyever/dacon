import numpy as np
import pandas as pd
import re


datasets = pd.read_csv('./data/train_data.csv',  sep=',', 
                        index_col='index', header=0)

pred =  pd.read_csv('./data/test_data.csv',  sep=',', 
                        index_col='index', header=0)

def clean_text(sent):
      sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", sent)
      return sent_clean

datasets["cleaned_title"] = datasets["title"].apply(lambda x : clean_text(x))
pred["cleaned_title"]  = pred["title"].apply(lambda x : clean_text(x))

x = datasets.iloc[:, -1]
y = datasets.iloc[:, 1]
x_pred = pred.iloc[:, -1]

x = x.to_numpy()
y = y.to_numpy()
x_pred = x_pred.to_numpy()

np.save('./save/npy/x_data.npy', arr=x)
np.save('./save/npy/y_data.npy', arr=y)
np.save('./save/npy/x_pred_data.npy', arr=x_pred)

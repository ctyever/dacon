import numpy as np
import pandas as pd
import re
from konlpy.tag import Okt,Mecab


datasets = pd.read_csv('./data/train_data.csv',  sep=',', 
                        index_col='index', header=0)

pred =  pd.read_csv('./data/test_data.csv',  sep=',', 
                        index_col='index', header=0)

def clean_text(sent):
      sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", sent)
      return sent_clean

datasets["cleaned_title"] = datasets["title"].apply(lambda x : clean_text(x))
pred["cleaned_title"]  = pred["title"].apply(lambda x : clean_text(x))

# 형태소 분석기(Okt) 불러오기 
okt=Okt() 

# 조사, 어미, 구두점 제거
def func(text):
    clean = []
    for word in okt.pos(text, stem=True): #어간 추출
        if word[1] not in ['Josa', 'Eomi', 'Punctuation']: #조사, 어미, 구두점 제외 
            clean.append(word[0])  
    return " ".join(clean) 

datasets["cleaned_title"] = datasets["cleaned_title"].apply(lambda x : func(x))
pred["cleaned_title"] = pred["cleaned_title"].apply(lambda x : func(x))


x = datasets.iloc[:, -1]
y = datasets.iloc[:, 1]
x_pred = pred.iloc[:, -1]
# print(x)

x = x.to_numpy()
y = y.to_numpy()
# x_pred = x_pred.to_numpy()

np.save('./save/npy/x_okt_data.npy', arr=x)
np.save('./save/npy/y_okt_data.npy', arr=y)
np.save('./save/npy/x_okt_pred_data.npy', arr=x_pred)

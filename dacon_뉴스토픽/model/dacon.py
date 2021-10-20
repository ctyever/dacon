from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Flatten
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
import re
import matplotlib.pyplot as plt

# 1. 데이터 구성

datasets = pd.read_csv('./data/train_data.csv',  sep=',', 
                        index_col='index', header=0)

pred =  pd.read_csv('./data/test_data.csv',  sep=',', 
                        index_col='index', header=0)

# print(datasets)
# print(pred)

def clean_text(sent):
      sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", sent)
      return sent_clean

datasets["cleaned_title"] = datasets["title"].apply(lambda x : clean_text(x))
pred["cleaned_title"]  = pred["title"].apply(lambda x : clean_text(x))

# print(datasets)

x = datasets.iloc[:, -1]
y = datasets.iloc[:, 1]
x_pred = pred.iloc[:, -1]

x = x.to_numpy()
y = y.to_numpy()
x_pred = x_pred.to_numpy()
# print(x)
'''
['인천 핀란드 항공기 결항 휴가철 여행객 분통' '실리콘밸리 넘어서겠다 구글   조원 들여  전역 거점화'
 '이란 외무 긴장완화 해결책은 미국이 경제전쟁 멈추는 것' ... '게시판 키움증권      키움 영웅전 실전투자대회'
 '답변하는 배기동 국립중앙박물관장' '     한국인터넷기자상 시상식 내달  일 개최 특별상 김성후']
'''
# print(y)  # [4 4 4 ... 1 2 2]
# print(x_pred)
# print(x.shape, y.shape, x_pred.shape) # (45654,) (45654,) (9131,)
# print(np.unique(y)) # [0 1 2 3 4 5 6]
# print(type(x)) # <class 'numpy.ndarray'>


from tensorflow.keras.preprocessing.text import Tokenizer

token = Tokenizer(num_words=10000)
token.fit_on_texts(x)
x = token.texts_to_sequences(x)
x_pred = token.texts_to_sequences(x_pred)

# print(x_pred)
# # print(x)
# # # print(y)
# # print("뉴스기사의 최대길이 : ", max(len(i) for i in x_pred)) # 뉴스기사의 최대길이 :  13 / pred 뉴스기사의 최대길이 :  10
# # print("뉴스기사의 평균길이 : ", sum(map(len, x_pred)) / len(x_pred)) # 뉴스기사의 최대길이 :  6.623954089455469 / pred 뉴스기사의 평균길이 :  4.112912057824992
# # # plt.hist([len(s) for s in x], bins=50)
# # # plt.show()

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

pad_x = pad_sequences(x, padding='pre', maxlen=10)
x_pred = pad_sequences(x_pred, padding='pre', maxlen=10)
# print(pad_x.shape) # (45654, 10)
# # print(pad_x)
# print(x_pred[0])

x_train, x_test, y_train, y_test = train_test_split(pad_x, y, 
         train_size=0.8, shuffle=True, random_state=66)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train.shape, y_test.shape) # (36523, 7) (9131, 7)


# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding

model = Sequential()
                 # 단어사전의 개수                   단어수, 길이
model.add(Embedding(input_dim=10000, output_dim=11, input_length=10))
# input_length 안 쒀줘도 되는데 자동으로 인식하면서 None 으로 인식함
# model.add(Embedding(128, 77)) # input_dim 이 단어개수 보다 많으면 됨, 그런데 맞춰주는게 좋음
model.add(LSTM(32, activation='relu'))
model.add(Dense(7, activation='softmax'))

# print(x_train)

# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['acc'])

import time
start_time = time.time()
model.fit(x_train, y_train, epochs=4, batch_size=256, validation_split=0.1)
end_time = time.time() - start_time

# 4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]
print("acc : ", acc)
print("걸린시간 : ", end_time)

tmp_pred = model.predict(x_pred)
pred = np.argmax(tmp_pred, axis = 1)
# print(len(pred)) # 9131
# print('예측값 : ', pred) # ex) 예측값 :  [2 2 2 ... 1 1 6]
submission = pd.read_csv('./data/sample_submission.csv')
submission['topic_idx'] = pred
# print(submission.head())
submission.to_csv('./data/submission3.csv', index=False)

'''
1차
acc :  0.7425254583358765
2차
acc :  0.75008213520050
제출용(first)
3차
acc :  0.7524915337562
4차 ( submission3)
정답률 : 0.7161


'''

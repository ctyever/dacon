import numpy as np
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from icecream import ic
from sklearn.metrics import accuracy_score,log_loss
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import plot_model, to_categorical
PATH = './data/'
train = pd.read_csv('./data/train_data.csv', header=0)
test = pd.read_csv('./data/test_data.csv', header=0)
submission = pd.read_csv(PATH + "sample_submission.csv")
# null값 제거
# datasets_train = datasets_train.dropna(axis=0)
# datasets_test = datasets_test.dropna(axis=0)

# x = datasets_train.iloc[:, -2]
# y = datasets_train.iloc[:, -1]
# x_pred = datasets_test.iloc[:, -1]
train['doc_len'] = train.title.apply(lambda words: len(words.split()))

x_train = np.array([x for x in train['title']])
x_predict = np.array([x for x in test['title']])
y_train = np.array([x for x in train['topic_idx']])

def text_cleaning(docs):
    for doc in docs:
        doc = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", doc)
    return docs
x = text_cleaning(x_train)
x_predict = text_cleaning(x_predict)
# ic(x.shape) ic| x.shape: (45654,)

# 불용어 제거, 특수문자 제거
# import string
# def define_stopwords(path):
#     sw = set()
#     for i in string.punctuation:
#         sw.add(i)

#     with open(path, encoding='utf-8') as f:
#         for word in f:
#             sw.add(word)

#     return sw
# x = define_stopwords(x)

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()  
tokenizer.fit_on_texts(x)
sequences_train = tokenizer.texts_to_sequences(x)
sequences_test = tokenizer.texts_to_sequences(x_predict)

#리스트 형태의 빈값 제거  --> 양방향에서는 오류남..
# sequences_train = list(filter(None, sequences_train))
# sequences_test = list(filter(None, sequences_test))

#길이 확인
# x1_len = max(len(i) for i in sequences_train)
# ic(x1_len) # ic| x1_len: 11
# x_pred = max(len(i) for i in sequences_test)
# ic(x_pred) # ic| x_pred: 9

xx = pad_sequences(sequences_train, padding='pre', maxlen = 14)
# ic(xx.shape) ic| xx.shape: (42477, 11)
yy = pad_sequences(sequences_test, padding='pre', maxlen=14)

y = to_categorical(y_train)

from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(xx, y, train_size=0.7, shuffle=True, random_state=66)
# np.save('./_save/_npy/dacon_x_train2.npy', arr=x_train)
# np.save('./_save/_npy/dacon_y_train2.npy', arr=y_train)
# np.save('./_save/_npy/dacon_x_test2.npy', arr=x_test)
# np.save('./_save/_npy/dacon_y_test2.npy', arr=y_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, Bidirectional

# model = Sequential()
# model.add(Embedding(input_dim=101082, output_dim=77, input_length=11))
# model.add(LSTM(128, activation='relu'))
# model.add(Dense(64, activation= 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation= 'relu'))
# model.add(Dense(7, activation='softmax'))
model = Sequential([Embedding(101082, 200, input_length =14),
        tf.keras.layers.Bidirectional(LSTM(units = 32, return_sequences = True, activation='relu')),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(LSTM(units = 16, return_sequences = True, activation='relu')),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(LSTM(units = 8, activation='relu')),
        Dense(7, activation='softmax')    # 결과값이 0~4 이므로 Dense(5)
    ])
import datetime
import time

model.compile(loss= 'categorical_crossentropy', #여러개 정답 중 하나 맞추는 문제이므로 손실 함수는 categorical_crossentropy
              optimizer= 'adam',
              metrics = ['acc'])

date = datetime.datetime.now()
date_time = date.strftime('%m%d_%H%M')
path = './_save/mcp/'
info = '{epoch:02d}_{val_loss:.4f}'
filepath = ''.join([path, 'test', '_', date_time, '_', info, '.hdf5'])

es = EarlyStopping(monitor='val_acc', patience=50, mode='max', verbose=1)

##########################################################
import datetime
date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")

filepath = './_save/'
filename = '.{epoch:04d}-{val_loss:.4f}.hdf5'
modelpath = "".join([filepath, "dacon", date_time, "_", filename])
#################################################################

mcp = ModelCheckpoint(monitor='val_acc', mode='max', verbose=1, save_best_only=True,
                       filepath= modelpath) 
history = model.fit(xx, y, epochs=10, batch_size=512, validation_split= 0.1, callbacks=[es, mcp])


n_fold = 5  
seed = 66

cv = StratifiedKFold(n_splits = n_fold, shuffle=True, random_state=seed)

# 테스트데이터의 예측값 담을 곳 생성
test_y = np.zeros((yy.shape[0], 7))

# ##########################################################
# import datetime
# date = datetime.datetime.now()
# date_time = date.strftime("%m%d_%H%M")

# filepath = './_save/'
# filename = '.{epoch:04d}-{val_loss:.4f}.hdf5'
# modelpath = "".join([filepath, "dacon", date_time, "_", filename])
# #################################################################

# # 조기 종료 옵션 추가
# es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5,
#                    verbose=1, mode='min', baseline=None, restore_best_weights=True)
# cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', verbose=1, filepath=modelpath)
start_time = time.time()
for i, (i_trn, i_val) in enumerate(cv.split(xx, y_train), 1):
    # print(f'training model for CV #{i}')

    # model.fit(xx[i_trn], 
    #         to_categorical(y_train[i_trn]),
    #         validation_data=(xx[i_val], to_categorical(y_train[i_val])),
    #         epochs=5,
    #         batch_size=256,
    #         )     # 조기 종료 옵션
    
    model = load_model('./_save/dacon0809_1456_.0002-0.0080.hdf5')                    
    test_y += model.predict(yy) / n_fold  

topic = []
for i in range(len(test_y)):
    topic.append(np.argmax(test_y[i]))
end_time = time.time() - start_time
# loss = model.evaluate(x_test, y_test)
# ic('loss = ', loss[0])
# ic('acc = ', loss[1])
# ic('val_acc = ', loss[-1])
# ic('time taken(s) = ', end_time)

# submission['topic_idx'] = topic
# submission.to_csv(PATH + 'submission16.csv',index = False)

# ic| 'loss = ', loss[0]: 0.7068963646888733
# ic| 'acc = ', loss[1]: 0.7655690908432007
# ic| 'time taken(s) = ', end_time: 190.59102249145508

# Epoch 5/5
# 72/72 [==============================] - 17s 237ms/step - loss: 4.0520e-04 - accuracy: 0.9999 - val_loss: 0.0023 - val_accuracy: 0.9996

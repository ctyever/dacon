import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Flatten
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
import re
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Flatten
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# 1. 데이터 구성

x = np.load('./save/npy/x_data.npy', allow_pickle=True)
y = np.load('./save/npy/y_data.npy', allow_pickle=True)
x_pred = np.load('./save/npy/x_pred_data.npy', allow_pickle=True)

# print(x, x_pred)

from tensorflow.keras.preprocessing.text import Tokenizer

token = Tokenizer(num_words=2000)
token.fit_on_texts(x)
x = token.texts_to_sequences(x)
x_pred = token.texts_to_sequences(x_pred)

# print(len(x), len(x_pred)) # 45654 9131

from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical

pad_x = pad_sequences(x, padding='pre', maxlen=14)
x_pred = pad_sequences(x_pred, padding='pre', maxlen=14)

from tensorflow.keras.utils import plot_model, to_categorical
train_y = to_categorical(y)
# print(y)
# print(y.shape) # (45654, 7)

# 2. 모델 구성

#파라미터 설정
vocab_size = 2000 
embedding_dim = 200  
max_length = 14    # 위에서 그래프 확인 후 정함
padding_type='post'
#oov_tok = "<OOV>"

model3 = Sequential([Embedding(vocab_size, embedding_dim, input_length =max_length),
        tf.keras.layers.Bidirectional(LSTM(units = 64, return_sequences = True)),
        tf.keras.layers.Bidirectional(LSTM(units = 64, return_sequences = True)),
        tf.keras.layers.Bidirectional(LSTM(units = 64)),
        Dense(7, activation='softmax')    # 결과값이 0~4 이므로 Dense(5)
    ])
    
model3.compile(loss= 'categorical_crossentropy', #여러개 정답 중 하나 맞추는 문제이므로 손실 함수는 categorical_crossentropy
              optimizer= 'adam',
              metrics = ['accuracy']) 
# model3.summary()
'''
Model: "sequential"
_________________________________________________________________        
Layer (type)                 Output Shape              Param #
=================================================================        
embedding (Embedding)        (None, 14, 200)           400000
_________________________________________________________________        
bidirectional (Bidirectional (None, 14, 128)           135680
_________________________________________________________________        
bidirectional_1 (Bidirection (None, 14, 128)           98816
_________________________________________________________________        
bidirectional_2 (Bidirection (None, 128)               98816
_________________________________________________________________        
dense (Dense)                (None, 7)                 903
=================================================================        
Total params: 734,215
Trainable params: 734,215
Non-trainable params: 0
'''


# 모델 실행해보기
history = model3.fit(pad_x, train_y, epochs=50, batch_size=100, validation_split= 0.2) 

# 계층 교차 검증
n_fold = 5  
seed = 42

cv = StratifiedKFold(n_splits = n_fold, shuffle=True, random_state=seed)

# 테스트데이터의 예측값 담을 곳 생성
test_y = np.zeros((x_pred.shape[0], 7))

# 조기 종료 옵션 추가
es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3,
                   verbose=1, mode='min', baseline=None, restore_best_weights=True)


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

for i, (i_trn, i_val) in enumerate(cv.split(pad_x, y), 1):
    print(f'training model for CV #{i}')

    model3.fit(pad_x[i_trn], 
            to_categorical(y[i_trn]),
            validation_data=(pad_x[i_val], to_categorical(y[i_val])),
            epochs=10,
            batch_size=512,
            callbacks=[es, mcp])     # 조기 종료 옵션
                      
    test_y += model3.predict(x_pred) / n_fold    # 나온 예측값들을 교차 검증 횟수로 나눈다

# print(test_y)

topic = []
for i in range(len(test_y)):
    topic.append(np.argmax(test_y[i]))

# print(len(pred)) # 9131
# print('예측값 : ', pred) # ex) 예측값 :  [2 2 2 ... 1 1 6]
submission = pd.read_csv('./data/sample_submission.csv')
submission['topic_idx'] = topic
# print(submission.head())
submission.to_csv('./data/submission16.csv', index=False)
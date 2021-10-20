import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import os
import tqdm

from konlpy.tag import Okt

import sklearn
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import log_loss, accuracy_score,f1_score
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import *
from tensorflow.keras.models import load_model

train=pd.read_csv('./data/train.csv')
test=pd.read_csv('./data/test.csv')
sample_submission=pd.read_csv('./data/sample_submission.csv')

# 데이터 전처리

# #해당 baseline 에서는 과제명 columns만 활용했습니다.
# #다채로운 변수 활용법으로 성능을 높여주세요!
# train=train[['과제명','label']]
# test=test[['과제명']]

# #1. re.sub 한글 및 공백을 제외한 문자 제거
# #2. okt 객체를 활용해 형태소 단위로 나눔
# #3. remove_stopwords로 불용어 제거 
# def preprocessing(text, okt, remove_stopwords=False, stop_words=[]):
#     text=re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ]","", text)
#     word_text=okt.morphs(text, stem=True)
#     if remove_stopwords:
#         word_review=[token for token in word_text if not token in stop_words]
#     return word_review

# stop_words=['은','는','이','가', '하','아','것','들','의','있','되','수','보','주','등','한']
# okt=Okt()
# clean_train_text=[]
# clean_test_text=[]

# #시간이 많이 걸립니다.
# for text in tqdm.tqdm(train['과제명']):
#     try:
#         clean_train_text.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
#     except:
#         clean_train_text.append([])

# for text in tqdm.tqdm(test['과제명']):
#     if type(text) == str:
#         clean_test_text.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
#     else:
#         clean_test_text.append([])

# #텐서플로의 전처리 모듈을 활용해 토크나이징 객체를 만든 후 인덱스 벡터로 전환
# tokenizer=Tokenizer()
# tokenizer.fit_on_texts(clean_train_text)

# train_sequences=tokenizer.texts_to_sequences(clean_train_text)
# test_sequences=tokenizer.texts_to_sequences(clean_test_text)
# word_vocab=tokenizer.word_index

# #패딩 처리
# train_inputs=pad_sequences(train_sequences, maxlen=40, padding='post')
# test_inputs=pad_sequences(test_sequences, maxlen=40, padding='post')

labels=np.array(train['label'])

# #추후 재사용 가능하도록 npy로 전환
# DATA_IN_PATH='./save/'
# TRAIN_INPUT_DATA = 'train_input.npy'
# TEST_INPUT_DATA = 'test_input.npy'

# import os
# if not os.path.exists(DATA_IN_PATH):
#     os.makedirs(DATA_IN_PATH)
    
# np.save(open(DATA_IN_PATH+TRAIN_INPUT_DATA, 'wb'), train_inputs)
# np.save(open(DATA_IN_PATH+TEST_INPUT_DATA, 'wb'), test_inputs)

# data_configs={}
# data_configs['vocab']=word_vocab
# data_configs['vocab_size'] = len(word_vocab)+1
# json.dump(data_configs, open(DATA_IN_PATH+'data_configs.json', 'w'), ensure_ascii=False)

# 모델링

#파라미터 설정
train_inputs = np.load('./save/train_input.npy')
test_inputs = np.load('./save/test_input.npy')

with open('./save/data_configs.json', 'r') as f:

    json_data = json.load(f)

data_configs = json.dumps(json_data) 

vocab_size =data_configs['vocab_size']
embedding_dim = 32
max_length = 40
oov_tok = "<OOV>"

#가벼운 NLP모델 생성
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(46, activation='softmax')
])

# compile model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

num_epochs = 30

# ##########################################################
# import datetime
# date = datetime.datetime.now()
# date_time = date.strftime("%m%d_%H%M")

# filepath = './save/'
# filename = '.{epoch:04d}-{val_loss:.4f}.hdf5'
# modelpath = "".join([filepath, "dacon", date_time, "_", filename])
# #################################################################

# mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True,
#                        filepath= modelpath) 

history = model.fit(train_inputs, labels, 
                    epochs=num_epochs, verbose=2, 
                    validation_split=0.2)

# 예측 후 제출
# model = load_model('./save/dacon0813_1839_.0009-0.4675.hdf5')


#평가지표가 Macro F1이기에 확률값으로 결과를 내면 안됩니다.
pred=model.predict(test_inputs)
pred=tf.argmax(pred, axis=1)

sample_submission['label']=pred

sample_submission.to_csv('./data/lstm_baseline3.csv', index=False)
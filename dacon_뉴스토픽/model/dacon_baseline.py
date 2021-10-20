import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras import Sequential

# 1. 데이터 생성

train      = pd.read_csv("./data/train_data.csv")
test       = pd.read_csv("./data/test_data.csv")
submission = pd.read_csv("./data/sample_submission.csv")
# topic_dict = pd.read_csv("/content/drive/MyDrive/klue/open/topic_dict.csv")

# print(train)

def clean_text(sent):
      sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", sent)
      return sent_clean

train["cleaned_title"] = train["title"].apply(lambda x : clean_text(x))
test["cleaned_title"]  = test["title"].apply(lambda x : clean_text(x))

train_text = train["cleaned_title"].tolist()
test_text = test["cleaned_title"].tolist()

# print(train_text)
train_label = np.asarray(train.topic_idx)

tfidf = TfidfVectorizer(analyzer='word', sublinear_tf=True, ngram_range=(1, 2), max_features=150000, binary=False)

tfidf.fit(train_text)

train_tf_text = tfidf.transform(train_text).astype('float32')
test_tf_text  = tfidf.transform(test_text).astype('float32')

# 2. 모델 생성
def dnn_model():
      model = Sequential()
      model.add(Dense(128, input_dim = 150000, activation = "relu"))
      model.add(Dropout(0.8))
      model.add(Dense(7, activation = "softmax"))
      return model

model = dnn_model()

# 3. 컴파일, 훈련
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.optimizers.Adam(0.001), metrics = ['accuracy'])

history = model.fit(x = train_tf_text[:40000], y = train_label[:40000],
                    validation_data =(train_tf_text[40000:], train_label[40000:]),
                    epochs = 4)

# 4. 평가, 예측
tmp_pred = model.predict(test_tf_text)
print(tmp_pred)
pred = np.argmax(tmp_pred, axis = 1)

submission.topic_idx = pred
print(submission.sample(3))
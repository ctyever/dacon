import pandas as pd
import re
from konlpy.tag import Okt,Mecab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score
from lightgbm import LGBMClassifier
import numpy as np

# 1. 데이터 구성

x = np.load('./save/npy/x_okt_data.npy', allow_pickle=True)
y = np.load('./save/npy/y_okt_data.npy', allow_pickle=True)
x_pred = np.load('./save/npy/x_okt_pred_data.npy', allow_pickle=True)


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

token = Tokenizer(num_words=2000)
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
# from tensorflow.keras.utils import to_categorical

pad_x = pad_sequences(x, padding='pre', maxlen=14)
x_pred = pad_sequences(x_pred, padding='pre', maxlen=14)
# print(pad_x.shape) # (45654, 10)
# # print(pad_x)
# print(x_pred[0])

x_train, x_test, y_train, y_test = train_test_split(pad_x, y, 
         train_size=0.7, shuffle=True, random_state=42)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# print(y_train.shape, y_test.shape) # (36523, 7) (9131, 7)


# 2. 모델구성

lgbm = LGBMClassifier(random_state=42)
lgbm.fit(x_train, y_train)

pred = lgbm.predict(x_test)
accuracy = accuracy_score(y_test,pred)

print('정확도', accuracy)

# 정확도 0.46835073373731473

# tmp_pred = lgbm.predict(x_pred)
# # pred = np.argmax(tmp_pred, axis = 1)
# # print(len(pred)) # 9131
# # print('예측값 : ', pred) # ex) 예측값 :  [2 2 2 ... 1 1 6]
# submission = pd.read_csv('./data/sample_submission.csv')
# submission['topic_idx'] = tmp_pred
# # print(submission.head())
# submission.to_csv('./data/submission18.csv', index=False)


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

4차
sparse_categorical
acc :  0.751724898815155

5차
acc :  0.7631146907806396
모델 변경/ submission5

6차 / 얼리스탑핑
acc :  0.7549008727073669

7차 / mcp
acc :  0.7888511419296265

8차  / 서브미션8 /  / dacon0728_2301_.0002-0.6420.hdf5
acc :  0.7972840070724487

9차
acc :  0.7911510467529297

10차 / dacon0802_2357_.0002-0.6495 / 정답률 제일 높음 / 데이콘5  모델
acc :  0.800700843334198

11차 / dacon0803_2305_.0003-0.6580
acc :  0.8055190443992615
'''

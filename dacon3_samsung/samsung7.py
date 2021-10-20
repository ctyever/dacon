ffpp = "pattern"

import pandas as pd
train = pd.read_csv("./data/train.csv")
dev = pd.read_csv("./data/dev.csv")
test = pd.read_csv("./data/test.csv")

ss = pd.read_csv("./data/sample_submission.csv")

train = pd.concat([train,dev])

train['ST1_GAP(eV)'] = train['S1_energy(eV)'] - train['T1_energy(eV)']

import kora.install.rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model


import numpy as np
import pandas as pd

import math
train_fps = []#train fingerprints
train_y = [] #train y(label)

for index, row in train.iterrows() : 
  try : 
    mol = Chem.MolFromSmiles(row['SMILES'])
    if ffpp == 'maccs' :    
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif ffpp == 'morgan' : 
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4)
    elif ffpp == 'rdkit' : 
        fp = Chem.RDKFingerprint(mol)
    elif ffpp == 'pattern' : 
        fp = Chem.rdmolops.PatternFingerprint(mol)
    elif ffpp == 'layerd' : 
        fp = Chem.rdmolops.LayeredFingerprint(mol)

    train_fps.append(fp)
    train_y.append(row['ST1_GAP(eV)'])
  except : 
    pass

np_train_fps = []
for fp in train_fps:
  arr = np.zeros((0,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  np_train_fps.append(arr)

np_train_fps_array = np.array(np_train_fps)

pd.Series(np_train_fps_array[:,0]).value_counts()

import math
test_fps = []#test fingerprints
test_y = [] #test y(label)

for index, row in test.iterrows() : 
  try : 
    mol = Chem.MolFromSmiles(row['SMILES'])

    if ffpp == 'maccs' :    
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif ffpp == 'morgan' : 
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4)
    elif ffpp == 'rdkit' : 
        fp = Chem.RDKFingerprint(mol)
    elif ffpp == 'pattern' : 
        fp = Chem.rdmolops.PatternFingerprint(mol)
    elif ffpp == 'layerd' : 
        fp = Chem.rdmolops.LayeredFingerprint(mol)

    test_fps.append(fp)
    test_y.append(row['ST1_GAP(eV)'])
  except : 
    pass

np_test_fps = []
for fp in test_fps:
  arr = np.zeros((0,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  np_test_fps.append(arr)

np_test_fps_array = np.array(np_test_fps)

# print(np_test_fps_array.shape)
# print(len(test_y))

pd.Series(np_test_fps_array[:,0]).value_counts()

# import tensorflow as tf
# from tensorflow.keras.layers import Dense
# def create_deep_learning_model():
#     model = Sequential()
#     model.add(Dense(1024, input_dim=2048, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(512, activation='relu'))
#     model.add(Dense(256, activation='relu'))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(16, activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal'))
#     model.compile(loss='mean_absolute_error', optimizer='adam')
#     return model

# X, Y = np_train_fps_array , np.array(train_y)

# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import KFold

# estimators = []
# # estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=create_deep_learning_model, epochs=10)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=5)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# # print("%.2f (%.2f) MAE" % (results.mean(), results.std()))

# model = create_deep_learning_model()

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

# model.fit(X, Y, epochs = 100, callbacks=[mcp], validation_split=0.1)

model = load_model('./save/dacon0824_2151_.0011-0.2199.hdf5')

test_y = model.predict(np_test_fps_array)
ss['ST1_GAP(eV)'] = test_y

ss.to_csv("./data/pattern_mlp9.csv",index=False)

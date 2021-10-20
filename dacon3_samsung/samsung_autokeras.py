import autokeras as ak



#1. 데이터
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

import numpy as np
import pandas as pd

import math
train_fps = [] #train fingerprints
train_y = [] #train y(label)
ffpp = "pattern"

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

X, Y = np_train_fps_array , np.array(train_y)

#2. 모델

model = ak.StructuredDataRegressor(
    overwrite=True,
    max_trials=2
)

# 3. 컴파일, 훈련
model.fit(X, Y, epochs=5)

# 4. 평가 예측
test_y = model.predict(np_test_fps_array)
ss['ST1_GAP(eV)'] = test_y

ss.to_csv("./data/pattern_mlp28.csv",index=False)


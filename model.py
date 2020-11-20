import pandas as pd
import numpy as np
dftrain = pd.read_csv('train.csv')
dftest = pd.read_csv('test.csv')
dftrain = pd.concat([dftrain, dftest], axis=0, sort=True)
dftrain=dftrain.drop(['nums/length username', 'nums/length fullname','external URL' ], axis=1)
X_train = dftrain[pd.notnull(dftrain['fake'])].drop(['fake'], axis=1)
y_train = dftrain[pd.notnull(dftrain['fake'])]['fake']
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=85, batch_size=32, validation_split=0.2)
model.save('insta.h5')
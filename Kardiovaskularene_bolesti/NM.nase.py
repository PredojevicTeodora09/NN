import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.simplefilter('ignore')

data=pd.read_csv('22_cardiovascular_disease_dataset.csv');

print(data.info())
stats=data.describe();
#data.hist(bins=100)

#lose vrednosti zamenimo sa Null

data_ispitivanje=data[['age_years','height','weight']];
#data_ispitivanje.hist(bins=100);

for i in range(70000):
    if data.ap_lo[i]<=0:
        data.ap_lo=data.ap_lo.replace(data.ap_lo[i],np.NaN)
    elif data.ap_lo[i]>=250:
        data.ap_lo=data.ap_lo.replace(data.ap_lo[i],np.NaN)
    elif data.ap_hi[i]>300:
        data.ap_hi=data.ap_hi.replace(data.ap_hi[i],np.NaN)
    elif data.ap_hi[i]<=0:
        data.ap_hi=data.ap_hi.replace(data.ap_hi[i],np.NaN)
    elif data.age_years[i]<30:
        data.age_years=data.age_years.replace(data.age_years[i],np.NaN)
    elif data.height[i]<100:
        data.height=data.height.replace(data.height[i],np.NaN)
    elif data.height[i]>205:
        data.height=data.height.replace(data.height[i],np.NaN)
    elif data.weight[i]<35:
        data.weight=data.weight.replace(data.weight[i],np.NaN)
        
        
#print('\n')        
#print('Statistika podataka nakon sredjivanja :')
#print('\n')
#data.hist(bins=100)
 
#izbacivanje svih NaN
data.dropna(axis=0,inplace=True)
#print(data.info())  
#%%

# NM
X=data.iloc[:, :-1]; 
D=data.iloc[:, -1]; 

Xmean = np.mean(X, axis=0) # srednja vrednost po kolonama; bez axis sr vrednost cele matrice
Xmax = np.max(X, axis=0)
Xstd = np.std(X, axis = 0)
X = (X - Xmean)/Xmax
from sklearn.model_selection import train_test_split
#delimo skup na obucavajuci i testirajuci
Xtrening,Xtest,Dtrening,Dtest=train_test_split(X,D,test_size=0.6, shuffle=True)
#import tensorflow.compat.v2 as tf
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2

#%% JEDAN SKRIVENI SLOJ SA 20 NEURONA
model=Sequential()
#dodajemo jedan skriveni sloj koji ima 20 neurona
#za skriveni sloj se najcesce koristi relu aktivaciona fja (0 ako je x<0, x inace)
model.add(Dense(20,activation='relu'))
#izlazni sloj
#najcesce sigmoid aktivaciona fja
model.add(Dense(1,activation='sigmoid'))


model.build((None,X.shape[1]))
print(model.summary())

#optimizator predstavlja metodu trazenja minimuma krit fje
#cross entropy krit fja
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history=model.fit(Xtrening,Dtrening,epochs=30,batch_size=32,validation_data=(Xtest,Dtest))


plt.figure()
plt.plot(history.history['loss'], label='Training data')
plt.plot(history.history['val_loss'], label='Validation data')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')

Ypom=model.predict(Xtest)
Ypred=1*(Ypom>0.5)
score,acc=model.evaluate(Xtest,Dtest)
print('Test accuracy:', acc)
from sklearn import metrics
prec=metrics.precision_score(Dtest,Ypred)
print('Test precision:', prec)
rec=metrics.recall_score(Dtest,Ypred)
print('Test recall:', rec)
#ACCURACY JE TACNOST, RECALL JE OSETLJIVOST(TP/P), PRECISION JE PPV(TP/Ppred)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Dtest,Ypred)
plt.figure()
sns.heatmap(cm,annot=True,fmt='g',cbar=False)


#%% JEDAN SKRIVENI SLOJ SA 5 NEURONA
model=Sequential()
model.add(Dense(5,activation='relu'))

model.add(Dense(1,activation='sigmoid'))


model.build((None,X.shape[1]))
print(model.summary())


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history=model.fit(Xtrening,Dtrening,epochs=30,batch_size=32,validation_data=(Xtest,Dtest))


plt.figure()
plt.plot(history.history['loss'], label='Training data')
plt.plot(history.history['val_loss'], label='Validation data')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')

Ypom=model.predict(Xtest)
Ypred=1*(Ypom>0.5)
score,acc=model.evaluate(Xtest,Dtest)
print('Test accuracy:', acc)
from sklearn import metrics
prec=metrics.precision_score(Dtest,Ypred)
print('Test precision:', prec)
rec=metrics.recall_score(Dtest,Ypred)
print('Test recall:', rec)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Dtest,Ypred)
plt.figure()
sns.heatmap(cm,annot=True,fmt='g',cbar=False)


#%% JEDAN SKRIVENI SLOJ SA 50 NEURONA
model=Sequential()

model.add(Dense(50,activation='relu'))

model.add(Dense(1,activation='sigmoid'))


model.build((None,X.shape[1]))
print(model.summary())


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


history=model.fit(Xtrening,Dtrening,epochs=50,batch_size=32,validation_data=(Xtest,Dtest))

plt.figure()
plt.plot(history.history['loss'], label='Training data')
plt.plot(history.history['val_loss'], label='Validation data')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')

Ypom=model.predict(Xtest)
Ypred=1*(Ypom>0.5)
score,acc=model.evaluate(Xtest,Dtest)
print('Test accuracy:', acc)
from sklearn import metrics
prec=metrics.precision_score(Dtest,Ypred)
print('Test precision:', prec)
rec=metrics.recall_score(Dtest,Ypred)
print('Test recall:', rec)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Dtest,Ypred)
plt.figure()
sns.heatmap(cm,annot=True,fmt='g',cbar=False)


#%% DVA SKRIVENA SLOJA 3 I 2 NEURONA
model=Sequential()

model.add(Dense(3,activation='relu',input_dim=12))
#model.add(Dropout(0.5))
model.add(Dense(2,activation='relu'))


model.add(Dense(1,activation='sigmoid'))

model.build((None,X.shape[1]))
print(model.summary())

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history=model.fit(Xtrening,Dtrening,batch_size=32,epochs=30,validation_data=(Xtest,Dtest))

plt.figure()
plt.plot(history.history['loss'], label='Training data')
plt.plot(history.history['val_loss'], label='Validation data')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')

Ypom=model.predict(Xtest)
Ypred=1*(Ypom>0.5)
score,acc=model.evaluate(Xtest,Dtest)
print('Test accuracy:', acc)
from sklearn import metrics
prec=metrics.precision_score(Dtest,Ypred)
print('Test precision:', prec)
rec=metrics.recall_score(Dtest,Ypred)
print('Test recall:', rec)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Dtest,Ypred)
plt.figure()
sns.heatmap(cm,annot=True,fmt='g',cbar=False)

#%% DVA SKRIVENA SLOJA 25 I 15 NEURONA
model=Sequential()

model.add(Dense(25,activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(15,activation='relu'))


model.add(Dense(1,activation='sigmoid'))

model.build((None,X.shape[1]))
print(model.summary())

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history=model.fit(Xtrening,Dtrening,batch_size=32,epochs=50,validation_data=(Xtest,Dtest))

plt.figure()
plt.plot(history.history['loss'], label='Training data')
plt.plot(history.history['val_loss'], label='Validation data')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')

Ypom=model.predict(Xtest)
Ypred=1*(Ypom>0.5)
score,acc=model.evaluate(Xtest,Dtest)
print('Test accuracy:', acc)
from sklearn import metrics
prec=metrics.precision_score(Dtest,Ypred)
print('Test precision:', prec)
rec=metrics.recall_score(Dtest,Ypred)
print('Test recall:', rec)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Dtest,Ypred)
plt.figure()
sns.heatmap(cm,annot=True,fmt='g',cbar=False)

#%% VISE SLOJEVA
model=Sequential()

model.add(Dense(100,activation='relu',input_dim=12))
#model.add(Dropout(0.5))
model.add(Dense(40,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.build((None,X.shape[1]))
print(model.summary())

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


history=model.fit(Xtrening,Dtrening,batch_size=32,epochs=30,validation_data=(Xtest,Dtest))

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

Ypom=model.predict(Xtest)
Ypred=1*(Ypom>0.5)
score,acc=model.evaluate(Xtest,Dtest)
print('Test accuracy:', acc)
from sklearn import metrics
prec=metrics.precision_score(Dtest,Ypred)
print('Test precision:', prec)
rec=metrics.recall_score(Dtest,Ypred)
print('Test recall:', rec)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Dtest,Ypred)
plt.figure()
sns.heatmap(cm,annot=True,fmt='g',cbar=False)

#%% ZASTITA OD PREOBUCAVANJA - RANO ZAUSTAVLJANJE
model=Sequential()

model.add(Dense(20,activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(15,activation='relu'))


model.add(Dense(1,activation='sigmoid'))

model.build((None,X.shape[1]))
print(model.summary())

from keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss',patience=7)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


history=model.fit(Xtrening,Dtrening,batch_size=32,epochs=50,validation_data=(Xtest,Dtest),callbacks=[es])


plt.figure()
plt.plot(history.history['loss'], label='Training data')
plt.plot(history.history['val_loss'], label='Validation data')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')

Ypom=model.predict(Xtest)
Ypred=1*(Ypom>0.5)
score,acc=model.evaluate(Xtest,Dtest)
print('Test accuracy:', acc)
from sklearn import metrics
prec=metrics.precision_score(Dtest,Ypred)
print('Test precision:', prec)
rec=metrics.recall_score(Dtest,Ypred)
print('Test recall:', rec)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Dtest,Ypred)
plt.figure()
sns.heatmap(cm,annot=True,fmt='g',cbar=False)

#%% REGULACIJA


model = Sequential()
model.add(Dense(200, input_dim=np.shape(Xtrening)[1], 
                activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(100, activation='relu')) 
model.add(Dense(50, activation='relu'))

model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
history = model.fit(Xtrening, Dtrening, 
                    epochs=50,
                    batch_size=32, 
                    validation_data=(Xtest, Dtest), 
                    verbose=0,
                   )

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

YpredTrening = model.predict(Xtrening, verbose=0)
YpredTrening = np.argmax(YpredTrening, axis=1)

YpredTest = model.predict(Xtest, verbose=0)
YpredTest = np.argmax(YpredTest, axis=1)

from sklearn.metrics import accuracy_score
Atrening = accuracy_score(Dtrening, YpredTrening)
print('Tačnost na trening skupu iznosi: ' + str(Atrening*100) + '%.')

Atest = accuracy_score(Dtest, YpredTest)
print('Tačnost na test skupu iznosi: ' + str(Atest*100) + '%.')

prec=metrics.precision_score(Dtest,Ypred)
print('Test precision:', prec)
rec=metrics.recall_score(Dtest,Ypred)
print('Test recall:', rec)

Ntest = 100
x = np.linspace(-4, 7, Ntest)
y = np.linspace(-4, 7, Ntest)
Xgrid, Ygrid = np.meshgrid(x, y)
Xgrid = Xgrid.reshape((1, Ntest**2))
Ygrid = Ygrid.reshape((1, Ntest**2))

grid = np.append(Xgrid, Ygrid, axis=0).T
Ypred =  np.argmax(model.predict(grid, verbose=0), axis=1)
        
K1pred = grid[Ypred==0, :]
K2pred = grid[Ypred==1, :]  

#%%
plt.figure()
plt.plot(K1pred[:, 0], K1pred[:, 1], 'r.', alpha=0.1)
plt.plot(K2pred[:, 0], K2pred[:, 1], 'b.', alpha=0.1)

plt.plot(K1[:, 0], K1[:, 1], 'r.')
plt.plot(K2[:, 0], K2[:, 1], 'b.')

plt.show()
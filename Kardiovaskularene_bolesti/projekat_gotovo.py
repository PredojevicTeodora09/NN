import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter('ignore')

data=pd.read_csv('22_cardiovascular_disease_dataset.csv');

#print(data.info())
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
        
#izbacivanje svih NaN
data.dropna(axis=0,inplace=True) 


#%%


corr=data.corr(method='spearman');
plt.figure()
sns.heatmap(corr,annot=True)

#Korelaciona analiza:
def izracunajR(corr):
    k=corr.shape[0]-1; 
    rzi=np.mean(corr.iloc[-1,:-1]); 
    rii=np.mean(np.mean(corr.iloc[:-1,:-1]))-1/k; 
    r=k*rzi/np.sqrt(k+k*(k-1)*rii);
    return r

R=izracunajR(corr);
print('r bez izbacivanja jedne kolone:')
print(R)
i=1;
r=np.zeros((12,2));
for ob in range(data.shape[1]-1):
    data_1=data.drop(data.columns[ob], axis=1); # izbacujemo jednu kolonu
    corr=data_1.corr(method='spearman');
    print('izabcivanje', i, '. kolone:')
    print(izracunajR(corr))
    r[i-1,0]=izracunajR(corr);
    if r[i-1,0]<R:
        r[i-1,1]=-1;
    elif r[i-1,0]>R:
        r[i-1,1]=1;
    print('--------------')
    i=i+1;

col_names=data.columns.values;

#IG:
print(' ')
print('IG')
klasa=data.cardio;

def izracunajInfoD(kol):
    un=np.unique(kol);
    infoD=0;
    for u in un:
        pi=np.sum(kol==u)/len(kol);
        infoD+= -pi*np.log2(pi);
    return infoD

infoD=izracunajInfoD(klasa);

N=5; 
IG=np.zeros((data.shape[1]-1,2)); 
for ob in range(data.shape[1]-1):
    kol = data.iloc[:,ob]; 
    korak=(np.max(kol)-np.min(kol))/N;
    nova_kol=np.floor(kol/korak)*korak;
    
    
    un=np.unique(nova_kol);
    infoDA=0;
    for u in un:
        pom=klasa[nova_kol==u];
        infoDi=izracunajInfoD(pom);
        
        Di=np.sum(nova_kol==u);
        D=len(nova_kol);
        
        infoDA+=Di*infoDi/D;
        
    
    IG[ob,0]=infoD-infoDA;
    IG[ob,1]=ob+1;
    data.iloc[:,ob]=nova_kol;

IG=sorted(IG,key=lambda x: (x[0]))
IG=np.flipud(IG);
print(IG)

#%%
#LDA na 2D

X=data.iloc[:, :-1]; 
D=data.iloc[:, -1]; 

Xmean=np.mean(X, axis=0); 
Xmax=np.max(X,axis=0);
Xstd=np.std(X,axis=0);
Xmin=np.min(X,axis=0);
#Xnorm=(X-Xmean)/Xmax;
Xnorm=(X-Xmean)/Xstd;
#Xnorm=(X-Xmin)/(Xmax-Xmin);

X1=Xnorm.loc[D==0,:];
M1=X1.mean().values.reshape(X1.shape[1],1);
S1=X1.cov();
p1=X1.shape[0]/X.shape[0]


X2=Xnorm.loc[D==1,:];
M2=X2.mean().values.reshape(X2.shape[1],1);
S2=X2.cov();
p2=X2.shape[0]/X.shape[0]



M0=p1*M1+p2*M2;
Sw=p1*S1+p2*S2;
Sb=p1*(M1-M0)@(M1-M0).T+p2*(M2-M0)@(M2-M0).T;

Sm=Sb+Sw;

T=np.linalg.inv(Sw)@Sb; 


eigval, eigvec=np.linalg.eig(T);

sops_vr=np.zeros((12,2));
for i in range(12):
    sops_vr[i,0]=eigval[i];
    sops_vr[i,1]=i+1;
    
sops_vr=sorted(sops_vr,key=lambda x: (x[0]))
sops_vr=np.flipud(sops_vr);

# sortiranje sopstvenih vrednosti eigval
ind=np.argsort(eigval)[::-1]; 
eigval=eigval[ind];
eigvec=eigvec[:,ind]; 

m=2;
A=eigvec[:,:m]; 
Y=A.T @ Xnorm.T; 

data_lda=pd.concat([Y.T, D], axis=1) ;
data_lda.columns=['LDA1','LDA2','Class'];

import seaborn as sns
sns.scatterplot(data=data_lda, x='LDA1',y='LDA2',hue='Class');

#%%
# LDA 2D bez age_years
data1=data.drop(columns=['age_years']);
X=data1.iloc[:, :-1]; 
D=data1.iloc[:, -1]; 

Xmean=np.mean(X, axis=0); 
Xmax=np.max(X,axis=0);
Xstd=np.std(X,axis=0);
Xmin=np.min(X,axis=0);
#Xnorm=(X-Xmean)/Xmax;
Xnorm=(X-Xmean)/Xstd;
#Xnorm=(X-Xmin)/(Xmax-Xmin);

X1=Xnorm.loc[D==0,:];
M1=X1.mean().values.reshape(X1.shape[1],1);
S1=X1.cov();
p1=X1.shape[0]/X.shape[0]


X2=Xnorm.loc[D==1,:];
M2=X2.mean().values.reshape(X2.shape[1],1);
S2=X2.cov();
p2=X2.shape[0]/X.shape[0]

M0=p1*M1+p2*M2;
Sw=p1*S1+p2*S2;
Sb=p1*(M1-M0)@(M1-M0).T+p2*(M2-M0)@(M2-M0).T;

Sm=Sb+Sw;

T=np.linalg.inv(Sw)@Sb; 


eigval, eigvec=np.linalg.eig(T);

sops_vr=np.zeros((11,2));
for i in range(11):
    sops_vr[i,0]=eigval[i];
    sops_vr[i,1]=i+1;
    
sops_vr=sorted(sops_vr,key=lambda x: (x[0]))
sops_vr=np.flipud(sops_vr);

# sortiranje sopstvenih vrednosti eigval
ind=np.argsort(eigval)[::-1]; 
eigval=eigval[ind];
eigvec=eigvec[:,ind]; 

m=2;
A=eigvec[:,:m]; 
Y=A.T @ Xnorm.T; 

data_lda=pd.concat([Y.T, D], axis=1) ;
data_lda.columns=['LDA1','LDA2','Class'];

import seaborn as sns
sns.scatterplot(data=data_lda, x='LDA1',y='LDA2',hue='Class');


#%%
#LDA na 3D

X=data.iloc[:, :-1]; 
D=data.iloc[:, -1]; 

Xmean=np.mean(X, axis=0);  
Xmax=np.std(X,axis=0);
Xnorm=(X-Xmean)/Xmax;


X1=Xnorm.loc[D==0,:];
M1=X1.mean().values.reshape(X1.shape[1],1);
S1=X1.cov();
p1=X1.shape[0]/X.shape[0]


X2=Xnorm.loc[D==1,:];
M2=X2.mean().values.reshape(X2.shape[1],1);
S2=X2.cov();
p2=X2.shape[0]/X.shape[0]

M0=p1*M1+p2*M2;
Sw=p1*S1+p2*S2;
Sb=p1*(M1-M0)@(M1-M0).T+p2*(M2-M0)@(M2-M0).T;

Sm=Sb+Sw;

T=np.linalg.inv(Sw)@Sb; 

eigval, eigvec=np.linalg.eig(T);

sops_vr=np.zeros((12,2));
for i in range(12):
    sops_vr[i,0]=eigval[i];
    sops_vr[i,1]=i+1;
    
sops_vr=sorted(sops_vr,key=lambda x: (x[0]))
sops_vr=np.flipud(sops_vr);

# sortiranje sopstvenih vrednosti eigval
ind=np.argsort(eigval)[::-1]; 
eigval=eigval[ind];
eigvec=eigvec[:,ind]; 

m=3;
A=eigvec[:,:m]; 
Y=A.T @ Xnorm.T; 

data_lda=pd.concat([Y.T, D], axis=1) ;
data_lda.columns=['LDA1','LDA2','LDA3','Class'];

import seaborn as sns

ax = plt.axes(projection='3d')
xdata=data_lda.LDA1;
ydata=data_lda.LDA2;
zdata=data_lda.LDA3;
h=data_lda.Class;

ax.scatter3D(xdata, ydata, zdata, c=h)

#%%
# LDA 3D bez age_years:
data1=data.drop(columns=['age_years']);
X=data1.iloc[:, :-1]; 
D=data1.iloc[:, -1]; 

Xmean=np.mean(X, axis=0);  
Xmax=np.std(X,axis=0);
Xnorm=(X-Xmean)/Xmax;


X1=Xnorm.loc[D==0,:];
M1=X1.mean().values.reshape(X1.shape[1],1);
S1=X1.cov();
p1=X1.shape[0]/X.shape[0]


X2=Xnorm.loc[D==1,:];
M2=X2.mean().values.reshape(X2.shape[1],1);
S2=X2.cov();
p2=X2.shape[0]/X.shape[0]

M0=p1*M1+p2*M2;
Sw=p1*S1+p2*S2;
Sb=p1*(M1-M0)@(M1-M0).T+p2*(M2-M0)@(M2-M0).T;

Sm=Sb+Sw;

T=np.linalg.inv(Sw)@Sb; 


eigval, eigvec=np.linalg.eig(T);

sops_vr=np.zeros((12,2));
for i in range(11):
    sops_vr[i,0]=eigval[i];
    sops_vr[i,1]=i+1;
    
sops_vr=sorted(sops_vr,key=lambda x: (x[0]))
sops_vr=np.flipud(sops_vr);

# sortiranje sopstvenih vrednosti eigval
ind=np.argsort(eigval)[::-1]; 
eigval=eigval[ind];
eigvec=eigvec[:,ind]; 

m=3;
A=eigvec[:,:m]; 
Y=A.T @ Xnorm.T; 

data_lda=pd.concat([Y.T, D], axis=1) ;
data_lda.columns=['LDA1','LDA2','LDA3','Class'];

import seaborn as sns

ax = plt.axes(projection='3d')
xdata=data_lda.LDA1;
ydata=data_lda.LDA2;
zdata=data_lda.LDA3;
h=data_lda.Class;

ax.scatter3D(xdata, ydata, zdata, c=h)

#%%
# PCA na 2D
X=data.iloc[:, :-1]; 
D=data.iloc[:, -1];


Xmean=np.mean(X, axis=0);
Xmax=np.max(X,axis=0); 
Xstd=np.std(X,axis=0);
Xmin=np.min(X,axis=0);
#Xnorm=(X-Xmean)/Xmax;
Xnorm=(X-Xmean)/Xstd;
#Xnorm=(X-Xmin)/(Xmax-Xmin);


Sx=np.cov(Xnorm.T); 
eigval, eigvec=np.linalg.eig(Sx);

sopstv_vr=eigval;

ind=np.argsort(eigval)[::-1]; 
                              
eigval=eigval[ind];
eigvec=eigvec[:,ind]; 

m=2;
A=eigvec[:,:m]; 
Y=A.T @ Xnorm.T; 

data_pca=pd.concat([Y.T, D], axis=1) ;
data_pca.columns=['PCA1','PCA2','Class'];

import seaborn as sns
sns.scatterplot(data=data_pca, x='PCA1',y='PCA2',hue='Class');

#%%
# PCA 2D bez age_years
data1=data.drop(columns=['age_years']);
X=data1.iloc[:, :-1]; 
D=data1.iloc[:, -1];


Xmean=np.mean(X, axis=0);
Xmax=np.max(X,axis=0); 
Xstd=np.std(X,axis=0);
Xmin=np.min(X,axis=0);
#Xnorm=(X-Xmean)/Xmax;
Xnorm=(X-Xmean)/Xstd;
#Xnorm=(X-Xmin)/(Xmax-Xmin);


Sx=np.cov(Xnorm.T); 
eigval, eigvec=np.linalg.eig(Sx);

sopstv_vr=eigval;

ind=np.argsort(eigval)[::-1]; 
                              
eigval=eigval[ind];
eigvec=eigvec[:,ind]; 

m=2;
A=eigvec[:,:m]; 
Y=A.T @ Xnorm.T; 

data_pca=pd.concat([Y.T, D], axis=1) ;
data_pca.columns=['PCA1','PCA2','Class'];

import seaborn as sns
sns.scatterplot(data=data_pca, x='PCA1',y='PCA2',hue='Class');


#%%
# LDA na 1D


X=data.iloc[:, :-1]; 
D=data.iloc[:, -1]; 

Xmean=np.mean(X, axis=0); 
Xmax=np.max(X,axis=0);
Xstd=np.std(X,axis=0);
Xmin=np.min(X,axis=0);
#Xnorm=(X-Xmean)/Xmax;
Xnorm=(X-Xmean)/Xstd;
#Xnorm=(X-Xmin)/(Xmax-Xmin);

X1=Xnorm.loc[D==0,:];
M1=X1.mean().values.reshape(X1.shape[1],1);
S1=X1.cov();
p1=X1.shape[0]/X.shape[0]


X2=Xnorm.loc[D==1,:];
M2=X2.mean().values.reshape(X2.shape[1],1);
S2=X2.cov();
p2=X2.shape[0]/X.shape[0]

M0=p1*M1+p2*M2;
Sw=p1*S1+p2*S2;
Sb=p1*(M1-M0)@(M1-M0).T+p2*(M2-M0)@(M2-M0).T;

Sm=Sb+Sw;

T=np.linalg.inv(Sw)@Sb; 


eigval, eigvec=np.linalg.eig(T);

sops_vr=np.zeros((12,2));
for i in range(12):
    sops_vr[i,0]=eigval[i];
    sops_vr[i,1]=i+1;
    
sops_vr=sorted(sops_vr,key=lambda x: (x[0]))
sops_vr=np.flipud(sops_vr);

# sortiranje sopstvenih vrednosti eigval
ind=np.argsort(eigval)[::-1]; 
eigval=eigval[ind];
eigvec=eigvec[:,ind]; 

m=1;
A=eigvec[:,:m]; 
Y=A.T @ Xnorm.T; 

data_lda=pd.concat([Y.T, D], axis=1) ;
data_lda.columns=['LDA1','Class'];

import seaborn as sns

xdata=data_lda.LDA1;
ydata=np.zeros((xdata.shape[0],));
h=data_lda.Class;

sns.scatterplot(xdata, ydata, hue=h)

#%% parametarska klas - lin klas na bazi zeljenog izlaza

X=data.iloc[:, :-1]; 
D=data.iloc[:, -1]; 

Xmean=np.mean(X, axis=0); 
Xmax=np.max(X,axis=0);
Xstd=np.std(X,axis=0);
Xmin=np.min(X,axis=0);
#Xnorm=(X-Xmean)/Xmax;
Xnorm=(X-Xmean)/Xstd;
#Xnorm=(X-Xmin)/(Xmax-Xmin);

X1=Xnorm.loc[D==0,:];
M1=X1.mean().values.reshape(X1.shape[1],1);
S1=X1.cov();
p1=X1.shape[0]/X.shape[0]


X2=Xnorm.loc[D==1,:];
M2=X2.mean().values.reshape(X2.shape[1],1);
S2=X2.cov();
p2=X2.shape[0]/X.shape[0]

M0=p1*M1+p2*M2;
Sw=p1*S1+p2*S2;
Sb=p1*(M1-M0)@(M1-M0).T+p2*(M2-M0)@(M2-M0).T;

Sm=Sb+Sw;

T=np.linalg.inv(Sw)@Sb; 


eigval, eigvec=np.linalg.eig(T);

sops_vr=np.zeros((12,2));
for i in range(12):
    sops_vr[i,0]=eigval[i];
    sops_vr[i,1]=i+1;
    
sops_vr=sorted(sops_vr,key=lambda x: (x[0]))
sops_vr=np.flipud(sops_vr);

# sortiranje sopstvenih vrednosti eigval
ind=np.argsort(eigval)[::-1]; 
eigval=eigval[ind];
eigvec=eigvec[:,ind]; 

m=1;
A=eigvec[:,:m]; 
Y=A.T @ Xnorm.T; 

data_lda_fcn=pd.concat([Y.T, D], axis=1) ;
data_lda_fcn.columns=['LDA1','Class'];


f = data_lda_fcn.iloc[:,0] 
c = data_lda_fcn.iloc[:,1] 

from sklearn.model_selection import train_test_split
Ftrening,Ftest,Ctrening,Ctest=train_test_split(f,c,train_size=0.6,random_state=11,stratify=c)

K0 = Ftrening[Ctrening == 0].to_numpy()
K1 = Ftrening[Ctrening == 1].to_numpy()

N0 = np.shape(K0)[0]
N1 = np.shape(K1)[0]
N = N0+N1

K0 = np.zeros((1,N0)) + K0
K1 = np.zeros((1,N1)) + K1

Z0=np.append(-K0,-np.ones((1,N0)),axis=0)
Z1=np.append(K1,np.ones((1,N1)),axis=0)

U=np.append(Z0,Z1,axis=1)
#G=np.ones((N,1))
G=np.append(0.5*np.ones((N0,1)),np.ones((N1,1)),axis=0)
W=np.linalg.inv(U@U.T)@U@G 

V=W[0]
V0=W[1]

y=np.array([-0.04, 0.04])
x=(y-V0)/V

h = V*Ftest+V0
Ctest_got = (h>=0) * 1

from sklearn.metrics import accuracy_score, confusion_matrix
acc=accuracy_score(Ctest,Ctest_got)
print('Test accuracy:', 100*acc)
from sklearn import metrics
prec=metrics.precision_score(Ctest,Ctest_got)
print('Test precision:', 100*prec)
rec=metrics.recall_score(Ctest,Ctest_got)
print('Test recall:', 100*rec)

conf_mat=confusion_matrix(Ctest,Ctest_got)
print(conf_mat)
import seaborn as sns
plt.figure()
sns.heatmap(conf_mat,annot=True,fmt='g')

df = pd.DataFrame({'Ftest': Ftest, 'Column2': Ctest})
plt.figure()
#sns.scatterplot(data=data_lda_fcn, x = 'LDA1', y = 0, hue = 'Class')

#plt.plot(Ftest[Ctest==0], np.zeros((np.sum(Ctest==0),1)),'.')
#plt.plot(Ftest[Ctest==1], np.zeros((np.sum(Ctest==1),1)),'.',alpha=0.05)

sns.scatterplot(data=df, x = 'Ftest', y = 0, hue = 'Column2')
plt.legend()

plt.plot(x,y,'g')



#%%
# testiranje hipoteza ali pre toga izbacimo obelezjana osnovu ig
new_data=data.drop(columns=['alco','gender','smoke','height']);
X=new_data.iloc[:,:-1].values
Y=new_data.iloc[:,-1].values #da bi dobila niz
X0 = X[Y==0,:]
X1 = X[Y==1,:]
N0 = X0.shape[0]
N1 = X1.shape[0]

f = new_data.iloc[:,:-1] 
c = new_data.iloc[:,-1] 

from sklearn.model_selection import train_test_split
Xtren,Xtest,Ytren,Ytest=train_test_split(X,Y,train_size=0.6,random_state=11,stratify=c)

X0tren=Xtren[Ytren==0,:]
N0tren=X0tren.shape[0]
X1tren=Xtren[Ytren==1,:]
N1tren=X1tren.shape[0]

X0test=Xtest[Ytest==0,:]
X1test=Xtest[Ytest==1,:]

M0=np.mean(X0tren,axis=0)
S0=np.cov(X0tren.T) 

M1=np.mean(X1tren,axis=0)
S1=np.cov(X1tren.T)

def fgv(x,m,s):
    det=np.linalg.det(s)
    inv=np.linalg.inv(s)

    fgv_const=1/np.sqrt(2*np.pi*det)
    fgv_rest=np.exp(-0.5*(x-m).T@inv@(x-m))

    return fgv_const*fgv_rest

p0=N0tren/(N0tren+N1tren)
p1=N1tren/(N0tren+N1tren)
T=np.log(p0/p1)

decision0=np.zeros((N0-N0tren,1))
for i in range(N0-N0tren):
    x0=X0test[i,:]

    f0=fgv(x0,M0,S0)
    f1=fgv(x0,M1,S1)

    h0=np.log(f1)-np.log(f0)

    if h0<T:
        decision0[i]=0
    else:
        decision0[i]=1

decision1 = np.zeros((N1 - N1tren, 1))
for i in range(N1 - N1tren):
    x1 = X1test[i, :]

    f0 = fgv(x1, M0, S0)
    f1 = fgv(x1, M1, S1)

    h1 = np.log(f1) - np.log(f0)

    if h1 < T:
        decision1[i] = 0
    else:
        decision1[i] = 1


decision=np.append(decision0,decision1,axis=0)
true_val=np.append(np.zeros((N0-N0tren)), np.ones((N1-N1tren)),axis=0)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
conf_mat=confusion_matrix(true_val,decision)


acc=accuracy_score(true_val,decision)
print('Test accuracy:', 100*acc)
from sklearn import metrics
prec=metrics.precision_score(true_val,decision)
print('Test precision:', 100*prec)
rec=metrics.recall_score(true_val,decision)
print('Test recall:', 100*rec)

plt.figure()
sns.heatmap(conf_mat,annot=True,fmt='g')

#%% 
# NEPARAMETARSKA KLASIFIKACIJA
# stablo

outcome = data.cardio
features = data.drop(columns = ['cardio'])

X=features.values
Y=outcome.values

m=np.mean(X,axis=0)
s=np.std(X,axis=0)

Xnorm=(X-m)/s

from sklearn.model_selection import train_test_split
Xtren,Xtest,Ytren,Ytest=train_test_split(Xnorm,Y,train_size=0.6,random_state=11,stratify=Y)

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score

criterion=['gini','entropy']
max_depth=[i for i in range(2,15)]
params={'criterion':criterion,'max_depth': max_depth}

from sklearn.model_selection import GridSearchCV
models=GridSearchCV(DecisionTreeClassifier(), param_grid=params, cv=5)
#cv razdvaja na particije, za testiranje i obucavanje
models.fit(Xtren,Ytren)

max_depth_entropy = models.cv_results_['param_max_depth'][::2]
accuracies_entropy = models.cv_results_['mean_test_score'][::2]

max_depth_gini = models.cv_results_['param_max_depth'][1::2]
accuracies_gini = models.cv_results_['mean_test_score'][1::2]

max_depth_entropy, accuracies_entropy = zip(*sorted(zip(max_depth_entropy, accuracies_entropy)))
max_depth_gini, accuracies_gini = zip(*sorted(zip(max_depth_gini, accuracies_gini)))


plt.plot(max_depth_entropy, accuracies_entropy, 'o-', label='Entropy')
plt.plot(max_depth_gini, accuracies_gini, 'o-', label='Gini')
plt.title('Cross-Validation Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()

#print(models.best_estimator_.get_params())

best_cr=models.best_estimator_.get_params()['criterion']
best_md=models.best_estimator_.get_params()['max_depth']
model=DecisionTreeClassifier(criterion=best_cr, max_depth=best_md, class_weight='balanced')
#model=DecisionTreeClassifier(criterion=best_cr, max_depth=best_md)
model.fit(Xtren,Ytren)
pred=model.predict(Xtest)

acc=accuracy_score(Ytest,pred)
print('Test accuracy:', 100*acc)
from sklearn import metrics
prec=metrics.precision_score(Ytest,pred)
print('Test precision:', 100*prec)
rec=metrics.recall_score(Ytest,pred)
print('Test recall:', 100*rec)

cm=confusion_matrix(Ytest,pred)
plt.figure()
sns.heatmap(cm,annot=True,fmt='g')

plt.figure()
nazivi_kolona=data.columns
plot_tree(model,feature_names=nazivi_kolona, class_names=['0','1'],fontsize=7,\
          impurity=False, filled=True, rounded=True, precision=0, node_ids=False)








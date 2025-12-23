# Random Forest

import numpy as np, pandas as pd, sklearn.ensemble as ske
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

df=pd.read_excel("features.xlsx")   
df = df.dropna()
Y = df['ideal'] 
del df['ideal'] 
del df['Muestra']        

lista = ['Mean', 'Std', 'Skew', 'Curtosis', 'Sum','% I4', '% mean-std', 'Mean/mode', 'Mean/std',] 

df = df[lista] 
Names = df.columns
scaler = StandardScaler()  
scaled_data = scaler.fit_transform(df) 

X = df

iterations = 500 
ImpVec = np.zeros((iterations,len(X.columns))) 
for i in range(iterations): 
    reg = ske.RandomForestRegressor() 
    reg.fit(X,Y)    
    ImpVec[i,:] = reg.feature_importances_  

Desv = np.std(ImpVec,0)*100    
Importance = np.mean(ImpVec,0)*100 
                            
feature_names = np.array(reg.feature_names_in_) 
Result = np.array([feature_names, Importance, Desv]).T 

feat_ind = np.argsort(Importance)
feat_imp = Importance[feat_ind]


fig = plt.figure(figsize = (20,10)) 
ax  = plt.subplot()


plt.barh((np.arange(len(feat_imp))),(feat_imp), linewidth=2,height=1.2,
         edgecolor='k',color='b')

title = """Feature Selection"""
ax.set_yticklabels(feature_names)
ax.set_yticks(np.arange(len(feat_imp))+.5)
plt.suptitle(title,fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 13)
plt.ylim(0,len(feat_imp))
plt.xlabel('Weight (%)',fontsize = 20)
plt.show()

print(feature_names)
print(Importance)

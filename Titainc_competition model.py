# %%
import pandas as pd
import matplotlib as plt
import numpy as np


# %%
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,HistGradientBoostingClassifier,ExtraTreesClassifier,AdaBoostClassifier,BaggingClassifier,StackingClassifier,IsolationForest,VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# %%
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv('test.csv')

data = df_train

# %%
data.isna().sum().sort_values()

# %%
def title(word):
    parts = word.split(",")
    if len(parts )> 1 and "Mr" in parts[1]:
        return 0
    else:
        return 1
    
data['title']=data["Name"].apply(title)
    

# %%
data = data.drop(["PassengerId",'Name','Cabin'],axis=1)
data.shape

# %%
data.isna().sum().sort_values()

# %%
mean_fare = np.mean(data['Fare'])
data['Fare']=data['Fare'].fillna(mean_fare)
data['Age']=data['Age'].fillna(np.mean(data['Age']))
data =data.dropna(subset=['Embarked'])

# %%
data
def age_adult_child(a):
    if a <= 15:
        return 1
    else:
        return 0
data['Age']=data['Age'].apply(age_adult_child)

# %%

data['Age']=data['Age'].astype('int')

# %%
data = data.drop(['Ticket'],axis=1)
def assigner(a):
    if a == True:
        return 1
    else:
        return 0
data = pd.get_dummies(data,columns=['Sex','Embarked'])
data



# %%
for i in ['Sex_male','Sex_female','Embarked_C','Embarked_Q','Embarked_S']:
    data[i]=data[i].apply(assigner)
data

# %%
#changing class weights
data['Pclass'] = data['Pclass'].map({1:3, 2:2, 3: 1})
data

# %%
X_train = data.drop(['Survived'],axis=1).values
Y_train = data['Survived'].values


# %%
model = {'LogisticRegression':0,'RandomForestClassifier':0,'Xgboost':0}
logreg = LogisticRegression(solver='liblinear')
Rclf = RandomForestClassifier()
Xgboost = GradientBoostingClassifier(subsample= 0.8, n_estimators= 100, min_samples_split= 10, min_samples_leaf= 1,max_features= None, max_depth= 5, learning_rate= 0.01)
his = HistGradientBoostingClassifier()
ex = ExtraTreesClassifier()
ada =AdaBoostClassifier()
Bas = BaggingClassifier()
#Vot = VotingClassifier()
models =[logreg,Rclf,Xgboost,his,ex,ada,Bas]
scores =list()
for i in models:
    Kf = KFold(n_splits=5,shuffle=True,random_state=42)
    cross = cross_val_score(i,X_train,Y_train,cv=Kf)
    scores.append(np.mean(cross))
print(scores)
    


# %%
data = df_test
data['title']=data["Name"].apply(title)
data = data.drop(["PassengerId",'Name','Cabin'],axis=1)
mean_fare = np.mean(data['Fare'])
data['Fare']=data['Fare'].fillna(mean_fare)
data['Age']=data['Age'].fillna(np.mean(data['Age']))
data =data.dropna(subset=['Embarked'])
data['Age']=data['Age'].apply(age_adult_child)

# %%
data.isna().sum().sort_values()

# %%
data = pd.get_dummies(data,columns=['Sex','Embarked'])
for i in ['Sex_male','Sex_female','Embarked_C','Embarked_Q','Embarked_S']:
    data[i]=data[i].apply(assigner)
data['Pclass'] = data['Pclass'].map({1:3, 2:2, 3: 1})
data

# %%
#grid search
data = data.drop(['Ticket'],axis=1)
data

# %%
from sklearn.model_selection import RandomizedSearchCV



# Define the parameter grid
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', None]
}

# Initialize GridSearchCV
grid_search = RandomizedSearchCV(estimator=Xgboost, param_distributions=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

# Fit the model
grid_search.fit(X_train, Y_train)

# Best parameters
best_params = grid_search.best_params_

# Best estimator
best_model = grid_search.best_estimator_

print("Best Parameters:", best_params)




# %%
Xgboost.fit(X_train,Y_train)
Xgboost.score(X_train,Y_train)
x = Xgboost.predict(data)

# %%
submission=pd.DataFrame({"PassengerId":df_test['PassengerId'],"Survived": x})

# %%
submission.head()



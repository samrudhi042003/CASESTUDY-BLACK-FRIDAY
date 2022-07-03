import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
svc = SVR()

data = pd.read_csv('train.csv')

data['Age']=(data['Age'].str.strip('+'))
data['Stay_In_Current_City_Years']=(data['Stay_In_Current_City_Years'].str.strip('+').astype('float'))

data['Product_Category_2'].fillna(data['Product_Category_2'].mean(),inplace=True)
data['Product_Category_3'].fillna(data['Product_Category_3'].mean(),inplace=True)

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Age'] = le.fit_transform(data['Age'])
data['City_Category'] = le.fit_transform(data['City_Category'])

x = data.drop(['Purchase','Product_ID','User_ID'], axis=1)
x = x.astype('int')
y = data['Purchase']


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3 )

svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)


print(mean_squared_error(y_test,y_pred))

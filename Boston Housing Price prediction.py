import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,accuracy_score,mean_squared_error,r2_score

#Load the Dataset
df=pd.read_csv("HousingData.csv")

df=df.fillna(0)
print(df.head(5))

#Visualizing the correct data 
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
plt.show()

sns.scatterplot(x="RM",y="MEDV",data=df)
plt.show()
sns.scatterplot(x="LSTAT",y="MEDV",data=df)
plt.show()


X=df[["RM","LSTAT","PTRATIO"]]
y=df["MEDV"]

#Train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)

#Feature Scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


#Train the regression model
model=LinearRegression()
model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)

print("Mean Absolute Error:",mean_absolute_error(y_test,y_pred))
print("MSE",mean_squared_error(y_test,y_pred))
print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))
print("R2",r2_score(y_test,y_pred))

#Plot the predicted and actual
plt.scatter(y_test,y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()
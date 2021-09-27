import pandas as pd 
from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt






Data = pd.read_csv("nyc_cyclist_counts.csv") 

X = Data[Data.columns[[1,2,3]]]
y = Data[Data.columns[4]]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


reg = TweedieRegressor(power=1, alpha=0.5, link='log')
reg.fit(X_train,y_train)
print(reg.coef_)
print(reg.intercept_)

#%% Eval

y_test_pred = reg.predict(X_test)
avg_error = np.sqrt(mean_squared_error(y_test, y_test_pred))/len(y_test)
plt.scatter(y_test_pred, y_test, color = 'r', label = 'Class II'  )
plt.xlabel("Real Data")
plt.ylabel("Prediction")
print("Average Error: ", avg_error)
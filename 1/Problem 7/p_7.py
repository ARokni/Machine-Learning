import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
#np.random.seed(42)
pi = np.pi

model_size = 300   
test_size = 10000
mu = 0; std = 1
X = 2*(1 * np.random.rand(model_size, 1) - 0.5)
#X = np.random.rand(model_size,1)* 2 *pi - 0.5 * pi
y = np.cos(2*np.pi*X) + np.random.normal(mu, std, (model_size,1))
X_test = np.sort(2 * np.random.rand(test_size, 1) - 1)
y_test = np.cos(2*np.pi*X_test) + np.random.normal(mu, std, (test_size,1))


plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-2, 2, -5, 5])
plt.grid()
plt.show()
#%%

poly_degree = [1,4,8,12,15,20]
val_errors = []
for deg in poly_degree:
    poly_features = PolynomialFeatures(degree=deg, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    print(lin_reg.intercept_, lin_reg.coef_)
    
    X_new=np.linspace(-1, 1, model_size).reshape(model_size, 1)
   # X_new = np.random.rand(model_size,1)* 2 *pi - 0.5 * pi
    X_new_poly = poly_features.transform(X_new)
    y_new = lin_reg.predict(X_new_poly)
    plt.plot(X, y, "b.")
    plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.title(label = 'Ploynomial Degree{}'.format(deg),  loc='center')
    plt.legend(loc="upper left", fontsize=14)
    plt.axis([-2, 2, -5, 5])
    plt.show()
    #%% MSE
    X_new_poly = poly_features.transform(X_test)
    y_new = lin_reg.predict(X_new_poly)
    val_errors.append(mean_squared_error(y_new, y_test))
#%%
  

plt.plot(poly_degree,(val_errors), "r-+", linewidth=2, label="Val")
plt.legend(loc="upper right", fontsize=14) 
plt.xlabel("Degree of Model", fontsize=14) 
plt.ylabel("MSE", fontsize=14)              


































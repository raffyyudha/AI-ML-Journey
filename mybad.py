print("lvyoda")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X = np.array([2, 3 , 12, 5, 1, 5, 6, 1, 6, 9]).reshape(-1,1)
y = np.array([23, 21, 80, 5, 3, 2, 4, 2, 52 , 6])

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

X_predict = np.linspace(0, 11, 100).reshape(-1,1)
X_predict_poly = poly.transform(X_predict)
y_predict = model.predict(X_predict_poly)

plt.scatter(X, y , color='red' , label ='Real Data')
plt.plot(X_predict, y_predict, color='blue' , label='Polynomial Regression(deg=2)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()
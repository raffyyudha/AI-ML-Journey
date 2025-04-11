import  numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.array([[150], [153], [160], [170], [176]])
y = np.array([0, 0 , 0, 0, 1])

model = LogisticRegression()
model.fit(X,y)

height = np.linspace(140, 180, 300).reshape(-1, 1)
probability = model.predict_proba(height)[:,1]

plt.plot(height, probability, color='blue' , label = 'Handsdome Rate')
plt.scatter(X, y, color = 'red' , label = 'Data Asli')
plt.xlabel("Height")
plt.ylabel("Handsome Rate")
plt.title("Indonesian Handsome Rate")
plt.grid(True)
plt.legend()
plt.show()

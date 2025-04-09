import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

data = {
    "Year": [2005, 2010, 2015, 2020, 2025],
    "Indomie_Price": [1.0, 1.5, 2.0, 2.5, 3.0]
}

df = pd.DataFrame(data)
X = df[["Year"]]
y = df["Indomie_Price"]

model = LinearRegression()
model.fit(X, y)

predict_year = np.array([[2030]])
prediksi = model.predict(predict_year)
print(f"Indomie Price in 2030: {prediksi[0]:.2f} IDR")

plt.scatter(X, y, label="Real Data")
plt.plot(X, model.predict(X), color='red', label="Linear Regression")
plt.scatter(2030, prediksi, color='green', label="Prediction 2030", zorder=5)
plt.xlabel("Year")
plt.ylabel("Indomie Price (IDR)")
plt.title("The Price Increase of Indomie")
plt.legend()
plt.grid(True)
plt.show()

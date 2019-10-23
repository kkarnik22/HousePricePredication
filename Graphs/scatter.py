import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Graphs/train.csv', delimiter=',', nrows=100)

y = df['GarageYrBlt'];
x = df['GarageArea']

plt.figure(figsize=(8, 5))
plt.style.use('ggplot')
plt.xlabel('GarageArea', fontsize=8)
plt.ylabel('GarageYrBlt', fontsize=8)
plt.title('Scatter Plot : GarageArea,GarageYearBuilt')
plt.scatter(x,y)
plt.show()

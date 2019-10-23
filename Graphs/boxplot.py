import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

train = pd.read_csv('Graphs/train.csv')
plt.figure(figsize=(11, 6))
plt.style.use('ggplot')
plt.xlabel('SalePrice', fontsize=9)
plt.ylabel('Probability', fontsize=9)

sns.boxplot(x=train.OverallQual, y=train.SalePrice).set_title('Box Plot')
plt.show()

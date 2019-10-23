import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

plt.style.use('ggplot')

df = pd.read_csv('Graphs/train.csv', delimiter=',', nrows=100)

sns.distplot(df['SalePrice'],hist=True, kde=False, 
             color = 'blue',
             hist_kws={'edgecolor':'black'})


plt.ylabel('Frequency')


plt.xlabel('SalePrice', fontsize=8)
plt.ylabel('Frequency', fontsize=8)
plt.title('Histogram')
plt.show()

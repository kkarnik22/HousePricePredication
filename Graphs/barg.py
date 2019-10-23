import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('Graphs/train.csv', delimiter=',', nrows=100)
#missing value counts in each of these columns
Isnull = df.isnull().sum()/len(df)*100
Isnull = Isnull[Isnull>0]
Isnull.sort_values(inplace=True, ascending=False)
#print(Isnull)
#Convert into dataframe
Isnull = Isnull.to_frame()
Isnull.columns = ['count']
Isnull.index.names = ['Name']
Isnull['Name'] = Isnull.index
#plot Missing values
plt.figure(figsize=(10, 8))
#print(Isnull)
sns.set(style='whitegrid')
sns.barplot(x='Name', y='count', data=Isnull).set_title('LIST OF NULL VALUES')
plt.xticks(rotation = 22, fontsize=8)


plt.show()

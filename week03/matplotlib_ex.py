import pandas as pd

titanic = pd.read_csv('3.1.1.titanic.csv')

print(titanic.head())

print(titanic.info())

pclass_survived_mean = titanic.groupby('Pclass')['Survived'].mean().reset_index()
pclass_survived_mean

import matplotlib.pyplot as plt

plt.plot(pclass_survived_mean['Pclass'], pclass_survived_mean['Survived'],
         marker='o', linestyle='-', color='violet')
plt.title('Survival Rate Variation Across Passenger Classes')
plt.xlabel('Pclass')
plt.ylabel('Survived Rate')
plt.xticks([1,2,3])
plt.grid(True)
plt.savefig('Fugure01.png')
plt.close()
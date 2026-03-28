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

survived_counts = titanic[titanic['Survived'] == 1]['Embarked'].value_counts()
print(survived_counts)

plt.bar(survived_counts.index, survived_counts,
        color = ['mediumorchid', 'darkviolet', 'indigo'])
plt.title('Survived Counts by Emarked Port on Titanic')
plt.xlabel('Embarked Port')
plt.ylabel('Count')
plt.xticks(survived_counts.index, ['Southampton', 'Cherbourg', 'Queenstown'])
plt.legend(['Survived'], loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, value in enumerate(survived_counts):
  plt.text(i, value + 1, str(value), ha='center', va='bottom')

plt.savefig('Figure02.png')
plt.close()

survived_counts = titanic[titanic['Survived'] == 1]['Sex'].value_counts()
print(survived_counts)

bars = plt.barh(survived_counts.index, survived_counts, color=['darkturquoise', 'salmon'])
plt.title('Survived Counts by Gender on Titanic')
plt.xlabel('Count')
plt.ylabel('Gender')
plt.legend(bars, ['Survived - Female', 'Survived - Male'], loc='upper right')

plt.axvline(x=survived_counts['male'], color='gray',linestyle='--', linewidth=1)

for i, value in enumerate(survived_counts):
  plt.text(value + 1, i, str(value), ha='left', va='center')

plt.savefig('Figure03.png')
plt.close()

print(titanic.info(), '\n')

titanic = titanic.dropna(subset=['Age', 'Fare', 'Survived'])
print(titanic.info())

plt.figure(figsize=(12, 8))
scatter = plt.scatter(x='Age', y='Fare', data=titanic, c=titanic['Survived'],cmap='Set2', alpha=0.7)

plt.title('Age and Fare Relationship with Survival on the Titanic')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(handles=scatter.legend_elements()[0], title='Servived',
           labels=['Not Survived', 'Survived'], loc='upper right')
plt.savefig('Figure04.png')
plt.close()

survived_counts = titanic['Survived'].value_counts()
print(survived_counts)

plt.figure(figsize=(8 ,8))
plt.pie(survived_counts, labels=['Not Survived', 'Survived'], colors = ['orange', 'gold'],
        autopct='%0.1f%%', startangle=90, shadow=True, explode=(0, 0.1))

plt.title('Survival Distribution on the Titanic')
plt.savefig('Figure05.png')
plt.close()

print(titanic.info(), '\n')

titanic = titanic.dropna(subset=['Age'])
print(titanic.info())

plt.figure(figsize=(10, 6))
plt.hist(titanic['Age'], bins=20, color='seagreen', edgecolor='black')

plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Ages on the Titanic')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('Figure06.png')
plt.close()

titanic = titanic.dropna(subset={'Age', 'Fare'})

correlation_matrix = titanic.drop('PassengerId', axis=1).corr(numeric_only=True)
print(correlation_matrix)

plt.matshow(correlation_matrix, cmap='PuRd_r')
plt.colorbar()

plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)

plt.title('Correlation Heatmap of Titanic')
plt.savefig('Figure07.png')
plt.close()

titanic = titanic.dropna(subset=['Age', 'Fare'])

age_groups = pd.cut(titanic['Age'], bins=range(0, 81, 5))

survived_counts = titanic.groupby([age_groups, 'Survived'], observed=False).size().unstack().file.na(0)
print(survived_counts)

plt.figure(figsize=(10, 6))

plt.fill_between(survived_counts.index.astype(str), survived_counts[1],
                 color='purple', alpha=0.9, label='Survived')

plt.fill_between(survived_counts.index.astype(str), survived_counts[0],
                 color='hotpink', alpha=0.6, label='Not Survived')

plt.title('Survival by Age Group on Titanic')
plt.xlabel('Age')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.savefig('Figure08.png')
plt.close()




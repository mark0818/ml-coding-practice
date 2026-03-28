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

survived_counts = titanic.groupby([age_groups, 'Survived'], observed=False).size().unstack().fillna(0)
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

titanic = titanic.dropna(subset=['Age'])
print(titanic.info())

plt.boxplot([titanic[titanic['Pclass'] == 1]['Age'],
             titanic[titanic['Pclass'] == 2]['Age'],
             titanic[titanic['Pclass'] == 3]['Age']],
            labels=['1st Class', '2nd Class', '3rd Class'])

plt.title('Box Plot for Age by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Age')
plt.savefig('Figure09.png')
plt.close()

titanic['Age'] = titanic['Age'].fillna(titanic['Age'].mean())
print(titanic.info())

plt.figure(figsize=(10, 6))

violin_plot = plt.violinplot([titanic[titanic['Pclass'] == 1]['Age'],
                              titanic[titanic['Pclass'] == 2]['Age'],
                              titanic[titanic['Pclass'] == 3]['Age']],
                             showmeans=False, showmedians=True)

plt.title('Violin Plot of Age by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Age')

plt.xticks([1, 2, 3], ['1st Class', '2nd Class', '3rd Class']) 

plt.legend(violin_plot['bodies'], ['1st Class', '2nd Class', '3rd Class'],
           title='Pclass', loc="upper right")
plt.savefig('Figure10.png')
plt.close()

fare_means = titanic.groupby('Parch')['Fare'].mean()
print(fare_means, '\n')

fare_std = titanic.groupby('Parch')['Fare'].std()
print(fare_std)

plt.figure(figsize=(10,6))

plt.errorbar(fare_means.index, fare_means, yerr=fare_std, fmt='o',
             capsize=5, capthick=1, label='Fare')

plt.title('Error Bar Plot of Fare by Parch')
plt.xlabel('Parch')
plt.ylabel('Fare')
plt.xticks(fare_means.index)
plt.legend()
plt.savefig('Figure11.png')
plt.close()

plt.subplot(2, 2, 1)
plt.plot([1, 2, 3])

plt.subplot(2, 2, 2)
plt.plot([4, 5, 6])

plt.subplot(2, 2, 3)
plt.plot([7, 8, 9])

plt.subplot(2, 2, 4)
plt.plot([10, 11, 12])
plt.savefig('Figure12.png')
plt.close

titanic = pd.read_csv('3.1.1.titanic.csv')
parch_counts = titanic.groupby('Parch')['Survived'].value_counts().unstack().fillna(0)
print(parch_counts)

x= parch_counts.index.astype(str)
y1 = parch_counts[0].values
y2 = parch_counts[1].values

plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)
plt.plot(x, y1, '-o', color='indigo', markersize=7, linewidth=3, alpha=0.7,
label='Not Survived')
plt.xlabel('Parch')
plt.ylabel('Not Survived Count', color='indigo')
plt.tick_params(axis='y', labelcolor='indigo')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)
plt.bar(x, y2, color='deeppink', alpha=0.7, width=0.5, label='Survived')
plt.xlabel('Parch')
plt.ylabel('Survived Count', color='deeppink')
plt.tick_params(axis='y', labelcolor='deeppink')
plt.legend(loc='upper right')

plt.suptitle('Survival Analysis by Number of Parents/Children (Parch) on the Titanic')
plt.tight_layout()
plt.savefig('Figure13.png')
plt.close()

fig, axes = plt.subplots(2, 2)

axes[0, 0].plot([1, 2, 3])
axes[0, 1].plot([4, 5, 6])
axes[1, 0].plot([7, 8, 9])
axes[1, 1].plot([10, 11, 12])
plt.savefig('Figure14.png')
plt.close()

parch_counts = titanic.groupby('Parch')['Survived'].value_counts().unstack().fillna(0)
print(parch_counts)

x = parch_counts.index.astype(str) 
y1 = parch_counts[0].values
y2 = parch_counts[1].values

fig, axes = plt.subplots(2, 1, figsize=(10,10))

axes[0].plot(x, y1, '-o', color='indigo', markersize=7, linewidth=3, alpha=0.7, label='Not Survived')
axes[0].set_xlabel('Parch')
axes[0].set_ylabel('Not Survived Count', color='indigo')
axes[0].tick_params(axis='y', labelcolor='indigo')
axes[0].legend(loc='upper right')

axes[1].bar(x, y2, color='deeppink', alpha=0.7, width=0.5, label='Survived')
axes[1].set_xlabel('Parch')
axes[1].set_ylabel('Survived Count', color='deeppink')
axes[1].tick_params(axis='y', labelcolor='deeppink')
axes[1].legend(loc='upper right')

fig.suptitle('Survival Analysis by Number of Parents/Children (Parch) on the Titanic')
fig.tight_layout()
plt.savefig('Figure15.png')
plt.close()

parch_counts = titanic.groupby('Parch')['Survived'].value_counts().unstack().fillna(0)
print(parch_counts)

x = parch_counts.index.astype(str) 
y1 = parch_counts[0].values
y2 = parch_counts[1].values

fig, ax1 = plt.subplots()

ax1.plot(x, y1, '-s', color='indigo', markersize=7, linewidth=5, alpha=0.7, label='Not Survived')
ax1.set_xlabel('Parch')
ax1.set_ylabel('Not Survived Count', color='indigo')
ax1.tick_params(axis='y', labelcolor='indigo')
ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))

ax2 = ax1.twinx()

ax2.bar(x, y2, color='deeppink', alpha=0.7, width=0.7, label='Survived')
ax2.set_ylabel('Survived Count', color='deeppink')
ax2.tick_params(axis='y', labelcolor='deeppink')
ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.9))

plt.suptitle('Survived Analysis by Number of Parents/Children (Parch) on the Titanic')
plt.tight_layout()
plt.savefig('Figure16.png')
plt.close()

plt.hist(titanic['Age'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Ages on the Titanic')
plt.grid(axis='y', linstyle='--', alpha=0.7)

plt.savefig('Figure17.png')
plt.close()
               
           

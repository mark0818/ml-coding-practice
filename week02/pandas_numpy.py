import pandas as pd

file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/sample.csv'
sample = pd.read_csv(file_url)

print(sample.head())
print(sample.tail())

sample.info()
sample.describe()

sample_dic = {'name': ['john', 'Ann', 'Kevin'], 'age': [23,22,21]}
a = pd.DataFrame(sample_dic)

a.info()

pd.DataFrame([[1,2,],[3,4],[5,6], [7,8]])
pd.DataFrame([[1,2],[3,4],[5,6], [7,8]], columns = ['var_1', 'var_2'], index=['a', 'b', 'c', 'd'])

import pandas as pd
file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/sample_df.csv'

sample_df = pd.read_csv(file_url, index_col=0)
print(sample_df.head())

print(sample_df['var_5'])

print(sample_df[['var_1', 'var_4']])

print(sample_df.loc['a'])
print(sample_df.loc[['a','c','e']])
print(sample_df.loc['a':'c'])

print(sample_df.iloc[[0,1,2]])
print(sample_df.iloc[0:2])
print(sample_df.iloc[0:3])
print(sample_df.iloc[0:3, 2:4])

print(sample_df.drop(['var_1', 'var_3'], axis=1))
print(sample_df.drop(['var_1', 'var_2'], axis=1))
print(sample_df.drop(['a','b','c'], axis=0))

netflix= pd.read_csv('2.1.1.netflix.csv')
print(netflix.head())

print(netflix['release_year'])
print(netflix['release_year'] > 2015)

more2015 = netflix[netflix['release_year'] > 2015]
print(more2015.head(10))

print(~(netflix['release_year'] > 2015))
less2015 = netflix[~(netflix['release_year'] > 2015)]
print(less2015.head())

print((netflix['release_year'] > 2015) & (netflix['type'] == 'TV Show'))
more2015_tv = netflix[(netflix['release_year'] > 2015) & (netflix['type'] == 'TV Show')]
print(more2015_tv.head())

data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Hannah'],
    'comment_length': [150, 200, 50, 300, 120, 180, 75, 160],
    'likes': [25, 30, 10, 45, 20, 35, 5, 28],
    'is_spam': [False, False, True, False, False, True, False, False],
    'has_image': [True, False, True, True, False, False, True, True]
}
df = pd.DataFrame(data)
print(df.head())

condition = (
    (df['comment_length'] >= 100) &
    (df['likes'] >= 20) &
    (~df['is_spam']) &
    (df['has_image'])
)

winner_df = df[condition]
print(winner_df)
print(sample_df.reset_index())
print(sample_df.reset_index(drop=True))
print(sample_df.set_index('var_1'))

print(sample_df.describe())
print(sample_df.std())
print(sample_df.agg(['count', 'mean', 'std', 'min', 'max']))

file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/iris.csv'
iris = pd.read_csv(file_url)

print(iris.head())

print(iris.groupby('class').std())

print(iris.drop('class', axis=1).agg(['sum','mean','std']))

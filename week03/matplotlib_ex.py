import pandas as pd
import os

titanic = pd.read_csv('3.1.1.titanic.csv')
# Get the directory where the script is located
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, '3.1.1.titanic.csv')

titanic = pd.read_csv(file_path)
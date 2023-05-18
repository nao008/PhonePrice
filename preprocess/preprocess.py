import pandas as pd
from sklearn.model_selection import train_test_split
import os

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
class DataCleansing():
     def __init__(self,original_df=pd.read_csv(f"{current_dir}/data/train.csv",index_col=0)):
          self.original_x = original_df.drop(['price_range'],axis=1)
          self.original_y = original_df["price_range"]
          
          
     def train_split(self):
          X_train, X_test, y_train, y_test = train_test_split(self.original_x, self.original_y, 
                                                                   test_size=0.3, 
                                                                   random_state=42
                                                                   )
          return X_train, X_test, y_train, y_test
          
import pandas as pd
from sklearn.model_selection import train_test_split
import os

directory = os.getcwd()
# directory = os.path.dirname(directory)

class DataCleansing():
     def __init__(self,original_df=pd.read_csv(f"{directory}/data/official_data/train.csv",index_col=0)):#trainデータを取得
          self.original_x = original_df.drop(['price_range'],axis=1)
          self.original_y = original_df["price_range"]
          
     def rt_train_data(self):
          x = self.original_x
          y = self.original_y
          return x, y
     
     def train_split(self):
          #学習データと検証用データに分割
          X_train, X_test, y_train, y_test = train_test_split(self.original_x, self.original_y,
                                                                   test_size=0.3,
                                                                   random_state=42
                                                                   )
          return X_train, X_test, y_train, y_test
          
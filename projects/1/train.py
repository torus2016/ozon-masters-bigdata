#!/opt/conda/envs/dsenv/bin/python

import os, sys
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump

#
# Import model definition
#
from model import model
#, fields_t


#
# Logging initialization
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#
# Read script arguments
#
try:
  proj_id = sys.argv[1] 
  train_path = sys.argv[2]
except:
  logging.critical("Need to pass both project_id and train dataset path")
  sys.exit(1)


logging.info(f"TRAIN_ID {proj_id}")
logging.info(f"TRAIN_PATH {train_path}")


#
# Read dataset
#

file_fields = ["id","label"] + ["if"+str(i) for i in range(1,14)] + ["cf"+str(i) for i in range(1,27)] + ["day_number"]


read_table_opts = dict(sep="\t", file_fields, index_col=False)
df = pd.read_table(train_path, **read_table_opts)

df2 = df.iloc[:,1]
df.drop('label',axis = 1, inplace = True)
df.drop('cf4',axis = 1, inplace = True)
df.drop('cf6', axis = 1, inplace = True)
df.drop('cf7',axis = 1, inplace = True)
df.drop('cf8',axis = 1, inplace = True)
df.drop('cf9',axis = 1, inplace = True)
df.drop('cf14', axis = 1, inplace = True)
df.drop('cf15', axis = 1, inplace = True)
df.drop('cf16', axis = 1, inplace = True)
df.drop('cf17', axis = 1, inplace = True)
df.drop('cf18', axis = 1, inplace = True)
df.drop('cf19', axis = 1, inplace = True)
df.drop('cf25',axis = 1, inplace = True)
df.drop('cf26', axis = 1, inplace = True)
df.drop('day_number', axis = 1, inplace = True)

#split train/test
X_train, X_test, y_train, y_test = train_test_split(
    df, df2, test_size=0.33, random_state=42
)

#print(df.head(10));
#print(df.iloc[:,13:].head(10));
#print(df.iloc[:,21:].head(10));
#print(df2.head(10));
#
# Train the model
#
model.fit(X_train, y_train)

model_score = model.score(X_test, y_test)

logging.info(f"model score: {model_score:.3f}")

# save the model
dump(model, "{}.joblib".format(proj_id))

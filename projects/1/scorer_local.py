#!/opt/conda/envs/dsenv/bin/python

#
# This is a MAE scorer
#

import sys
import os
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#
# Read true values
#
try:
    true_path, pred_path = sys.argv[1:]
except:
    logging.critical("Parameters: true_path (local) and pred_path (url or local)")
    sys.exit(1)

logging.info(f"TRUE PATH {true_path}")
logging.info(f"PRED PATH {pred_path}")


#open true path
#df_true = pd.read_csv(true_path, header=None, index_col=0, names=["id", "true"])
df_true = pd.read_csv(true_path, sep="\t", header=None, index_col=0, names=["id", "true"])

#open pred_path
#df_pred = pd.read_csv(pred_path, header=None, index_col=0, names=["id", "pred"])
df_pred = pd.read_csv(pred_path, sep="\t", header=None, index_col=0, names=["id", "pred"])

len_true = len(df_true)
len_pred = len(df_pred)

logging.info(f"TRUE RECORDS {len_true}")
logging.info(f"PRED RECORDS {len_pred}")

assert len_true == len_pred, f"Number of records differ in true and predicted sets"


#print(df_true.head(10))
#print (df_pred.head(10))

#print ('np.any(np.isnan(df_true))')
#print(np.any(np.isnan(df_true)))
#print ('np.any(np.isnan(df_pred))')
#print(np.any(np.isnan(df_pred)))
#print(np.isnan(df_true).head(10))

#print('len(df_pred)')
#print(len(df_pred))
#df_pred = df_pred.dropna(how='any',axis=0)
#print('len(df_pred)')
#print(len(df_pred))

df = df_true.join(df_pred)
len_df = len(df)
assert len_true == len_df, f"Combined true and pred has different number of records: {len_df}"

#df = df.dropna(how='any',axis=0)
score = log_loss(df['true'], df['pred'])

print(score)

sys.exit(0)
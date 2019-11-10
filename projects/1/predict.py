#!/opt/conda/envs/dsenv/bin/python

import sys, os
import logging
from joblib import load
import pandas as pd

sys.path.append('.')
#from model import fields

#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#load the model
model = load("1.joblib")

#read and infere
file_fields = ["id"] + ["if"+str(i) for i in range(1,14)] + ["cf"+str(i) for i in range(1,27)] + ["day_number"]
read_opts=dict(
        sep='\t', names=file_fields, index_col=False, header=None,
        iterator=True, chunksize=10000
)

for df in pd.read_csv(sys.stdin, **read_opts):
    pred = model.predict(df)
    out = zip(df.id, pred)
    print("\n".join(["{0}\t{1}".format(*i) for i in out]))
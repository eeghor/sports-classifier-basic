"""
--------------------------------------------------------------------
This script loads a trained sport classifier and does classification 
--------------------------------------------------------------------
version: 23FEB2017 
author: igor k.
--------------------------------------------------------------------
note:
	* the trained model MUST be sitting in the MODEL subfolder
input:
	* a CSV TAB-SEPARATED file where each line is of the form
		[pk_event_dim] [some text]
	  should be a commend line argument;
	  the top line is assumed to be a HEADER (because it is by default 
	  when one exports SQL tables to CSV)
output:
	* the same as input but with the predicted sport type attached:
		[pk_event_dim], [some text], [precdicted sport type]
"""

import sys, os
import pandas as pd
import pickle
import csv

csv_in = sys.argv[1]

df_in = pd.read_csv(csv_in, sep="\t", index_col=0)

# check if a trained model is available in the MODEL directory
if not os.path.isdir("model"):
	print("where is your MODEL directory?")
	sys.exit()
elif not os.path.isfile("model/trained_classifier.pkl"):
	print("your MODEL FILE is missing...")
	sys.exit()

# read the trained model
model = pickle.load(open("model/trained_classifier.pkl","rb"))

y_pred = model.predict(df_in.iloc[:,0])

# save this to a file
pd.concat([df_in, pd.Series(y_pred, name="sport_p", index=df_in.index)], axis=1).to_csv("data_tmp/" + csv_in.split(".")[0] + "_CLASSIFIED.csv",sep="\t")
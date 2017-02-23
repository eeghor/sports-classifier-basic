"""
--------------------------------------------------------------------
Train a sport classifier that decides if a text is about 
	a particular sport or something else
--------------------------------------------------------------------
version: 23FEB2017 
author: igor k.
--------------------------------------------------------------------
note:
	* this is a minimalistic (feature-wise) classifier that still 
		performs well
	* more advanced versions that will follow are expected to show
		only marginally improved performance
	* relies on a few "advanced" scikit-learn (http://scikit-learn.org/)
		features such as pipelines and feature unions
"""

import pandas as pd
import time
from datetime import datetime as dt
import sys, os, shutil
import pickle

####### h o u s e k e e p i n g

def setup_dirs(dnames, todo="check"):

	"""
	if any directory from the directory list exists, delete it and create anew;
	alternatively, check if a directory exists and if it doesn't, create it
	"""

	for dname in dnames:

		if os.path.isdir(dname):  # if dir exists
			if todo == "reset":  #  we need to delete it and create anew 
				shutil.rmtree(dname)
				os.mkdir(dname)		
			elif todo == "check":  # only had to check if exists, do nothing
				pass
		else:  # if directory doesn't exist
			if todo == "check":
				sys.exit(">> error >> directory {} is MISSING!".format(dname))
			elif todo == "reset":
				os.mkdir(dname)

####### c u s t o m  t r a n s f o r m e r s

from sklearn.base import BaseEstimator, TransformerMixin

class WordEndingIsS_ft(BaseEstimator, TransformerMixin):

	"""
	feature: how many words in a STRING end with "s"
	"""

	def __init__(self):
		pass  # do nothing

	def count_s_endings(self, st):

		c = 0  # counter

		for word in st.lower().split():
			if word.strip() and word.strip()[-1] == "s":
				c += 1
		return c

	def transform(self, df, y=None):

		out = df.apply(self.count_s_endings).values
		out = out.reshape(out.shape[0],1)

		return  out 

	def fit(self, df, y=None):
		return self

####### c l a s s i f i e r

class SportsClassifier(object):

	def __init__(self):
		
		self.DATA_DIR = "data/"
		self.MODEL_DIR = "model/"
		self.TEMP_DATA_DIR = "data_tmp/"	

		setup_dirs([self.DATA_DIR])		
		setup_dirs([self.TEMP_DATA_DIR, self.MODEL_DIR], "reset")

		self.data_file = self.DATA_DIR + "training_data.csv.gz"
		self.pred_file = self.TEMP_DATA_DIR + "predictions.csv"  # save predictions on testing set here
		self.model_file = self.MODEL_DIR + "trained_classifier.pkl"

		self.data = pd.read_csv(self.data_file, index_col=0, compression="gzip")

		# note that in self.data the column pk_event_dim is index, so event_text is  column 0 and sport is 1

		self.sports = list(self.data.iloc[:,1].unique())

		print("""
			|
			| SPORTS CLASSIFIER
			| 
			\n::: {} :::\n""".format(" ".join([w for w in self.sports if w != "nonsport"])))
		
	def train(self, ptest=0.3):
		"""
		split data into the training and teating sets; 
		proportion ptest left for testing;
		recall that the data has columns
		pk_event_dim | event_text | sport
		and we use pk_event_dim as index

		"""
		from sklearn.model_selection import train_test_split

		self.train_nofeatures, self.test_nofeatures, self.y_train, self.y_test = \
				train_test_split(self.data.iloc[:,0], self.data.iloc[:,1], test_size=ptest, 
					stratify = self.data.iloc[:,1], random_state=113)
		
		NR_train = self.train_nofeatures.shape[0]
		NR_test  = self.test_nofeatures.shape[0]
		NR = self.data.shape[0]

		print("training set: {} rows ({:.1f}%)".format(NR_train, 100*NR_train/NR))
		print("testing set: {} rows ({:.1f}%)".format(NR_test, 100*NR_test/NR))

		from sklearn.feature_extraction.text import TfidfVectorizer
		from sklearn.ensemble import RandomForestClassifier
		from sklearn.pipeline import Pipeline, FeatureUnion
		from sklearn.metrics import classification_report
		from sklearn.metrics import accuracy_score

		"""
		set up a pipeline:
			- no need to generate any features to feed into the pipeline, only provide the training set as is 
			 	along with the target column
			- works with pandas columns, no conversion to numpy arrays needee
			- same for testing: need the testing set as is and the target column
		"""
		print("setting up pipeline...", end="")

		ppl = Pipeline([("ext_features", 
						FeatureUnion([("ngrams", TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', binary=True,
										strip_accents="ascii", stop_words="english", min_df=1)),
										("s_endings", WordEndingIsS_ft())])), 
					("classifier", RandomForestClassifier())])

		ppl.set_params(classifier__n_estimators=200)
		print("ok")

		print("training the model...", end="")
		t0 = time.time()
		model = ppl.fit(self.train_nofeatures, self.y_train)

		print("ok")
		
		print("elapsed time: {:.1f} seconds".format(time.time() - t0))
		pickle.dump(model, open(self.model_file,"wb"))
		print("saved trained model to", self.model_file)

		print("predicting...", end="")
		y_pred = model.predict(self.test_nofeatures)
		print("ok")
		
		print(classification_report(self.y_test, y_pred))
		print("accuracy: {:.3f}".format(accuracy_score(self.y_test, y_pred)))
		
		pd.concat([self.test_nofeatures, self.y_test, 
				pd.Series(y_pred, index=self.test_nofeatures.index, name="sport_p")], axis=1, 
				join_axes=[self.test_nofeatures.index]).to_csv(self.pred_file)

		print("see predictions on testing data in", self.pred_file)

if __name__ == '__main__':

	from spc_b import WordEndingIsS_ft
	
	SportsClassifier().train()












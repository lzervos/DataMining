from __future__ import division
from ast import literal_eval
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from KNN import KNN
import pandas as pd
import numpy as np
import warnings




if __name__ == '__main__':
	#That is for ignoring all the warnings from the sklearn "library"
	warnings.filterwarnings(module='sklearn*',action='ignore')

	#Here starts the main code
	train_data=pd.read_csv('train_set.csv',converters={"Trajectory":literal_eval},index_col='tripId')
	test_data=pd.read_csv('test_set_a2.csv',sep="\t",converters={"Trajectory":literal_eval})
	#train_data=train_data[:800]
	

	le=preprocessing.LabelEncoder()
	le.fit(train_data["journeyPatternId"])
	y=le.transform(train_data["journeyPatternId"])
	X=train_data["Trajectory"]

	Knn=KNN(n_neighbors=5)
	Knn.fit(X,y)

	Test=test_data["Trajectory"]
	y_pred=Knn.predict(Test)
	predicted_categories=le.inverse_transform(y_pred)

	#X=X[:20]
	#y=y[:20]
	#scores=cross_val_score(Knn,X,y,scoring='accuracy',cv=10,n_jobs=-1)
	#accuracy=scores.mean()

	#print "Accuracy of model is ",accuracy


	dictio={'Test_Trip_ID':[],'Predicted_JourneyPatternID':[]}
	for i in range(len(predicted_categories)):
		dictio['Predicted_JourneyPatternID'].append(predicted_categories[i])
		dictio['Test_Trip_ID'].append(i+1)

	df=pd.DataFrame(data=dictio,columns=['Test_Trip_ID','Predicted_JourneyPatternID'])
	df.to_csv('testSet_JourneyPatternIDs.csv',sep='\t',index=False)

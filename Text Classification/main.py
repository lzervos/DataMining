from __future__ import division
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing,svm
from sklearn.model_selection import GridSearchCV,cross_validate
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
from tempfile import mkdtemp
from shutil import rmtree
from KNN import KNN
from textblob import TextBlob
import scipy.sparse as sp
import warnings
import pandas as pd
import numpy as np


def textblob_tokenizer(str_input):
	blob=TextBlob(str_input.lower())
	tokens=blob.words
	words=[token.stem() for token in tokens]
	return words

def cleanPrep(data):
	prep_list=['that','as','about','above','across','after','against','along','among','around','at','before','behind','below','beneath','beside','between','by','down','during','except','for','from','in','in front of','inside','instead of','into','like','near','of','off','on','onto','on top of','out of','outside','over','past','since','through','to','toward','under','underneath','until','up','upon','with','within','without','according to','because of','by way of','in addition to','front of','in place of','regard to','in spite of','instead of','on account of','out of']
	for index,row in data.iteritems():
		dataWords=row.split()
		Words=[word for word in dataWords if word.lower() not in prep_list]
		result=' '.join(Words)
		data.replace(to_replace=row,value=result)

	return data	


def grid(X,y,pipe):
	Cs=[0.001,0.01,0.1,1,10,100,1000]
	gammas=[0.001,0.01,0.1,1,10,100,1000]
	kernels=['linear','rbf']
	nfolds=None
	param_grid={'C':Cs,'gamma':gammas,'kernel':kernels}
	grid=GridSearchCV(pipe,param_grid,scoring='accuracy',cv=nfolds,n_jobs=-1)
	grid.fit(X,y)
	return grid.best_params_

def wordcloud(train_data):
	wordcloud_list={}
	for cat in np.unique(train_data["Category"]):
		lst=train_data.loc[(train_data["Category"]==cat),["Content"]]
		s=[]
		for val in lst.values:
			s.append(val)

		g=''.join(str(x).encode('utf-8') for x in s)
		g.encode('utf-8')

		stop=set(ENGLISH_STOP_WORDS)
		stop.add("now")
		stop.add("said")
		stop.add("like")
		stop.add("u2013")
		stop.add("u201")
		stop.add("u201d")
		stop.add("u2019")
		stop.add("u2019s")
		wordcloud=WordCloud(stopwords=stop).generate(g)
		wordcloud_list[cat]=wordcloud
		image=wordcloud.to_image()
		image.show()	

	return wordcloud_list


if __name__=='__main__':
	#That is for ignoring all the warnings from the sklearn "library"
	warnings.filterwarnings(module='sklearn*',action='ignore')

	#Here starts the main code
	train_data = pd.read_csv('train_set.csv', sep="\t",index_col=False,encoding='utf-8')
	test_data = pd.read_csv('test_set.csv', sep="\t",index_col=False,encoding='utf-8')
	le = preprocessing.LabelEncoder()
	le.fit(train_data["Category"])
	y = le.transform(train_data["Category"])
	count_vectorizer= CountVectorizer(stop_words=ENGLISH_STOP_WORDS,min_df=0.02,max_df=0.7,analyzer='word',tokenizer=textblob_tokenizer)
	#This is for title
	X = count_vectorizer.fit_transform(train_data['Content'])
	Test= count_vectorizer.transform(test_data['Content'])
	X_Title = count_vectorizer.fit_transform(train_data['Title'])
	Test_Title=count_vectorizer.transform(test_data['Title'])
	
	#count_vectorizer= count_vectorizer.fit(train_data['Content'])
	#X=count_vectorizer.transform(train_data['Content'])
	#Test= count_vectorizer.transform(test_data['Content'])	
	#X_Title=count_vectorizer.transform(train_data['Title'])
	#Test_Title=count_vectorizer.transform(test_data['Title'])


	X1=sp.hstack((X,X_Title))
	Test1=sp.hstack((Test,Test_Title))

	#Here is the transformer and the classifiers
	cachedir=mkdtemp()
	svd=TruncatedSVD(n_components=100)
	Rf=RandomForestClassifier()
	MyMethod=svm.SVC(C=0.001,gamma=0.001,kernel='linear')
	Svm=svm.SVC()
	Mult_NB=MultinomialNB()
	k=int(raw_input("Give me k for neighbors:"))

	Knn=KNN(n_neighbors=k)

	scoring={'acc':'accuracy','prec_macro':'precision_macro','rec_macro':'recall_macro','f1_mac':'f1_macro'}
	mv={'Random_Forest':[],'Naive_Bayes':[],'KNN':[],'SVM':[],'My Method':[]}


	clf_list={'Random_Forest':Rf,'SVM':Svm,'My Method':MyMethod,'KNN':Knn}

	for (nm,clf) in clf_list.iteritems():
		estimators=[('svd',svd),('clf',clf)]
		pipe=Pipeline(steps=estimators)
		pipe.fit(X1,y)

		if nm=='My Method':
			y_pred = pipe.predict(Test1)
			predicted_categories = le.inverse_transform(y_pred)

		scores=cross_validate(pipe,X1,y,scoring=scoring,cv=10,n_jobs=-1,return_train_score=False)

		mv[nm].append(scores['test_acc'].mean())
		mv[nm].append(scores['test_prec_macro'].mean())
		mv[nm].append(scores['test_rec_macro'].mean())
		mv[nm].append(scores['test_f1_mac'].mean())


	Mult_NB.fit(X1,y)	
	scores=cross_validate(Mult_NB,X1,y,scoring=scoring,cv=10,return_train_score=False)


	mv['Naive_Bayes'].append(scores['test_acc'].mean())
	mv['Naive_Bayes'].append(scores['test_prec_macro'].mean())
	mv['Naive_Bayes'].append(scores['test_rec_macro'].mean())
	mv['Naive_Bayes'].append(scores['test_f1_mac'].mean())

	id_data=test_data['Id']
	dictio={'ID':[],'Predicted_Category':[]}
	for i in range(len(predicted_categories)):
		dictio['ID'].append(id_data[i])
		dictio['Predicted_Category'].append(predicted_categories[i])

	out=pd.DataFrame(data=dictio)
	df=pd.DataFrame(data=mv,index=['Accuracy','Precision','Recall','F-Measure'])
	#df.columns.name="Statistic Measure"
	df.index.name="Statistic Measure"

	out.to_csv('testSet_categories.csv',sep='\t',encoding='utf-8',index=False)
	df.to_csv('evaluationMetric_10fold.csv',sep='\t',encoding='utf-8')
	rmtree(cachedir)
	wordcloud_list=wordcloud(train_data)

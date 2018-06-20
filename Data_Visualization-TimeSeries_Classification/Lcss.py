from ast import literal_eval
from haversine import haversine
import pandas as pd
import gmplot
import time


def lcs(test_lst,train_lst,len_test,len_train):
	if len_test==0 or len_train==0:
		return (0, ())
	elif haversine(test_lst[len_test-1],train_lst[len_train-1]) <= 0.2:
		(count, points) = lcs(test_lst,train_lst,len_test-1,len_train-1)
		m_points = points + (test_lst[len_test-1],train_lst[len_train-1])
		return (count + 1, m_points)
	else:
		return max(lcs(test_lst,train_lst,len_test,len_train-1),
					lcs(test_lst,train_lst,len_test-1,len_train))

def lcs_d(X, Y):
    m = len(X)
    n = len(Y)
 
    L = [[None]*(n+1) for i in xrange(m+1)]

    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif haversine(X[i-1], Y[j-1]) <= 0.2:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j] , L[i][j-1])

    return L[m][n]

def lat_lng(row):
	lst=[]
	for tpl in (x for x in row):
		lst.append((tpl[2],tpl[1]))
	return lst	 



def plotMap(lst, test_index, train_index=None, lst2=None):
	langt=[x[0] for x in lst]
	longt=[x[1] for x in lst]

	medianSpot = ( # the 'center' of the route
		min(langt) + (max(langt)-min(langt)) / 2, 
		min(longt) + (max(longt)-min(longt)) / 2
	)
	gmap = gmplot.GoogleMapPlotter(medianSpot[0], medianSpot[1], 12)
	

	if train_index is None:
		filename = "a2_test" + str(test_index+1) + ".html"
		gmap.plot(langt, longt, 'green', edge_width=5)
	else:
		filename = "a2_test" + str(test_index+1) + "_n" + str(train_index) + ".html"
		langt2=[x[0] for x in lst2]
		longt2=[x[1] for x in lst2]
		gmap.plot(langt2, longt2, 'green', edge_width=5)
		gmap.plot(langt, longt, 'red', edge_width=5)

	gmap.draw(filename)	


if __name__ == '__main__':
	train_data = pd.read_csv('train_set.csv',converters={"Trajectory":literal_eval},index_col='tripId')
	test_data = pd.read_csv('test_set_a2.csv',sep="\t",converters={"Trajectory":literal_eval})

	neighbors_times = []

	for index,row in test_data['Trajectory'].iteritems():
		neighbors_counter = 0
		start_time = time.time()
		
		test_lst = lat_lng(row)
		plotMap(test_lst,index)
		matches = []

		for index_tr,row_tr in train_data["Trajectory"].iteritems():
			points = []
			train_lst = lat_lng(row_tr) 
			JP_ID = train_data["journeyPatternId"][index_tr]
			matching_points = lcs_d(test_lst, train_lst)
			matches.append((matching_points, JP_ID, train_lst))

		matches = sorted(matches, reverse=True)[:5]
		Dt = time.time() - start_time
		neighbors_counter = 1

		print "Test trip", index + 1, "Dt:", Dt 
		for tpl in matches:
			print "\t Neighbor", neighbors_counter, "JP_ID", tpl[1], "Matches", tpl[0]
			plotMap(test_lst, index, neighbors_counter, tpl[2])
			neighbors_counter += 1
		




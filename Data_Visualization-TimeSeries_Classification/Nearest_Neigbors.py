from ast import literal_eval
from fastdtw import fastdtw
from haversine import haversine
from lcs import lat_lng
import pandas as pd
import numpy as np
import gmplot 
import time




def plotMap(lst,test_index,train_index=None):
	langt=[x[0] for x in lst]
	longt=[x[1] for x in lst]

	if train_index is None:
		filename = "a1_test" + str(test_index+1) + ".html"
	else:
		filename = "a1_test" + str(test_index+1) + "_n" + str(train_index) + ".html"

	medianSpot = ( # the 'center' of the route
		min(langt) + (max(langt)-min(langt)) / 2, 
		min(longt) + (max(longt)-min(longt)) / 2
	)
	gmap = gmplot.GoogleMapPlotter(medianSpot[0], medianSpot[1], 12)
	gmap.plot(langt,longt,'blue',edge_width=5)
	gmap.draw(filename)	 






if __name__ == '__main__':
	train_data = pd.read_csv('train_set.csv',converters={"Trajectory":literal_eval},index_col='tripId')
	test_data = pd.read_csv('test_set_a1.csv',sep="\t",converters={"Trajectory":literal_eval})

	neighbors_times = []

	for index, row in test_data['Trajectory'].iteritems():
		neighbors_counter = 0
		start_time = time.time()

		test_trip = lat_lng(row)
		plotMap(test_trip, index)
		test_arr = np.array(test_trip)
		matches = []

		for index_tr, row_tr in train_data['Trajectory'].iteritems():
			train_lst = lat_lng(row_tr)
			train_arr = np.array(train_lst)
			JP_ID = train_data["journeyPatternId"][index_tr]
			distance, path = fastdtw(test_arr, train_arr, dist=haversine)
			matches.append((distance, JP_ID, train_lst))

		matches = sorted(matches, key=lambda x:(x[0],x[1]))[:5]
		Dt = time.time() - start_time
		neighbors_counter = 1

		print "Test trip", index + 1, "Dt:", Dt 
		for tpl in matches:
			print "\t Neighbor", neighbors_counter, "JP_ID", tpl[1], "DTW", tpl[0]
			plotMap(tpl[2], index, neighbors_counter)
			neighbors_counter+=1


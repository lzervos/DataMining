from ast import literal_eval
import gmplot
import warnings
import pandas as pd
import numpy as np


def plotMaps(train_data):
	j_number=0
	###We use a set because has O(1) time for search
	lst_files=[]
	journeySet=set()

	for index,row in train_data["journeyPatternId"].iteritems():
		if j_number>=5:
			break

		if row in journeySet:
			continue

		###Gathering all the coordinates of the route	
		langt=[]
		longt=[]
		for (tm,lg,ln) in (x for x in train_data["Trajectory"][index]):
			langt.append(ln)
			longt.append(lg)

		###Here we are using the gmplot library for visualizing the bus routes	
		medianSpot = ( # the 'center' of the route
				min(langt) + (max(langt)-min(langt)) / 2, 
				min(longt) + (max(longt)-min(longt)) / 2
		)
		gmap=gmplot.GoogleMapPlotter(medianSpot[0], medianSpot[1], 12)
		gmap.plot(langt, longt, 'green', edge_width=4)

		filename="Map_" + row + ".html"
		print "Writing JP_ID " + row + " in " + filename
		gmap.draw(filename)	

		lst_files.append(filename)

		###Adding the journeyPatternId to the set 
		journeySet.add(row)
		j_number+=1

	return lst_files



if __name__ == '__main__':
	#That is for ignoring all the warnings from the sklearn "library"
	warnings.filterwarnings(module='sklearn*',action='ignore')

	#Here starts the main code
	train_data=pd.read_csv('train_set.csv',converters={"Trajectory":literal_eval},index_col='tripId')
	files=plotMaps(train_data)

import pandas
import xlwt
import math
from sklearn import neighbors, datasets
from numpy.random import permutation
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

data = pandas.read_csv('processed_data2.csv')
univ = ['Arizona State University','California Institute of Technology','Carnegie Mellon University', 
        'Clemson University','Columbia University','Cornell University', 'George Mason University',
        'Georgia Institute of Technology','Harvard University','Johns Hopkins University','Massachusetts Institute of Technology',
        'New Jersey Institute of Technology', 'New York University', 'Northeastern University','Northwestern University','North Carolina State University',
        'Ohio State University Columbus','Purdue University','Rutgers University New Brunswick/Piscataway',
        'Stanford University','SUNY Buffalo','SUNY Stony Brook','Syracuse University','Texas A and M University College Station',
        'University of Arizona','University of California Davis','University of California Irvine','University of California Los Angeles',
        'University of California San Diego','University of California Santa Barbara','University of California Santa Cruz',
        'University of Cincinnati','University of Colorado Boulder','University of Florida','University of Illinois Chicago',
        'University of Illinois Urbana-Champaign','University of Maryland College Park','University of Massachusetts Amherst',
        'University of Michigan Ann Arbor','University of Minnesota Twin Cities','University of North Carolina Chapel Hill','University of North Carolina Charlotte',
        'University of Pennsylvania','University of Southern California','University of Texas Arlington','University of Texas Austin',
        'University of Texas Dallas','University of Utah','University of Washington','University of Wisconsin Madison',
        'Virginia Polytechnic Institute and State University','Wayne State University','Northwestern University']

list_number = []
list_univ = []
    
for i in range(0,53):
    sum=0
    for j in range(0,13893):
        if univ[i]==(data['univName'][j]):
           sum+=1
    list_univ.append(univ[i])
    list_number.append(sum)
dict = {'UnivName': list_univ, 'Number':list_number}       
data2 = pandas.DataFrame(dict)
data2.to_csv('univ_number.csv')


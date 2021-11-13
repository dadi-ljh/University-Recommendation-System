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

list_int = []
list_greq = []
list_grev = []
list_grea = []
list_gpa = []

for i in range(0,53):
    sum_int=0.
    sum_greq=0.
    sum_grev=0.
    sum_grea=0.
    sum_gpa=0.    
    aver_int=0.
    aver_greq=0.
    aver_grev=0.
    aver_grea=0.
    aver_gpa=0.
    sum1=0.
    for j in range(0,13893):
        if univ[i]==(data['univName'][j]):
           sum_int=sum_int+data['researchExp'][j]
           sum_greq=sum_greq+data['greQ'][j]
           sum_grev=sum_grev+data['greV'][j]
           sum_grea=sum_grea+data['greA'][j]
           sum_gpa=sum_gpa+data['cgpa'][j]  
           sum1=sum1+1.0
    if sum1==0:
        aver_int=0
        aver_greq=0
        aver_grev=0
        aver_grea=0
        aver_gpa=0
    else:
        aver_int=sum_int/sum1
        aver_greq=sum_greq/sum1
        aver_grev=sum_grev/sum1
        aver_grea=sum_grea/sum1
        aver_gpa=sum_gpa/sum1
        
    list_int.append(aver_int)
    list_greq.append(aver_greq)
    list_grev.append(aver_grev)
    list_grea.append(aver_grea)
    list_gpa.append(aver_gpa)

        

dict = {'researchExp': list_int, 'greV':list_grev, 'greQ':list_greq, 'greA':list_grea, 'cgpa':list_gpa}
data2 = pandas.DataFrame(dict)
print(data2)
data2.to_csv('univ.csv')


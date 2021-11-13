import pandas as pd
import math
from sklearn import neighbors, datasets
from numpy.random import permutation
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

def normalize_gpa(data,cgpa,totalcgpa):
    cgpa = data[cgpa].tolist()
    totalcgpa = data[totalcgpa].tolist()
    for i in range(len(cgpa)):
        if totalcgpa[i] != 0:
            cgpa[i] = cgpa[i] / totalcgpa[i]
        else:
            cgpa[i] = 0
    data['cgpa'] = cgpa
    return data

def caldis(x,data_scale_1,data_score,data_rank):
    n=0 
    while n<53:
      Vec = np.vstack([data_scale_1[n,:],data_scale_1[x,:]])
      dist2 = pdist(Vec,'euclidean')
      data_score.append(dist2[0])
      data_rank.append(n)
      n+=1

def perform_tenuniv(x):
    data_original = pd.read_csv('univ.csv')
    data=np.array([data_original.loc[0]])
    k=1
    while k<53:
      data=np.vstack([data,data_original.loc[k]])
      k=k+1 
    minmax_scaler = preprocessing.MinMaxScaler()
    data_scale_1 = minmax_scaler.fit_transform(data[:,1:])
    
    data_schooll=data[0:53,0]
    data_score=[]
    data_rank=[]
    caldis(x,data_scale_1,data_score,data_rank) 

    i=1
    n=0

    while n<52:
      while i+n<53:
          if data_score[n]>data_score[i+n]:
              data_score[n],data_score[i+n]=data_score[i+n],data_score[n]
              data_rank[n],data_rank[i+n]=data_rank[i+n],data_rank[n]
          i+=1
      i=1
      n+=1

    n=0    
    while n<10:
      k=data_rank[n]
      h=(int)(data_schooll[k-1])
      print(univ[h],'\n')
      n+=1
    
data = pd.read_csv('processed_data.csv')
data = data.drop('Unnamed: 0',1)
data = normalize_gpa(data,'cgpa','cgpaScale')
data = data.drop('cgpaScale',1)

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


test_cutoff = math.floor(len(data)/5)
length = len(data)-1
test = data.loc[0:test_cutoff-1]

train = data.loc[test_cutoff:length]
train_output_data = train['univName']

train_input_data = train
train_input_data = train_input_data.drop('univName',1)

test_output_data = test['univName']
test_input_data = test

test_input_data = test_input_data.drop('univName',1)

clf = svm.SVC(gamma='auto')
clf.fit(train_input_data,train_output_data)
predicted_output = clf.predict(test_input_data)

'''
n=0
x=0
f=0
while f<length:
  while n<52:
      if predicted_output[f]==univ[n]:
         x=n
      n+=1   
  perform_tenuniv(x)  
  print("-----------------------------------")
  f+=1'''
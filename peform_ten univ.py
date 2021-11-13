import numpy as np

import pandas as pd

from sklearn import preprocessing

from scipy.spatial.distance import pdist

def caldis(x):
    n=0 
    while n<51:
      Vec = np.vstack([data_scale_1[n,:],data_scale_1[x,:]])
      dist2 = pdist(Vec,'euclidean')
      data_score.append(dist2[0])
      data_rank.append(n)
      #print(dist2,'\n')
      n+=1


univ = ['Arizona State University','California Institute of Technology','Carnegie Mellon University', 
        'Clemson University','Columbia University','Cornell University', 'George Mason University',
        'Georgia Institute of Technology','Harvard University','Johns Hopkins University','Massachusetts Institute of Technology',
        'New Jersey Institute of Technology', 'New York University', 'Northeastern University','Northwestern University',
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

data_original = pd.read_csv('univ.csv')
data=np.array([data_original.loc[0]])
k=1
while k<52:
    data=np.vstack([data,data_original.loc[k]])
    k=k+1    
    
minmax_scaler = preprocessing.MinMaxScaler()

data_scale_1 = minmax_scaler.fit_transform(data[1:])


data_school=data[0:51,1:]

data_schooll=data[0:51,0]


data_score=[]
data_rank=[]

caldis(2)

i=1
n=0


while n<50:
    while i+n<51:
        if data_score[n]>data_score[i+n]:
            data_score[n],data_score[i+n]=data_score[i+n],data_score[n]
            data_rank[n],data_rank[i+n]=data_rank[i+n],data_rank[n]
        i+=1
    i=1
    n+=1
    


n=1    
while n<11:
    k=data_rank[n]
    h=(int)(data_schooll[k])
    print(univ[h],'\n')
    n+=1

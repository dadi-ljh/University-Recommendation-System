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
import tourch
from CNN import model

def normalize_gpa(data1,cgpa,totalcgpa):
    cgpa = data1[cgpa].tolist()
    totalcgpa = data1[totalcgpa].tolist()
    for i in range(len(cgpa)):
        if totalcgpa[i] != 0:
            cgpa[i] = cgpa[i] / totalcgpa[i]
        else:
            cgpa[i] = 0
    data1['cgpa'] = cgpa
    return data1

def caldis(x,data_scale_1,data_score,data_rank):
    n=0 
    while n<53:
      Vec = np.vstack([data_scale_1[n,:],data_scale_1[x,:]])
      dist2 = pdist(Vec,'euclidean')
      data_score.append(dist2[0])
      data_rank.append(n)
      n+=1
 
def TopN(data):
     
      
def eval(data_iter, model, args):
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        #feature.data.t_(), target.data.sub_(1)  # batch first, index align
        with torch.no_grad():
            feature = feature.data.t() # 转置，将[W, batch] 转化为[batch, W]
            target = target.data.sub(1) # 因为label是1,2，所以要减一
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return avg_loss, accuracy      
      
def nerual_network(data):
    assert isinstance(text, str)
    model.eval()
    #print(text)
    # dat = data_field.tokenize(text)
    text = text_field.preprocess(text)
    #print(text)
    
    text = [[text_field.vocab.stoi[x] for x in text]]
    #print(text)
    #os._exit(1)
    
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    #print(x)
    output = model(x)
    predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0]+1]

def perform_tenuniv(x):
    univ_dict = {}
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
      univ_dict[n+1] = univ[h]
      n+=1
    print(univ_dict)   
    
data = pd.read_csv('processed_data.csv')
data = data.drop('Unnamed: 0',1)
data = normalize_gpa(data,'cgpa','cgpaScale')
data = data.drop('cgpaScale',1)

target_data = pd.read_excel('test.xlsx')
target_data = normalize_gpa(target_data,'cgpa','cgpaScale')
target_data = target_data.drop('cgpaScale',1)

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

length = len(data)-1
train = data.loc[0:length]
train_output_data = train['univName']

train_input_data = train
train_input_data = train_input_data.drop('univName',1)

test_input_data = target_data.loc[0:1]
length1 = len(test_input_data)

clf = svm.SVC(gamma='auto')
clf.fit(train_input_data,train_output_data)
predicted_output = clf.predict(test_input_data)

neural_network(train_output_data)

n=0
x=0
f=0
while f<length1:
  while n<52:
      if predicted_output[f]==univ[n]:
         x=n
      n+=1   
  perform_tenuniv(x)  
  '''print("-----------------------------------")'''
  f+=1
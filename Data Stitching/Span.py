import numpy as np
import pandas as pd
import time
import sys
from scipy.sparse import coo_matrix,csc_matrix
from scipy.sparse.linalg import svds,eigs
import itertools
import functools as ft
from pyjarowinkler import distance

def genQgramsFromSentence(q,s):
	return  [s[i:i+q] for i in iter(range((len(s)-q+1)))]

def genQgramsFromList(q,l):
	qgrams=[]
	for s in l:
		qgrams.extend(genQgramsFromSentence(q,str(s)))
	return list(set(qgrams))

def countQ(qram,record):
	return record.count(qram)

def genB1(R,qgrams):
	vgen=([str(rr).count(qq) for qq in qgrams if str(rr).count(qq) != 0] for rr in R)
	qgen=([qgrams.index(qq) for qq in qgrams if str(rr).count(qq) != 0] for rr in R)
	rgen=([R.index(rr) for qq in qgrams if str(rr).count(qq) != 0] for rr in R)
	rch = np.fromiter(itertools.chain(*rgen), np.int)
	qch = np.fromiter(itertools.chain(*qgen), np.int)
	vch = np.fromiter(itertools.chain(*vgen), np.int)
	x=coo_matrix((vch, (rch, qch)), shape=(len(R), len(qgrams)))
	return x   

def delta(x):
	return 1 if x >= 1 else 0

vdelta=np.vectorize(delta)


def genOne(n):
	return np.array(np.matrix(np.ones(n)).T)


def second_largest(numbers):
    count = 0
    n1 = n2 = float('-inf')
    for x in numbers:
        count += 1
        if x > n2:
            if x >= n1:
                n1, n2 = x, n1            
            else:
                n2 = x
    return n2 if count >= 2 else None

def iterBipart(R,ID):
	#print('"""""""""""""""""""')
	start_time2 = time.time()
	qGrams= sorted(genQgramsFromList(3,R))
	#print("--- %s seconds generating qgrams---" % (time.time() - start_time2))
	start_time2 = time.time()
	B1 = genB1(R,qGrams)
	#print("--- %s seconds generating B1---" % (time.time() - start_time2))
	W = np.log(len(R)/np.array([B1.getcol(i).nnz for i in range(B1.shape[1])]))
	B2 = np.array(W)*(B1.toarray())
	#print("--- %s seconds generating B2---" % (time.time() - start_time2))
	start_time2 = time.time()
	Y = np.sqrt(np.power(np.sum(B2,axis=1),2))
	Y = np.power(np.sqrt(np.power(np.sum(B2,axis=1),2)),-1)#here
	Y = np.array(np.matrix(Y).T)
	np.nan_to_num(Y)
	B = np.nan_to_num(np.array(Y*B2))
	#print(B)
	#print("--- %s seconds generating B---" % (time.time() - start_time2))
	start_time2 = time.time()
	one = genOne(len(R))
	D = B.dot(B.T.dot(one)).T[0]		
	C = np.nan_to_num(np.diag(np.power(D,-1/2)).dot(B))
	#print("--- %s seconds generating D then C---" % (time.time() - start_time2))
	start_time2 = time.time()
	#Csc=csc_matrix(C)
	#P, S, Q = svds(Csc,k=Csc.shape[0]-1)
	P, S, Q = np.linalg.svd(C, full_matrices=False)
	#print(len(S),second_largest(list(S)))
	index= list(S).index(second_largest(list(S)))
	#print(index,P[:,index])
	Z=P[:,index]
	Rp=[]
	IDp=[]
	oneP = np.array(np.matrix(np.zeros(len(Z))).T)
	oneN = np.array(np.matrix(np.zeros(len(Z))).T)
	Rn=[]
	IDn=[]
	for i in range(len(Z)):
		if Z[i] >=0:
			Rp.append(R[i])
			IDp.append(ID[i])
			oneP[i] = [1]
		else:
			Rn.append(R[i])
			IDn.append(ID[i])
			oneN[i] = [1]
	if len(Rn) == 0 or len(Rp) == 0 :
		print('------------------>',Rp,Rn)
	#print("--- %s seconds generating the 2 clusters---" % (time.time() - start_time2))
	start_time2 = time.time()
	L=np.dot(np.dot(B.T,one).T,np.dot(B.T,one))-len(R)
	L1 = oneP.T.dot(B.dot(B.T).dot(one))-len(Rp)
	L2 = oneN.T.dot(B.dot(B.T).dot(one))-len(Rn)
	O11 = oneP.T.dot(B.dot(B.T).dot(oneP))-len(Rp)
	O22 = oneN.T.dot(B.dot(B.T).dot(oneN))-len(Rn)
	Q = ((O11/L)-(L1/L)**2)+((O11/L)-(L1/L)**2)
	return(Rp,IDp,Rn,IDn,Q)
    
def abcd(X):
    L=[]
    for z in X:
        L.extend(list(itertools.combinations(z,2)))
    print(L,len(L))
    return set(L)

           
def azureml_main(dataframe1 = None, dataframe2 = None):

    start_time = time.time()	
    X=[]
    R=list(dataframe1['blokingK'])#list(pd.read_csv(dataframe1,encoding='ISO-8859-1'))[-1]
    ID = list(dataframe1['idd'])
    R1,ID1,R2,ID2,Q = iterBipart(R,ID)
    Rs=[]
    IDs=[]
    WXY=[]
    if(Q <= 0):
        X.append(ID)#R
        WXY.append(R)
    else:
        if(len(R1)==0 or len(R2)==0):
            X.append(ID)
            WXY.append(R)
        else:
            Rs.append(R1)
            Rs.append(R2)
            IDs.append(ID1)
            IDs.append(ID2)
    Y=[]
    
    for r in Rs:#first 2
        #print(len(Rs),len(r),len(X))
        if len(X)+len(Y)>=len(R):
            break
        if len(list(set(r))) > 2:
            #print(r)
            R1,ID1,R2,ID2,Q = iterBipart(r,IDs[Rs.index(r)])
            #print(Q)
            if Q <= 0:
                X.append(IDs[Rs.index(r)])
                WXY.append(R)#IDs[Rs.index(r)]
            else:
                if(len(R1)==0 or len(R2)==0):
                    X.append(ID)
                    WXY.append(R)
                else:
                    Rs.append(R1)
                    Rs.append(R2)
                    IDs.append(ID1)
                    IDs.append(ID2)
        elif len(list(set(r))) == 2:
            X.append(IDs[Rs.index(r)])
            WXY.append(r)#IDs[Rs.index(r)]
        else:
            Y.append(IDs[Rs.index(r)])
    print(len(X))
    l=list(abcd(X))
    print(len(l))
    candidate_pairs=pd.DataFrame(l,columns=["id1","id2"])
    print(candidate_pairs,len(candidate_pairs))
    complete_candidate_pairs=candidate_pairs.merge(dataframe1,left_on="id1",right_on="idd").merge(dataframe1,left_on="id2",right_on="idd")
    
    jaro_udf = lambda x,y: round(distance.get_jaro_distance(str(x),str(y),winkler=True, scaling=0.1),7)
    int_udf=lambda x,y: 0 if jaro_udf(x,y)!=1 else 1
    complete_candidate_pairs["label"]=np.vectorize(int_udf)(complete_candidate_pairs['cosmos_customerid_x'],complete_candidate_pairs['cosmos_customerid_y'])
    complete_candidate_pairs["first_name_dist"]=np.vectorize(jaro_udf)(complete_candidate_pairs['first_name_x'],complete_candidate_pairs['first_name_y'])
    complete_candidate_pairs["last_name_dist"]=np.vectorize(jaro_udf)(complete_candidate_pairs['last_name_x'],complete_candidate_pairs['last_name_y'])
    complete_candidate_pairs["address_dist"]=np.vectorize(jaro_udf)(complete_candidate_pairs['address_line1_x'],complete_candidate_pairs['address_line1_y'])
    complete_candidate_pairs["postal_code_dist"]=np.vectorize(int_udf)(complete_candidate_pairs['postal_code_x'],complete_candidate_pairs['postal_code_y'])
    complete_candidate_pairs["email_dist"]=np.vectorize(jaro_udf)(complete_candidate_pairs['email_x'],complete_candidate_pairs['email_y'])
    complete_candidate_pairs["device_id_dist"]=np.vectorize(int_udf)(complete_candidate_pairs['device_id_x'],complete_candidate_pairs['device_id_y'])
    similarity_scores=complete_candidate_pairs[["cosmos_customerid_x","cosmos_customerid_y","label","first_name_dist","last_name_dist","address_dist","postal_code_dist","email_dist","device_id_dist"]]
    return similarity_scores,
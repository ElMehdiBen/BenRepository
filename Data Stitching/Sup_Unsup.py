# Blocking : multiple column (first_name[0]+last_name[0]+postal_code)
# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to 
import pandas as pd

# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
def azureml_main(dataframe1):
    lam=lambda x:x[1][0][0]+x[2][0][0]+str(x[4])
    dataframe1["blocking_key"]=dataframe1.apply(lam,axis=1)
    #dataframe1["blocking_key"]=dataframe1["first_name"].values[0][0]
    return dataframe1,

# Generation of the candidate pairs
# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to 
import pandas as pd

# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
def azureml_main(dataframe1):
    dataframe1=dataframe1.merge(dataframe1,on="blocking_key")
    lam=lambda x:str(x[0])+x[1]+x[2]+x[3]+str(x[4])+x[5]+str(x[6])
    lam1=lambda x:str(x[8])+x[9]+x[10]+x[11]+str(x[12])+x[13]+str(x[14])
    lam2=lambda x:1 if x["check"]<x["check1"] else 2
    dataframe1["check"]=dataframe1.apply(lam,axis=1)
    dataframe1["check1"]=dataframe1.apply(lam1,axis=1)
    dataframe1["check2"]=dataframe1.apply(lam2,axis=1)
    dataframe1=dataframe1.loc[dataframe1["check2"]==1,]
    dataframe1=dataframe1.drop("check",axis=1)
    dataframe1=dataframe1.drop("check1",axis=1)
    dataframe1=dataframe1.drop("check2",axis=1)
    return dataframe1,

# Calculation of the similarity scores	
# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to 
import pandas as pd
import jellyfish as jf
# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
def azureml_main(dataframe1):
    dataframe2=pd.DataFrame()
    jaro_udf_first_name = lambda x: round(jf.jaro_winkler(x["first_name_x"],x["first_name_y"]),7)#approximate comparison
    jaro_udf_last_name = lambda x: round(jf.jaro_winkler(x["last_name_x"],x["last_name_y"]),7)#approximate comparison
    jaro_udf_address_line1 = lambda x: round(jf.jaro_winkler(x["address_line1_x"],x["address_line1_y"]),7)#approximate comparison
    jaro_udf_email = lambda x: round(jf.jaro_winkler(x["email_x"],x["email_y"]),7)#approximate comparison
    int_udf_deviceid = lambda x:1 if x["device_id_x"]==x["device_id_y"] else 0#approximate comparison
    dataframe2["cosmos_customerid_x"]=dataframe1["cosmos_customerid_x"]
    dataframe2["cosmos_customerid_y"]=dataframe1["cosmos_customerid_y"]
    dataframe2["first_name_dist"]=dataframe1.apply(jaro_udf_first_name,axis=1)
    dataframe2["last_name_dist"]=dataframe1.apply(jaro_udf_last_name,axis=1)
    dataframe2["address_dist"]=dataframe1.apply(jaro_udf_address_line1,axis=1)
    dataframe2["email_dist"]=dataframe1.apply(jaro_udf_email,axis=1)
    dataframe2["device_id_dist"]=dataframe1.apply(int_udf_deviceid,axis=1)
    return dataframe2,

# Labelization using K-Means	
# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to 
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
def azureml_main(dataframe1):
    X1=dataframe1[dataframe1.columns[2:7]].values
    
    initModel = np.array([[0,0,0,0,0],
                         [1, 1, 1, 1,1]],np.float64)
    
    model = KMeans(n_clusters=2,init=initModel)
    model.fit(X1)
    dataframe1["label"]=model.labels_
    return dataframe1,


# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 21:23:35 2018

@author: asif
"""
import pandas as pd
import numpy as np

df = pd.DataFrame({'A': [1, 1, 2, 2],
                  'B': [1, 2, 3, 4],
                  'C': np.random.randn(4)})

df
# In[1]:
df.groupby('A').agg('min')

# In[2]:
#convert column to datetimes
df['date_time'] = pd.to_datetime(df['date_time'])
df['Transportation_mode'].unique()
#create helper column for consecutive segment
s = df['Transportation_Mode'].ne(df['Transportation_Mode'].shift()).cumsum().rename('g')

#get top1 and top2 values
f1 = lambda x: x.sort_values(ascending=False).iloc[0]
f1.__name__ = 'Top_1'
f2 = lambda x: x.sort_values(ascending=False).iloc[1]
f2.__name__ = 'Top_2'

d = {'date_time':['first','last'], 'Acceleration':['mean', f1, f2]}

df = df.groupby(['Trip_id','Transportation_Mode',s], sort=False).agg(d)
#flatenning MultiIndex in columns
df.columns = df.columns.map('_'.join)
#MultiIndex in index to columns
df = df.reset_index(level=2, drop=True).reset_index()

# In[test commons]

import datetime as dt
import time as tm

print ('timestamp from jan 1 1970',tm.time())  # this is no of milliseconds passed from jan 1, 1970
dtnow = dt.datetime.fromtimestamp(tm.time())  # timestamp to date
dtnow

print (dtnow.year, dtnow.month, dtnow.day, dtnow.hour, dtnow.minute, dtnow.second)

delta = dt.timedelta(days = 100)
today = dt.date.today()
today - delta # 100 days before today


print (5 in [1,2,3]) # to check wheter 1 is in the list or not
print ('chris' in 'christophe brooks')

olymbic_df.groupby(['Edition','NOC','Medal']).size()



# In[groupby outlier removal]

import pandas as pd
df = pd.DataFrame({'Group': ['A','A','A','B','B','B','B'], 'count': [1.1,11.2,1.1,3.3,3.40,3.3,100.0]})
#print(pd.DataFrame(df.groupby('Group').quantile(.01)['count']))

#df = df[df.loc[:,'count'] < 80]
#df = df[df.loc[:,'count'] > 4].


X=70:1:100; 
% X is the range of %Y=linspace( min_Lat,max_long,100); 
Y=20:1:40; 
LL=length(X)*length(Y); 
for i=1:N 
    for j=1:30 
        for k=1:20 
            if(X(j)<=catalog(i,3)<X(j+1) && Y(k)<=catalog(i,2)<Y(k+1)) Z2=catalog(i,:); 
        end 
    end 
end

# In[]
import io
from io import StringIO
import pandas as pd
txt='''Trip_id   Latitude   Longitude  Acceleration    date_time    Transportation_Mode  
   1    39.98528333 116.3073667 186.6302183   5/26/2007 10:21       Walk   
   1    39.98521667 116.30955   20.69027793   5/26/2007 10:22       Walk   
   1    39.98513333 116.3097667 12.41329907   5/26/2007 10:22       Walk   
   1    39.9845     116.31      35.69170853   5/26/2007 10:25       Bike  
   1    39.98423333 116.3102333 28.01721471   5/26/2007 10:25       Bike  
   1    39.98403333 116.3104333 2921.070572   5/26/2007 10:25       Bike  
   1    39.98518333 116.3446    197.9064152   5/26/2007 10:29       Bike  
   1    39.96858333 116.3471167 409.3939156   5/26/2007 10:31       Walk   
   1    39.9649     116.3473333 174.0008214   5/26/2007 10:31       Walk   
   1    39.96335    116.3470333 500.6336985   5/26/2007 10:32       Walk   
   1    39.95885    116.3474    298.458933    5/26/2007 10:32       Car  
   1    39.95635    116.3486833 1445.861393   5/26/2007 10:32       Car  
   1    39.94336667 116.3499833 116.5939123   5/26/2007 10:34       Car  
   2    39.94231667 116.3499667 133.0986026   5/26/2007 10:34       Walk   
   2    39.94123333 116.3493    1503.18099    5/26/2007 10:34       Walk   
   2    39.9277     116.3497667 12.37086539   5/26/2007 10:36       Car  
   2    39.91055    116.35045   7.897042746   5/26/2007 10:38       Car '''


#df = pd.read_fwf(io.StringIO(txt), header=None, widths=[0, 80], names=['Trip_id', 'Latitude','Longitude','Acceleration','date_time','Transportation_Mode'])
TESTDATA = StringIO(txt)
df = pd.read_csv(TESTDATA, sep=" ")

# In[]

import io
import re

import pandas as pd


def _prepare_pipe_separated_str(str_input):
    substitutions = [
        ('^ *', ''),  # Remove leading spaces
        (' *$', ''),  # Remove trailing spaces
        (r' *\| *', '|'),  # Remove spaces between columns
    ]
    if all(line.lstrip().startswith(' ') and line.rstrip().endswith(' ') for line in str_input.strip().split('\n')):
        substitutions.extend([
            (r'^\|', ''),  # Remove redundant leading delimiter
            (r'\|$', ''),  # Remove redundant trailing delimiter
        ])
    for pattern, replacement in substitutions:
        str_input = re.sub(pattern, replacement, str_input, flags=re.MULTILINE)
    return str_input


def read_pipe_separated_str(str_input):
    """Read a Pandas object from a pipe-separated table contained within a string.

    Example:
        | int_score | ext_score | automation_eligible |
        |           |           | True                |
        | 221.3     | 0         | False               |
        |           | 576       | True                |
        | 300       | 600       | True                |

    The leading and trailing pipes are optional, but if one is present, so must be the other.

    In PyCharm, the "Pipe Table Formatter" plugin has a "Format" feature that can be used to neatly format a table.
    """
    str_input = _prepare_pipe_separated_str(str_input)
    return pd.read_csv(pd.compat.StringIO(str_input), sep=' ')

# In[]
    
str_input = '''        | int_score | ext_score | automation_eligible |
        |           |           | True                |
        | 221.3     | 0         | False               |
        |           | 576       | True                |
        | 300       | 600       | True                |'''
        
a = read_pipe_separated_str(txt)        


# In[ Running machine learning model on dataset]

import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from matplotlib import cm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
#from adspy_shared_utilities import load_crime_dataset
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# In[]
os.chdir(r"C:\Users\asif\Python Scripts\Combined Trajectory Reading\Combined Trajectory_Label_Geolife\singleuseranalysis")
dfsegments = pd.read_table('user141Segments.csv', sep=',')
df = dfsegments.copy()
# In[]
df['date_time_last'] = pd.to_datetime(df['date_time_last'])
le = preprocessing.LabelEncoder()
le.fit(df['time_slice']) 
list(le.classes_)
df['time_slice'] = le.transform(df['time_slice']) 

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
features = df.columns[:4]
y = pd.factorize(df['Transportation_Mode'])

columns_to_keep = [2,5] + list(range(7,52)) 
df = df.iloc[:,columns_to_keep]

X_segments = df.iloc[:,range(1,47)]
y_segments = df['Transportation_Mode']
    #X_crime = crime.ix[:,range(0,88)]
    #y_crime = crime['ViolentCrimesPerPop']
# In[]

X_train, X_test, y_train, y_test = train_test_split(X_segments, y_segments, random_state=0)    
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)

# In[Reading file from weblink]

a = pd.read_csv("")
a.head()


import pandas as pd
import numpy as np

# Define the headers since the data does not have any
headers = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
           "num_doors", "body_style", "drive_wheels", "engine_location",
           "wheel_base", "length", "width", "height", "curb_weight",
           "engine_type", "num_cylinders", "engine_size", "fuel_system",
           "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
           "city_mpg", "highway_mpg", "price"]

# Read in the CSV file and convert "?" to NaN
df = pd.read_csv("https://drive.google.com/open?id=1R_BBL00G_Dlo-6yrovYJp5zEYLwlMPi9", header=None, names=headers, na_values="?" )
df.head()

# In[]
# to check how many vales of attributes in a dataframe are null values
df.isnull().sum() 



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



## User 141

import numpy as np
import os
import glob
import pandas as pd
from geopy.distance import vincenty

import matplotlib.pyplot as plt

# In[1]: 1.1    Read all combined plot and label files into their data frames, combined_df and labels_df.  


# Change the current working directory to the location of 'Combined Trajectory_Label_Geolife' folder.
os.chdir(r"C:\Users\asif\Python Scripts\Combined Trajectory Reading\Combined Trajectory_Label_Geolife")
pltFileList = glob.glob("*.plt")
lblFileList = glob.glob("*.txt")
print (len(pltFileList))
print (len(lblFileList))

# In[Read textfile from file system and write csv backto it] - Output user141_original.csv

user_cols = ['lat','long','tep','ht','altitude','date','time']
tripid=1
filename = 'combined141.plt'
os.chdir(r"C:\Users\asif\Python Scripts\Combined Trajectory Reading\Combined Trajectory_Label_Geolife")

userid = filename[8:][:-4] # extract first 8 characters and last four characters from string using string slicing
trajectory = pd.read_table(filename,skiprows=6, sep=',', names=user_cols)    

path = os.getcwd()+'\\singleuseranalysis'
trajectory.to_csv(os.path.join(path,r'user141_original.csv'),index=False)

# In[Join plot file with label file and write the result backinto filesystem] - Output user141join.csv

trajectory["userid"] = userid # create new columns in trajectory dataframe storing userid
trajectory["date_time"] = pd.to_datetime(trajectory["date"] + " " +trajectory["time"], errors='coerce') # create new columns date_time so that it can be compared with labels data_time column
trajectory['lat'] = pd.to_numeric(trajectory['lat'], errors='coerce') # lat is object, so convert this into float
trajectory.dropna(subset = ['lat', 'long', 'date','time'], inplace = True)
print('userid',userid)
# Reading corresponding label file
labelFile = "labels"+userid+".txt"
label_data = pd.read_table(labelFile)
label_data["Start Time"] = pd.to_datetime(label_data["Start Time"])
label_data["End Time"] = pd.to_datetime(label_data["End Time"])
label_data.rename(columns={'Start Time':'Start_Time','End Time':'End_Time','Transportation Mode':'Transportation_Mode'}, inplace=True)

# Cross Join trajectory and label_data dataframes
trajectory.dropna(inplace=True)
start = pd.merge_asof(trajectory, label_data, left_on='date_time', right_on='Start_Time')
transportation_mode = start['Transportation_Mode'].loc[(start['Start_Time'] < start['date_time']) & (start['date_time'] < start['End_Time'])]
trajectory = pd.concat((trajectory, transportation_mode), axis=1) # add transportation_mode column in trajectory dataframe
trajectory.dropna(inplace=True)

path = os.getcwd()+'\\singleuseranalysis'
trajectory.to_csv(os.path.join(path,r'user141join.csv'),index=False)

# In[]

plt.rcParams['axes.xmargin'] = 0.1
plt.rcParams['axes.ymargin'] = 0.1
#plt.plot(trajectory['long'].values, trajectory['lat'].values)


#plt.plot(trajectory['long'].values, trajectory['lat'].values)
#plt.plot(trajectory['long'].values, trajectory['lat'].values, 'ro');


#plt.plot(trajectory['long'].values, trajectory['lat'].values)
#plt.plot(trajectory['long'].values[::150], trajectory['lat'].values[::150], 'ro');





# In[trajectory duplicate]
trajectorydup = trajectory.copy()

# In[Feature computation and outlier removal]

# Time Delta is in seconds
trajectorydup['time_delta'] = (trajectorydup.date_time - trajectorydup.groupby(['userid']).date_time.shift(1)).dt.seconds.shift(-1) # currently grouped on userid, later group on trip id when trips are identified                        
trajectorydup = trajectorydup.where(trajectorydup['time_delta'] > 0) # remove all points whose date-time value is not is oreder, or next point timestemp is less than previous timestamp
trajectorydup.dropna(inplace=True)

################## divide a track into trip based on threshold value of time_delta > 20 minutes    ####################    
trajectorydup['trip_id'] = (tripid+(trajectorydup['time_delta'] > 1200).cumsum()).shift() 
trajectorydup.dropna(inplace=True)
#grp = trajectorydup.groupby(['userid','trip_id','Transportation_Mode'])
#trajectorydup['segmentid'] = grp['Transportation_Mode'].apply(lambda x: x.ne(x.shift(1)).cumsum())
#trajectorydup['segmentid'] = trajectorydup.groupby([userid,tripid]).Transportation_Mode.apply(lambda x: x.ne(x.shift(1)).cumsum())

trajectorydup['segmentid'] = trajectorydup.groupby(['userid','trip_id']).Transportation_Mode.apply(lambda x: x.ne(x.shift(1)).cumsum())

# Vincenty distance is in meters
trajectorydup['lat_shifted'] = trajectorydup['lat'].shift(-1)
trajectorydup['long_shifted'] = trajectorydup['long'].shift(-1)
trajectorydup['Vincenty_distance'] = trajectorydup.dropna().apply(lambda x: vincenty((x['lat'], x['long']), (x['lat_shifted'], x['long_shifted'])).meters, axis = 1)
trajectorydup.drop(['lat_shifted','long_shifted'], axis=1, inplace=True) # drop temporary columns lat_shifted and long_shifted

# velocity Vi = Li / time_deltai
trajectorydup['velocity'] = trajectorydup['Vincenty_distance'] / trajectorydup['time_delta']
trajectorydup['acceleration'] = (-trajectorydup['velocity'] + trajectorydup.groupby(['trip_id']).velocity.shift(-1))/trajectorydup['time_delta'] # later on group on trip id
trajectorydup['velocity_rate'] = (-trajectorydup['velocity'] + trajectorydup.groupby(['trip_id']).velocity.shift(-1))/trajectorydup['velocity'] # Velcocity Rate - later on group on trip id
trajectorydup['jerk'] = (-trajectorydup['acceleration'] + trajectorydup.groupby(['trip_id']).acceleration.shift(-1))/trajectorydup['time_delta'] # later on group on trip id
trajectorydup['acc_rate'] = (-trajectorydup['acceleration'] + trajectorydup.groupby(['trip_id']).acceleration.shift(-1))/trajectorydup['acceleration'] # Acceleration Rate

trajectorydup['longr']= np.radians(trajectorydup['long'])
trajectorydup['latr'] = np.radians(trajectorydup['lat'])
trajectorydup['y'] = np.sin(trajectorydup.groupby(['trip_id']).longr.shift(-1) - trajectorydup['longr']) * np.cos(trajectorydup.groupby(['trip_id']).latr.shift(-1))
trajectorydup['x'] = np.cos(trajectorydup['latr']) * np.sin(trajectorydup.groupby(['trip_id']).latr.shift(-1)) - np.sin(trajectorydup['latr']) * np.cos(trajectorydup.groupby(['trip_id']).latr.shift(-1)) * np.cos(trajectorydup.groupby(['trip_id']).longr.shift(-1) - trajectorydup['longr'])
trajectorydup['bearing'] = np.degrees(np.arctan2(trajectorydup['y'],trajectorydup['x']))
trajectorydup['bearing_rate'] = trajectorydup.groupby(['trip_id']).bearing.shift(-1) - trajectorydup['bearing']        
trajectorydup['rate_bearing_rate'] = (trajectorydup.groupby(['trip_id']).bearing_rate.shift(-1) - trajectorydup['bearing_rate'])/trajectorydup['time_delta']
trajectorydup.drop(['longr','latr','y','x','tep','ht','altitude'], inplace=True ,axis=1) # drop temporary columns longr, latr, y, x
trajectorydup.dropna(inplace=True)
# remove autliers
res = trajectorydup.groupby("Transportation_Mode")["velocity"].quantile([0.05, 0.95]).unstack(level=1) # only velocity rows need to be deleted, 
trajectorydup = trajectorydup.loc[ (res.loc[ trajectorydup.Transportation_Mode, 0.05].values < trajectorydup.Vincenty_distance.values) & (trajectorydup.Vincenty_distance.values < res.loc[trajectorydup.Transportation_Mode, 0.95].values) ]

# In[write feature file to disk]

trajectorydup.to_csv(os.path.join(path,r'user141.csv'),index=False)

# In[Segmentation]
trajectorydup = trajectorydup
#trajectory = dfAllTrajectorires.dropna()
#trajectory = trajectory [trajectory['Transportation_Mode'] in ('taxi','bike','walk','bus','train')]
#trajectory = trajectory.dropna() # dropna is a must
trajectorydup['date_time'] = pd.to_datetime(trajectorydup['date_time'])

#create helper column for consecutive segment
s = trajectory['Transportation_Mode'].ne(trajectory['Transportation_Mode'].shift()).cumsum().rename('segment_id')

grp = trajectorydup.groupby(['userid','trip_id','Transportation_Mode',s])
trajectorydup = grp.filter(lambda x: len(x)>3) # filter all groups whose length is greater than 3

#get top1 and top2 values
f1 = lambda x: x.sort_values(ascending=False).iloc[0]
f1.__name__ = 'Top_1'
#for top2 return nan if not exist
f2 = lambda x: x.sort_values(ascending=False).iloc[1]
f2.__name__ = 'Top_2'

f3 = lambda x: x.sort_values(ascending=False).iloc[2] 
f3.__name__ = 'Top_3'

f4 = lambda x: len(x[x>10]) # count the frequency of bearing greater than threshold value
f4.__name__ = 'Frequency'

d = {'date_time':['first','last', 'count'], 'acceleration':['mean', f1, f2, f3,'count'], 'velocity':[f1, f2, f3, 'sum' ,'count'], 'bearing':['sum', f1, f2, f3, f4], 'bearing_rate':'sum','rate_bearing_rate':'mean', 'Vincenty_distance':'sum'}

df1 = trajectorydup.groupby(['userid','trip_id','Transportation_Mode',s], sort=False).agg(d)

#flatenning MultiIndex in columns
df1.columns = df1.columns.map('_'.join)
#MultiIndex in index to columns
#df1 = df1.reset_index(level=2, drop=False).reset_index()
df1 = df1.reset_index()

df1 = df1.where(df1['Transportation_Mode'].isin(['walk','car','bus','taxi','bike'])).dropna() # only consider the tranportation modes given in the list
df1['Transportation_Mode'].replace('taxi', 'car',inplace=True) # replace taxi segments with car
df1['time_delta'] = (df1.date_time_first - df1.date_time_last).dt.seconds
df1['mean_velocity'] = df1['Vincenty_distance_sum'] / df1['time_delta']


# In[write segments dataframe to disk file]

path = os.getcwd()+'\\singleuseranalysis'
#trajectory.to_csv(path,'user98a.csv')
df1.to_csv(os.path.join(path,r'user41Segments.csv'),index=False)


# In[smoothing track]

#points[n].latitude = points[n-1].latitude * 0.3 + points[n].latitude * .4 + points[n+1].latitude * .3
#points[n].longitude = points[n-1].longitude * 0.3 + points[n].longitude * .4 + points[n+1].longitude * .3

def smoothing():
    return np.nan

# In[]

# method 1
df = trajectorydup    
res = df.groupby("Transportation_Mode")["Vincenty_distance"].quantile([0.05, 0.95]).unstack(level=1)
df = df.loc[ (res.loc[ df.Transportation_Mode, 0.05].values < df.Vincenty_distance.values) & (df.Vincenty_distance.values < res.loc[df.Transportation_Mode, 0.95].values) ]

# method 2
df = trajectorydup
res = df.groupby("Transportation_Mode")["Vincenty_distance"].quantile([0.05, 0.95]).unstack(level=1)
#df.loc[ (res.loc[ df.Transportation_Mode, 0.05] < df.Vincenty_distance.values) & (df.Vincenty_distance.values < res.loc[df.Transportation_Mode, 0.95]) ]

m1 = (df.Transportation_Mode.map(res[0.05]) < df.Vincenty_distance)
m2 = (df.Vincenty_distance.values < df.Transportation_Mode.map(res[0.95]))

df = df[m1 & m2]
#print (df)  

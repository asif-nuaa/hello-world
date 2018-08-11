## User 141

import numpy as np
import os
import glob
import pandas as pd
from geopy.distance import vincenty
import bisect

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




# In[Deleting points outside Beijing city]
#df_beijing = trajectory.copy()

from shapely import geometry
import geopandas as gpd
#import pandas as pd

#file = r'C:\folder\user141.csv'
crs = {'init': 'epsg:4326'}

bottomLeft = (39.77750000, 116.17944444)
bottomRight = (39.77750000, 116.58888889)
topLeft = (40.04722222, 116.58888889)
topRight = (40.04722222, 116.17944444)

#Create a geoseries holding the single polygon. Coordinates in counter-clockwise order
pointList = [bottomLeft, bottomRight, topLeft, topRight]
poly = geometry.Polygon(pointList)
spoly = gpd.GeoSeries([poly],crs=crs)

#Create geodataframe of points
dfcsv = trajectory
geometry = [geometry.Point(xy) for xy in zip(dfcsv.lat, dfcsv.long)]
dfpoints = gpd.GeoDataFrame(dfcsv, crs=crs, geometry=geometry)

#Create a subset dataframe of points within the polygon
trajectory = dfpoints[dfpoints.within(spoly.geometry.iloc[0])]
print('Number of points within polygon: ', trajectory.shape[0])

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

# In[Grid Representation of a region, Identifying Row and Column of a GPS point]

cols = np.linspace(bottomLeft[1], bottomRight[1], num=18)
rows = np.linspace(bottomLeft[0], topLeft[0], num=15)
trajectory['col'] = np.searchsorted(cols, trajectory['long'])
trajectory['row'] = np.searchsorted(rows, trajectory['lat'])

# In[trajectory duplicate]
trajectorydup = trajectory.copy()

# In[Feature computation and outlier removal]

# Time Delta is in seconds
trajectorydup['time_delta'] = (trajectorydup.date_time - trajectorydup.groupby(['userid']).date_time.shift(1)).dt.seconds.shift(-1) # currently grouped on userid, later group on trip id when trips are identified                        
trajectorydup = trajectorydup.where(trajectorydup['time_delta'] > 0) # remove all points whose date-time value is not is oreder, or next point timestemp is less than previous timestamp
trajectorydup.dropna(inplace=True)
print ('1- Length of trajectorydup = ', len(trajectorydup))
################## divide a track into trip based on threshold value of time_delta > 20 minutes    ####################    
trajectorydup['trip_id'] = (tripid+(trajectorydup['time_delta'] > 1200).cumsum()).shift() 
trajectorydup.dropna(inplace=True)
print ('2- Length of trajectorydup = ', len(trajectorydup))
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
trajectorydup.drop(['longr','latr','y','x','tep','ht','altitude','geometry'], inplace=True ,axis=1) # drop temporary columns longr, latr, y, x
trajectorydup.dropna(inplace=True)

print ('3- Length of trajectorydup = ', len(trajectorydup))
# remove autliers
res = trajectorydup.groupby("Transportation_Mode")["velocity"].quantile([0.05, 0.95]).unstack(level=1) # only velocity rows need to be deleted, 
trajectorydup = trajectorydup.loc[ (res.loc[ trajectorydup.Transportation_Mode, 0.05].values < trajectorydup.Vincenty_distance.values) & (trajectorydup.Vincenty_distance.values < res.loc[trajectorydup.Transportation_Mode, 0.95].values) ]

print ('4- Length of trajectorydup = ', len(trajectorydup))
# In[write feature file to disk]

trajectorydup.to_csv(os.path.join(path,r'user141.csv'),index=False)

# In[Segmentation]

#trajectorydup = trajectorydup
#trajectory = dfAllTrajectorires.dropna()
#trajectory = trajectory [trajectory['Transportation_Mode'] in ('taxi','bike','walk','bus','train')]
#trajectory = trajectory.dropna() # dropna is a must
#trajectorydup['date_time'] = pd.to_datetime(trajectorydup['date_time'])

#create helper column for consecutive segment
#s = trajectorydup['Transportation_Mode'].ne(trajectorydup['Transportation_Mode'].shift()).cumsum().rename('segment_id')

grp = trajectorydup.groupby(['userid','trip_id','Transportation_Mode','segmentid'])
#grp.apply(lambda x: x['velocity'].cov(x['acceleration']))
#grp['cova'] = grp['velocity'].cov(grp['acceleration'])
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

f5 = lambda x: len(x[x<3.4]) # count the stop points with velocity less than threshold value 3.4
f5.__name__ = 'stop_frequency'

f6 = lambda x: len(x[x>0.2]) # count the points with velocity greater than threshold value 0.2
f6.__name__ = 'frequency'

f7 = lambda x: len(x[x>0.25]) # count the points with accelration greater than threshold value 0.25
f7.__name__ = 'frequency'

f8 = lambda x: x.quantile(0.85)
f8.__name__ = '85_percentile'

d = {'date_time':['first','last', 'count'], 
     'acceleration':['mean', f1, f2, f3,'count', f8, 'median', 'min'], 
     'velocity':[f1, f2, f3, f5, 'sum' ,'count', f8, 'median', 'min'], 
     'velocity_rate':f6,
     'acc_rate':f7,
     'bearing':['sum', f1, f2, f3, f4], 
     'bearing_rate':'sum',
     'rate_bearing_rate':'mean', 
     'Vincenty_distance':'sum'}

grp = trajectorydup.groupby(['userid','trip_id','Transportation_Mode','segmentid'])
df1 = trajectorydup.groupby(['userid','trip_id','Transportation_Mode','segmentid'], sort=False).agg(d)

#flatenning MultiIndex in columns
df1.columns = df1.columns.map('_'.join)
#MultiIndex in index to columns
#df1 = df1.reset_index(level=2, drop=False).reset_index()
df1 = df1.reset_index()
df1 = df1.rename(columns={'velocity_stop_frequency' : 'stop_points'})
df1 = df1.rename(columns={'velocity_rate_frequency' : 'velocity_change_pts'})
df1 = df1.rename(columns={'acc_rate_frequency' : 'acc_change_pts'})
df1 = df1.where(df1['Transportation_Mode'].isin(['walk','car','bus','taxi','bike'])).dropna() # only consider the tranportation modes given in the list
df1['Transportation_Mode'].replace('taxi', 'car',inplace=True) # replace taxi segments with car
df1['time_delta'] = (df1.date_time_first - df1.date_time_last).dt.seconds
df1['mean_velocity'] = df1['Vincenty_distance_sum'] / df1['time_delta']
df1['stop_rate'] = df1['stop_points']/df1['Vincenty_distance_sum'] # no of stop points in a segment per unit distance
df1['velocity_changerate'] = df1['velocity_change_pts']/df1['Vincenty_distance_sum'] # no of pts changing velocity per unit distance
df1['acc_changerate'] = df1['acc_change_pts']/df1['Vincenty_distance_sum'] # no of pts changing acceleration per unit distance
df1['time_slice'] = df1['date_time_first'].apply( lambda x: 'T_busy' if ((x.hour>=7 and x.hour<10) or (x.hour>=16 and x.hour<21)) else 'T_idle') # Time_Slice type of the day

# covariance between velocity and acceleration
df_cv = pd.DataFrame()
df_cv['Covariance'] = grp.apply(lambda x: x['velocity'].cov(x['acceleration']))
df_cv = df_cv.reset_index()
df1 = pd.merge(df1, df_cv, how='inner', on=['userid','trip_id','Transportation_Mode','segmentid'])
# In[write segments dataframe to disk file]

path = os.getcwd()+'\\singleuseranalysis'
#trajectory.to_csv(path,'user98a.csv')
df1.to_csv(os.path.join(path,r'user141Segments.csv'),index=False)
df_cv.to_csv(os.path.join(path,r'user141Covariance.csv'),index=False)

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

# In[]


df_cv = pd.DataFrame()
df_cv['Covariance'] = grp.apply(lambda x: x['velocity'].cov(x['acceleration']))
df_cv = df_cv.reset_index()
#ddf1['cov'] = df_cv['Covariance']
#df1.set_index(['userid','trip_id','Transportation_Mode','segment_id'])
#df_cv.set_index(['userid','trip_id','Transportation_Mode','segment_id'])

df1 = pd.merge(df1, df_cv, how='outer', on=['userid','trip_id','Transportation_Mode','segment_id'])
#result = df1.join(df_cv, rsuffix='_y')

# In[]

df1.to_csv(os.path.join(path,r'user141Covariance.csv'),index=False)

# In[Deleting points outside Beijing city]
df_beijing = trajectory.copy()

from shapely import geometry
import geopandas as gpd
#import pandas as pd

#file = r'C:\folder\user141.csv'
crs = {'init': 'epsg:4326'}

#Create a geoseries holding the single polygon. Coordinates in counter-clockwise order
pointList = [(39.77750000, 116.17944444),(39.77750000, 116.58888889),(40.04722222, 116.58888889),(40.04722222, 116.17944444)]
poly = geometry.Polygon(pointList)
spoly = gpd.GeoSeries([poly],crs=crs)

#Create geodataframe of points
dfcsv = df_beijing
geometry = [geometry.Point(xy) for xy in zip(dfcsv.lat, dfcsv.long)]
dfpoints = gpd.GeoDataFrame(dfcsv, crs=crs, geometry=geometry)

#Create a subset dataframe of points within the polygon
subset = dfpoints[dfpoints.within(spoly.geometry.iloc[0])]
print('Number of points within polygon: ', subset.shape[0])
# In[write beijing dataframe to csv]
#Number of points within polygon:  58772
subset.to_csv(os.path.join(path,r'user141Beijing.csv'),index=False)

# In[declaring timeslice]
trajectorydup['date_time'].apply(lambda x: 'TS1' if ((x.hour>=0 and x.hour<7) or (x.hour>=19 and x.hour<24)) else \
             'TS2' if ((x.hour>=7 and x.hour<9) or (x.hour>=16 and x.hour<19)) else 'TS3'  )
#else 'TS2' if (x.hour>9 && x.hour<16)  else 'TS3' if (x.hour >= 16 && x.hour < 19)

#trajectorydup['date_time'].apply(lambda x: 'TS1' if ((x.hour>=0 and x.hour<7) or (x.hour>=19 and x.hour<24)) else 'TS2')


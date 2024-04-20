import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature

output_path = "./output/"
bird_data = pd.read_csv("./data/bird_tracking.csv")
bird_data.info()     # Shows general info about the DataFrame

# 1) Get and plot coords of bird Eric
ix = bird_data.bird_name == "Eric"
x = bird_data.longitude[ix]
y = bird_data.latitude[ix]
plt.figure(figsize=(7,7))
plt.plot(x,y,'.')
plt.grid()
plt.savefig(output_path + "birds_ericCoords.png")



# 2) Get and plot coords of every bird
bird_names = pd.unique(bird_data.bird_name)
plt.figure(figsize=(7,7))
for bird_name in bird_names:
    ix = bird_data.bird_name == bird_name
    x = bird_data.longitude[ix]
    y = bird_data.latitude[ix]
    plt.plot(x,y,'.',label=bird_name)
    plt.grid()
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc="lower right")
plt.savefig(output_path + "birds_birdsCoords.png")



# 3) Examine flight speed
ix = bird_data.bird_name == "Eric"
speed = bird_data.speed_2d[ix]
np.isnan(speed).any() # At least one NaN value in the array?
sum(np.isnan(speed))  # How much NaN values?

ind = np.isnan(speed)
plt.figure(figsize=(8,4))
# Hist only values different than NaN negating isnan() output
plt.hist(speed[~ind], bins=np.linspace(0,30,20), density=True)  
plt.xlabel('2D speed (m/s)')
plt.ylabel('Frequency')
plt.savefig(output_path + 'birds_hist.png')

# There is no need to manage NaN explicitly using pandas!
bird_data.speed_2d.plot(kind='hist', range=[0,30])
plt.xlabel('2D speed')
plt.savefig(output_path + 'birds_pd_hist.png')



# 4) Use datetime for time operations
time1 = datetime.datetime.today()
time2 = datetime.datetime.today()
deltat = time2 - time1 # Timedelta object

# Take date_time entries, remove the UTC and format the remaining as a datetime object, timestamps
# Finally add an extra column to the DataFrame object
timestamps = [] 
for k in range(len(bird_data)):
    timestamps.append(datetime.datetime.strptime\
        (bird_data.date_time.iloc[k][:-3], '%Y-%m-%d %H:%M:%S'))

bird_data['timestamp'] = pd.Series(timestamps, index=bird_data.index)

# Get the difference between each timestamp with the first one with a list comprehension
times = bird_data.timestamp[bird_data.bird_name == 'Eric']
elapsed_time = [time - times[0] for time in times]

# Get elapsed time in days
elapsed_days = np.array(elapsed_time) / datetime.timedelta(days=1)
plt.figure()
plt.plot(elapsed_days)
plt.xlabel('Observation')
plt.ylabel('Elapsed time (days)')
plt.savefig(output_path + 'birds_timeplot.png')



# 5) Plot the daily mean speed (with uneven spaciated datapoints)
next_day = 1
inds = []
daily_mean_speed = []
for (i,t) in enumerate(elapsed_days):
    if t < next_day:
        inds.append(i)
    else:
        daily_mean_speed.append(np.mean(bird_data.speed_2d[inds]))
        next_day += 1
        inds = []

plt.figure(figsize=(8,6))
plt.plot(daily_mean_speed)
plt.xlabel('Day')
plt.ylabel('Mean speed (m/s)')
plt.savefig(output_path + 'birds_meanSpeed.png')



# 6) [Now (august 2019) anaconda distribution has already cartopy] Plot migration bird paths in a real map
proj = ccrs.Mercator()
plt.figure(figsize=(10,10))
ax = plt.axes(projection=proj)  # Create axis object with a given projection
ax.set_extent((-25.0, 20.0, 52.0, 10.0)) 
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
for name in bird_names:
    ix = bird_data['bird_name'] == name
    x = bird_data.longitude[ix]
    y = bird_data.latitude[ix]
    ax.plot(x,y,'.',transform=ccrs.Geodetic(),label=name)
plt.legend(loc='upper left')
plt.savefig(output_path + 'birds_cartopyBirds.png')

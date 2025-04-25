import pandas as pd
import utm
import matplotlib.pyplot as plt

# 1. Load the logfile
file_path = file_path = r"C:\Users\tkmir\OneDrive\Desktop\miniproject2\DJIFlightRecord_2021-03-18_[13-04-51]-TxtLogToCsv.csv"
df = pd.read_csv(file_path, encoding='latin1', low_memory=False)

# 2. Find first video recording segment
video_status = df['CAMERA_INFO.recordState']
start_idx = video_status[video_status.isin(['Start', 'Starting'])].index[0]
stop_idx = video_status[(video_status == 'Stop') & (video_status.index > start_idx)].index[0]
video_segment = df.loc[start_idx:stop_idx].copy()

# 3. Extract GPS and convert to UTM
gps_data = video_segment[['OSD.latitude', 'OSD.longitude']].dropna()
latitudes = gps_data['OSD.latitude'].astype(float).values
longitudes = gps_data['OSD.longitude'].astype(float).values

utm_points = [utm.from_latlon(lat, lon) for lat, lon in zip(latitudes, longitudes)]
utm_easting = [p[0] for p in utm_points]
utm_northing = [p[1] for p in utm_points]

# 4. Plot
plt.figure(figsize=(10, 6))
plt.plot(utm_easting, utm_northing, marker='o', linestyle='-', markersize=2)
plt.title('UAV Flight Path During First Video Recording (UTM)')
plt.xlabel('UTM Easting (m)')
plt.ylabel('UTM Northing (m)')
plt.grid(True)
plt.tight_layout()
plt.show()

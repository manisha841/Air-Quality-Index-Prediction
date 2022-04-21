import pandas as pd
import numpy as np
import matplotlib 
import urllib.request as urllib


df_ktm = pd.read_csv("dataset/raw_data_ktm.csv")


"""# extract Required Information"""

df_ktm = df_ktm[['local',"parameter", "value"]]

df_ktm.head()

"""# separate O3 and pm25"""

df_ktm_o3 = df_ktm[df_ktm['parameter']=="o3"]

df_ktm_pm25 = df_ktm[df_ktm["parameter"]=="pm25"]

df_ktm_pm25.drop(columns ="parameter", inplace=True)

df_ktm_o3.drop(columns="parameter", inplace=True)

df_ktm_pm25.head()

df_ktm_o3.head()

df_ktm_o3.value.unique()

AQI_cat=['good','moderate','unhealthy for sensitive groups','unhealthy','very unhealthy', 'hazardous', 'hazardous']

epa={'O3':[[],[],[0.000, 0.054],[0.055,0.070],[0.071, 0.085],[0.086, 0.105],[0.106,0.200]],
     'PM2.5':[[0.0,12.0],[12.1,35.4],[35.5,55.4],[55.5,150.4],[150.5,250.4],[250.5,350.4],[350.5,500.4]], 
     'AQI':[[0,50],[51,100],[101,150],[151,200],[201,300],[301,400],[401,500]]}

epa

df_ktm_o3 = df_ktm_o3.iloc[::-1]

df_ktm_pm25 = df_ktm_pm25.iloc[::-1]

df_ktm_o3 = df_ktm_o3.rename(columns={"value":"o3_value"})

df_ktm_pm25 = df_ktm_pm25.rename(columns={"value":"pm25_value"})

df_all = df_ktm_o3.merge(df_ktm_pm25, on="local", how="left")

df_all.head()

df_all.isnull().sum()

df_all.shape

epa={'O3':[[0.000, 0.054],[0.0055,0.070],[0.071, 0.085],[0.086,0.105],[0.106,0.200],[0.201,0.250]],
     'PM2.5':[[0.0,12.0],[12.1,35.4],[35.5,55.4],[55.5,150.4],[150.5,250.4],[250.5,350.4]],
     'AQI':[[0,50],[51,100],[101,150],[151,200],[201,300],[301,400]]}

AQI_cat=['good','moderate','unhealthy for sensitive groups','unhealthy','very unhealthy', 'hazardous']

def find_aqi(pollutant,conc):
    i=0
    # print(epa[pollutant][5][1])
    if np.isnan(conc) or conc<0:
        return 0
    if conc > epa[pollutant][5][1]:
        return int(epa['AQI'][5][1])
    for range in epa[pollutant]:
        if conc>= range[0] and conc<=range[1]:
            # print(conc)
            # print(range)
            break
        # print(f"====> {i}")
        i+=1
    
    aqi = (float(epa['AQI'][i][1] - epa['AQI'][i][0])/(range[1]-range[0]))*(conc-range[0]) + epa['AQI'][i][0]
    return(int(aqi))

import numpy as np

find_aqi("PM2.5", np.NaN)

df_all.head()

df_all['AQI_O3'] = df_all['o3_value'].apply(lambda x : find_aqi("O3",x))

df_all['AQI_PM25'] = df_all['pm25_value'].apply(lambda x : find_aqi("PM2.5",x))

df_all.head()

df_all['pollution'] = df_all[['AQI_PM25','AQI_O3']].max(axis=1)

df_all.head()

final_df = df_all[['local','pollution']].set_index("local")

final_df
final_df.to_csv("prepared_final_data.csv")
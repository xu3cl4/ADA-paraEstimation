from datetime import date, datetime

import pandas as pd

ref_1955 = 6.16635504e+10
REFERENCE_DATE = datetime(1955, 1, 1, 0, 0, 0)

# the original range of depths that I averaged over 
points = {
        95 : ["point" + str(num) for num in range(1, 32)],
        110: ["point" + str(num) for num in range(32, 48)]
        }

# the range of depths that Zexuan/Haruko averaged over
'''
points = {
        95 : ["point" + str(num) for num in range(13, 22)],
        110: ["point" + str(num) for num in range(35, 43)]
        }
'''
var_map_sim = {
            'water table': 'depth to water', 
            'Tritium aqueous concentration': 'tritium', 
            'UO2++ sorbed concentration': 'uranium', 
            'Al+++ aqueous concentration': 'aluminum', 
            'NO3- aqueous concentration': 'nitrate', 
            'pH': 'ph' 
        }

scaling_real = {
            'depth to water': 1, 
            'tritium': 3.446263e-14,    # Zexuan: 3.22*1.1e-13 / 1 pCi/mL * 10^3 mL/L * e-12 Ci/pCi * 1/9621 g/Ci * 1/3.01604926 mol/g
            'uranium': 1.235525e-8,     # 1 pCi/L * e-12 Ci/pCi * 1/(3.4e-7) g/Ci * 1/238.05078826 mol/g  
            'aluminum': 3.7062377e-8,   # 1 ug/L * e-6 g/ug * 1/26.98154 mol/g
            'nitrate': 1.6127757645e-5, # 1 mg/L * e-3 g/mg * 1/62.0049 mol/g
            'ph': 1
        }

def timeStamp2datetime(x: int):
    return datetime.fromtimestamp(x + float(datetime.strptime(str(REFERENCE_DATE), "%Y-%m-%d %H:%M:%S").strftime("%s")))


def modify_df_real(df):
    df['COLLECTION_DATE'] = pd.to_datetime(df['COLLECTION_DATE'])
    for attribute in df.columns: 
        if attribute == 'COLLECTION_DATE': continue 
        df[attribute] = df[attribute]*scaling_real[attribute]

    return df 

def modify_df_sim(df, well):
    # drop the unnecessary data (step1)
    df.columns = ['ob_name', 'region', 'functional', 'variable', 'time', 'value']
    df.drop(['ob_name', 'functional'], axis=1, inplace=True)
    df['region'] = df['region'].str.lstrip()
    df['region'] = df['region'].str.replace(r'Well', 'point')
    df = df[df['region'].isin(points[well])].copy()
    
    df['variable'] = df['variable'].copy().str.lstrip()
    df = df[df['variable'].isin(var_map_sim.keys())]
    
    # make conversions
    df['time'] = df['time'] - ref_1955
    df['time'] = df['time'].apply(timeStamp2datetime)
    df['time'] = pd.to_datetime(df['time'])
    df['variable'] = df['variable'].map(var_map_sim)
     
    # drop the unnecessary data (step2: extract the desired period)
    df = df[(df['time'].dt.year >= 1985) & (df['time'].dt.year <= 2025)]
    
    return df 

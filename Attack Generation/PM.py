import numpy as np
import pandas as pd

CONFIG = {
    "buffer_size": 2,
    "max_drift_pm": 25,                
    "min_drift_pm": 10,                
    "drift_intensity_factor_pm": 10000, 
}

def apply_pm_attack(group, target_n, cfg):
    max_start = len(group) - target_n - cfg['buffer_size']
    if max_start <= cfg['buffer_size']:
        return pd.DataFrame()
    
    start = np.random.randint(cfg['buffer_size'], max_start)
    mod_seg = group.iloc[start : start + target_n].copy()
    
    random_fluctuation = np.random.uniform(0.8, 1.2)
    calculated_deg = (cfg['drift_intensity_factor_pm'] / target_n) * random_fluctuation
    drift_deg = max(cfg['min_drift_pm'], min(cfg['max_drift_pm'], calculated_deg))
    drift_deg *= (1 if np.random.rand() > 0.5 else -1)
    
    n = len(mod_seg)
    drift_rad = np.linspace(0, np.radians(drift_deg), n)
    mod_seg['odom_yaw'] = (mod_seg['odom_yaw'] + drift_rad + np.pi) % (2 * np.pi) - np.pi

    half_yaw = mod_seg['odom_yaw'] / 2.0
    mod_seg['odom_ori_z'] = np.sin(half_yaw)
    mod_seg['odom_ori_w'] = np.cos(half_yaw)
    mod_seg['odom_ori_x'] = 0.0
    mod_seg['odom_ori_y'] = 0.0
    
    ox, oy = mod_seg['odom_x'].values, mod_seg['odom_y'].values
    nyaws = mod_seg['odom_yaw'].values
    nx, ny = np.zeros(n), np.zeros(n)
    nx[0], ny[0] = ox[0], oy[0]
    
    for i in range(n - 1):
        dist = np.sqrt((ox[i+1]-ox[i])**2 + (oy[i+1]-oy[i])**2)
        nx[i+1], ny[i+1] = nx[i] + dist * np.cos(nyaws[i]), ny[i] + dist * np.sin(nyaws[i])
    
    mod_seg['odom_x'], mod_seg['odom_y'], mod_seg['label'] = nx, ny, 1
    
    print(f"PM Attack -> Length: {target_n}, Drift Angle: {drift_deg:.2f}°")
    
    return mod_seg

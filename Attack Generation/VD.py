import numpy as np
import pandas as pd

CONFIG = {
    "buffer_size": 2,
    "min_scale_vd": 0.6,
    "max_scale_vd": 1.4
}

def apply_vd_attack(group, target_n, cfg):
    max_start = len(group) - target_n - cfg['buffer_size']
    if max_start <= cfg['buffer_size']:
        return pd.DataFrame()
    
    start = np.random.randint(cfg['buffer_size'], max_start)
    mod_seg = group.iloc[start : start + target_n].copy()
    
    scale = np.random.uniform(cfg['min_scale_vd'], cfg['max_scale_vd'])
    if 0.95 < scale < 1.05: scale += 0.1

    n = len(mod_seg)
    ox, oy = mod_seg['odom_x'].values, mod_seg['odom_y'].values
    yaws = mod_seg['odom_yaw'].values
    nx, ny = np.zeros(n), np.zeros(n)
    nx[0], ny[0] = ox[0], oy[0]
    
    for i in range(n - 1):
        dist_mod = np.sqrt((ox[i+1]-ox[i])**2 + (oy[i+1]-oy[i])**2) * scale
        nx[i+1], ny[i+1] = nx[i] + dist_mod * np.cos(yaws[i]), ny[i] + dist_mod * np.sin(yaws[i])
    
    mod_seg['odom_x'], mod_seg['odom_y'], mod_seg['label'] = nx, ny, 2
    return mod_seg
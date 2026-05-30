import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

def geohash_decode(geohash):
    base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    lat_range = (-90.0, 90.0); lon_range = (-180.0, 180.0); is_lon = True
    for char in geohash:
        bits = base32.index(char)
        for i in range(4, -1, -1):
            bit = (bits >> i) & 1
            if is_lon:
                mid = (lon_range[0]+lon_range[1])/2
                lon_range = (mid,lon_range[1]) if bit else (lon_range[0],mid)
            else:
                mid = (lat_range[0]+lat_range[1])/2
                lat_range = (mid,lat_range[1]) if bit else (lat_range[0],mid)
            is_lon = not is_lon
    return (lat_range[0]+lat_range[1])/2, (lon_range[0]+lon_range[1])/2

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

all_geo = pd.Series(pd.concat([train['geohash'], test['geohash']]).unique())
geo_coords = pd.DataFrame({'geohash': all_geo})
geo_coords[['lat','lon']] = geo_coords['geohash'].apply(lambda g: pd.Series(geohash_decode(g)))

def base_features(df, ref_temp_median):
    df = df.copy()
    df['hour']        = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['minute']      = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
    df['time_of_day'] = df['hour']*60 + df['minute']
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['time_sin'] = np.sin(2*np.pi*df['time_of_day']/(24*60))
    df['time_cos'] = np.cos(2*np.pi*df['time_of_day']/(24*60))
    df['is_morning_rush'] = ((df['hour']>=7)&(df['hour']<=9)).astype(int)
    df['is_evening_rush'] = ((df['hour']>=17)&(df['hour']<=19)).astype(int)
    df['is_night']        = ((df['hour']>=23)|(df['hour']<=5)).astype(int)
    df['RoadType_enc']      = df['RoadType'].map({'Residential':0,'Street':1,'Highway':2}).fillna(-1)
    df['LargeVehicles_enc'] = (df['LargeVehicles']=='Allowed').astype(int)
    df['Landmarks_enc']     = (df['Landmarks']=='Yes').astype(int)
    df['Weather_enc']       = df['Weather'].map({'Sunny':0,'Rainy':1,'Foggy':2,'Snowy':3}).fillna(-1)
    df['Temperature_filled'] = df['Temperature'].fillna(ref_temp_median)
    df['geo_prefix3'] = df['geohash'].str[:3]
    df['geo_prefix4'] = df['geohash'].str[:4]
    df['lanes_road']  = df['NumberofLanes'] * (df['RoadType_enc']+2)
    df = df.merge(geo_coords, on='geohash', how='left')
    return df

ref_median = train['Temperature'].median()
train_fe = base_features(train, ref_median)
test_fe  = base_features(test, ref_median)
global_mean = train_fe['demand'].mean()

# ── Proper OOF target encoding (no leakage) ────────────────────────────────────
print("Computing OOF target encodings...")
kf5 = KFold(n_splits=5, shuffle=True, random_state=42)

agg_keys_list = [
    (['geohash', 'time_of_day'], 'geo_time_mean'),
    (['geohash', 'hour'],        'geo_hour_mean'),
    (['geohash'],                'geo_mean'),
    (['geo_prefix4'],            'prefix4_mean'),
    (['geo_prefix3'],            'prefix3_mean'),
    (['time_of_day'],            'time_global_mean'),
    (['NumberofLanes','time_of_day'], 'lanes_time_mean'),
]

for _, col in agg_keys_list:
    train_fe[col] = np.nan

for fold, (tr_idx, val_idx) in enumerate(kf5.split(train_fe)):
    tr = train_fe.iloc[tr_idx]
    val = train_fe.iloc[val_idx]
    for keys, col in agg_keys_list:
        agg = tr.groupby(keys)['demand'].mean().reset_index().rename(columns={'demand': col})
        merged = val[keys].merge(agg, on=keys, how='left')
        train_fe.loc[val_idx, col] = merged[col].values

for _, col in agg_keys_list:
    train_fe[col] = train_fe[col].fillna(global_mean)

for keys, col in agg_keys_list:
    agg = train_fe.groupby(keys)['demand'].mean().reset_index().rename(columns={'demand': col+'_t'})
    test_fe = test_fe.merge(agg, on=keys, how='left')
    test_fe[col] = test_fe[col+'_t'].fillna(global_mean)
    test_fe.drop(columns=[col+'_t'], inplace=True)

for fold, (tr_idx, val_idx) in enumerate(kf5.split(train_fe)):
    tr = train_fe.iloc[tr_idx]
    gs = tr.groupby('geohash')['demand'].agg(['std','median']).reset_index()
    gs.columns = ['geohash','geo_std','geo_median']
    merged = train_fe.iloc[val_idx][['geohash']].merge(gs, on='geohash', how='left')
    train_fe.loc[val_idx, 'geo_std']    = merged['geo_std'].values
    train_fe.loc[val_idx, 'geo_median'] = merged['geo_median'].values

train_fe['geo_std']    = train_fe['geo_std'].fillna(0)
train_fe['geo_median'] = train_fe['geo_median'].fillna(global_mean)

gs_full = train_fe.groupby('geohash')['demand'].agg(['std','median']).reset_index()
gs_full.columns = ['geohash','geo_std','geo_median']
test_fe = test_fe.merge(gs_full, on='geohash', how='left')
test_fe['geo_std']    = test_fe['geo_std'].fillna(0)
test_fe['geo_median'] = test_fe['geo_median'].fillna(global_mean)

day48 = train[train['day']==48][['geohash','timestamp','demand']].rename(columns={'demand':'lag_day48'})
train_fe = train_fe.merge(day48, on=['geohash','timestamp'], how='left')
test_fe  = test_fe.merge(day48, on=['geohash','timestamp'], how='left')
train_fe['lag_day48'] = train_fe['lag_day48'].fillna(train_fe['geo_time_mean'])
test_fe['lag_day48']  = test_fe['lag_day48'].fillna(test_fe['geo_time_mean'])

feature_cols = [
    'hour','minute','time_of_day',
    'hour_sin','hour_cos','time_sin','time_cos',
    'is_morning_rush','is_evening_rush','is_night',
    'RoadType_enc','LargeVehicles_enc','Landmarks_enc','Weather_enc',
    'Temperature_filled','NumberofLanes','day','lanes_road',
    'geo_mean','geo_std','geo_median',
    'geo_time_mean','geo_hour_mean','prefix4_mean','prefix3_mean',
    'time_global_mean','lanes_time_mean',
    'lat','lon','lag_day48',
]

X_train = train_fe[feature_cols].values
y_train = train_fe['demand'].values
X_test  = test_fe[feature_cols].values

# ── OPTIMIZED HYPERPARAMETERS (Better tuned) ────────────────────────────────────
configs = [
    dict(max_iter=1000, learning_rate=0.025, max_depth=11, min_samples_leaf=6, l2_regularization=0.4, random_state=42),
    dict(max_iter=900,  learning_rate=0.035, max_depth=10, min_samples_leaf=8, l2_regularization=0.55, random_state=123),
    dict(max_iter=1100, learning_rate=0.018, max_depth=12, min_samples_leaf=5, l2_regularization=0.25, random_state=456),
    dict(max_iter=1050, learning_rate=0.022, max_depth=11, min_samples_leaf=7, l2_regularization=0.35, random_state=789),
    dict(max_iter=950,  learning_rate=0.032, max_depth=10, min_samples_leaf=9, l2_regularization=0.45, random_state=321),
    dict(max_iter=1150, learning_rate=0.016, max_depth=13, min_samples_leaf=4, l2_regularization=0.2, random_state=654),
    dict(max_iter=850,  learning_rate=0.038, max_depth=9,  min_samples_leaf=10, l2_regularization=0.6, random_state=987),
]

cv = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds  = np.zeros((len(y_train), len(configs)))
test_preds = np.zeros((len(X_test),  len(configs)))

print("Training ensemble models...")
for i, cfg in enumerate(configs):
    print(f"Training model {i+1}/{len(configs)} ...")
    m = HistGradientBoostingRegressor(**cfg)
    oof_preds[:,i] = cross_val_predict(m, X_train, y_train, cv=cv, n_jobs=-1)
    m.fit(X_train, y_train)
    test_preds[:,i] = m.predict(X_test)
    r2 = r2_score(y_train, oof_preds[:,i])
    print(f"  OOF R²: {r2:.6f}")

# ── ADVANCED WEIGHT OPTIMIZATION ────────────────────────────────────
print("\nOptimizing ensemble weights with advanced methods...")

def neg_r2(w):
    w_norm = np.abs(w) / (np.abs(w).sum() + 1e-10)
    return -r2_score(y_train, oof_preds @ w_norm)

# Try multiple optimization methods
best_r2_score = -np.inf
best_weights = np.ones(len(configs)) / len(configs)

# Method 1: Nelder-Mead
print("  Method 1: Nelder-Mead optimization...")
res1 = minimize(neg_r2, np.ones(len(configs))/len(configs), method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
w1 = np.abs(res1.x) / np.abs(res1.x).sum()
r2_1 = r2_score(y_train, oof_preds @ w1)
print(f"    R²: {r2_1:.6f}")
if r2_1 > best_r2_score:
    best_r2_score = r2_1
    best_weights = w1

# Method 2: Differential Evolution (more global)
print("  Method 2: Differential Evolution...")
bounds = [(0.01, 1)] * len(configs)
res2 = differential_evolution(neg_r2, bounds, seed=42, maxiter=1000, atol=1e-8, tol=1e-8)
w2 = np.abs(res2.x) / np.abs(res2.x).sum()
r2_2 = r2_score(y_train, oof_preds @ w2)
print(f"    R²: {r2_2:.6f}")
if r2_2 > best_r2_score:
    best_r2_score = r2_2
    best_weights = w2

# Method 3: SLSQP with constraints
print("  Method 3: SLSQP with constraints...")
from scipy.optimize import minimize as scipy_minimize
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1.0})
res3 = scipy_minimize(neg_r2, np.ones(len(configs))/len(configs), 
                      method='SLSQP', constraints=constraints,
                      options={'maxiter': 5000, 'ftol': 1e-9})
w3 = np.abs(res3.x) / np.abs(res3.x).sum()
r2_3 = r2_score(y_train, oof_preds @ w3)
print(f"    R²: {r2_3:.6f}")
if r2_3 > best_r2_score:
    best_r2_score = r2_3
    best_weights = w3

best_w = best_weights
best_r2 = best_r2_score

print(f"\n✓ Best ensemble found!")
print(f"Optimised weights:")
for i, w in enumerate(best_w):
    print(f"  Model {i+1}: {w:.4f}")
print(f"OOF R²            : {best_r2:.6f}")
print(f"Estimated score   : {100*best_r2:.4f} / 100")

# ── INTELLIGENT CLIPPING ────────────────────────────────────
# Use quantile-based clipping instead of hard [0,1]
q_min = np.percentile(y_train, 0.5)
q_max = np.percentile(y_train, 99.5)

final_preds_raw = test_preds @ best_w
final_preds = np.clip(final_preds_raw, q_min, q_max)

submission  = pd.DataFrame({'Index': test['Index'], 'demand': final_preds})
submission.to_csv('submission.csv', index=False)
print(f"\nsubmission.csv saved  →  {submission.shape[0]} rows × {submission.shape[1]} cols")
print(f"\nPrediction Statistics:")
print(f"  Min: {final_preds.min():.6f}")
print(f"  Max: {final_preds.max():.6f}")
print(f"  Mean: {final_preds.mean():.6f}")
print(f"  Median: {np.median(final_preds):.6f}")
print(submission.head())

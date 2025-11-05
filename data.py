import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "Formated_Data" / "Experiment_1"

# find all video_*.csv under any numbered subfolder
files = sorted(str(p) for p in DATA_ROOT.rglob("video_*.csv"))

print(f"[INFO] script dir = {ROOT}")
print(f"[INFO] data root   = {DATA_ROOT}")
print(f"[INFO] matched files: {len(files)}")
for p in files[:8]:
    print("  -", p)

assert len(files) > 0, "No CSV files found; check paths."

def quat_to_euler_xyz(qx,qy,qz,qw):
    sinr = 2*(qw*qx + qy*qz); cosr = 1-2*(qx*qx+qy*qy); roll = np.arctan2(sinr, cosr)
    sinp = 2*(qw*qy - qz*qx); pitch = np.arcsin(np.clip(sinp, -1, 1))
    siny = 2*(qw*qz + qx*qy); cosy = 1-2*(qy*qy+qz*qz); yaw = np.arctan2(siny, cosy)
    return yaw, pitch, roll

def load_and_resample(csv_path, hz=90.0):
    df = pd.read_csv(csv_path).drop_duplicates()

    # time -> seconds, sort, de-dup
    t = pd.to_datetime(df['Timestamp'], utc=True).astype('int64') / 1e9
    df = df.assign(_t=t).sort_values('_t').drop_duplicates(subset=['_t'], keep='last')

    # quat -> euler
    yaw, pitch, roll = quat_to_euler_xyz(
        df['UnitQuaternion.x'].to_numpy(np.float32),
        df['UnitQuaternion.y'].to_numpy(np.float32),
        df['UnitQuaternion.z'].to_numpy(np.float32),
        df['UnitQuaternion.w'].to_numpy(np.float32),
    )
    yaw = (np.unwrap(yaw) + np.pi) % (2*np.pi) - np.pi
    pitch = np.clip(np.unwrap(pitch), -np.pi/2, np.pi/2)

    # original (possibly sparse) timeline
    tsec = df['_t'].to_numpy(np.float64)

    # resample timeline
    step = 1.0 / hz
    t_new = np.arange(tsec[0], tsec[-1] + 1e-9, step)

    # resample angles
    yaw_i   = np.interp(t_new, tsec, yaw).astype(np.float32)
    pitch_i = np.interp(t_new, tsec, pitch).astype(np.float32)

    # resample head positions if present
    out = {
        'timestamp_s': t_new.astype(np.float32),
        'yaw':   yaw_i,
        'pitch': pitch_i,
    }
    for col in ('HmdPosition.x','HmdPosition.y','HmdPosition.z'):
        if col in df.columns:
            out[col] = np.interp(
                t_new, tsec, df[col].to_numpy(np.float32)
            ).astype(np.float32)

    return pd.DataFrame(out)


def make_windows(df, T=20, horizon=1, add_xyz_vel=False):
    yaw   = df['yaw'].to_numpy(np.float32)
    pitch = df['pitch'].to_numpy(np.float32)
    t     = df['timestamp_s'].to_numpy(np.float32)
    step  = float(np.median(np.diff(t))) if len(t) > 1 else 1/60

    # finite differences (velocity/acceleration)
    yaw_d    = np.gradient(yaw, step).astype(np.float32)
    pitch_d  = np.gradient(pitch, step).astype(np.float32)
    yaw_dd   = np.gradient(yaw_d, step).astype(np.float32)
    pitch_dd = np.gradient(pitch_d, step).astype(np.float32)

    feats = [np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch),
             yaw_d, pitch_d, yaw_dd, pitch_dd]

    if add_xyz_vel:
        for col in ('HmdPosition.x','HmdPosition.y','HmdPosition.z'):
            if col in df.columns:
                pos = df[col].to_numpy(np.float32)
                vel = np.gradient(pos, step).astype(np.float32)
                feats.append(vel)  # +3 dims

    F = np.stack(feats, axis=1)  # (N, in_dim)
    X, Y = [], []
    for i in range(T, len(F)-horizon+1):
        X.append(F[i-T:i])
        y, p = yaw[i+horizon-1], pitch[i+horizon-1]
        cp, sp = np.cos(p), np.sin(p); cy, sy = np.cos(y), np.sin(y)
        Y.append([cp*cy, sp, cp*sy])   # unit vector target
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


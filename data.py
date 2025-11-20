import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "Formated_Data" / "Experiment_1"

# find all video_*.csv under any numbered subfolder
# files = sorted(str(p) for p in DATA_ROOT.rglob("video_*.csv"))
#
# print(f"[INFO] script dir = {ROOT}")
# print(f"[INFO] data root   = {DATA_ROOT}")
# print(f"[INFO] matched files: {len(files)}")
# for p in files[:8]:
#     print("  -", p)
#
# assert len(files) > 0, "No CSV files found; check paths."

def quat_to_euler_xyz(qx,qy,qz,qw):
    sinr = 2*(qw*qx + qy*qz); cosr = 1-2*(qx*qx+qy*qy); roll = np.arctan2(sinr, cosr)
    sinp = 2*(qw*qy - qz*qx); pitch = np.arcsin(np.clip(sinp, -1, 1))
    siny = 2*(qw*qz + qx*qy); cosy = 1-2*(qy*qy+qz*qz); yaw = np.arctan2(siny, cosy)
    return yaw, pitch, roll

class KalmanFilter:
    def __init__(self, process_var=1e-4, meas_var=1e-2):
        self.process_var = process_var  #variance (expected change between timestamps)
        self.meas_var = meas_var        # measurement variance (how noisy the measurements are)
        self.x = None  # estimated value
        self.P = None  # estimated uncertainty

    def filter(self, z):
        """
        z: numpy array of measurements
        returns: numpy array of filtered values
        """
        x_est = np.zeros_like(z)
        for i, zi in enumerate(z):
            if self.x is None:
                # Initialize first measurement
                self.x = zi
                self.P = 1.0
            else:
                # Prediction step
                self.P = self.P + self.process_var
                # Update step
                K = self.P / (self.P + self.meas_var)
                self.x = self.x + K * (zi - self.x)
                self.P = (1 - K) * self.P
            x_est[i] = self.x
        return x_est

def load_and_resample(csv_path, hz=90.0, kalman_filter = True):
    df = pd.read_csv(csv_path)

    # Guard: required columns present
    req = ['Timestamp','UnitQuaternion.x','UnitQuaternion.y','UnitQuaternion.z','UnitQuaternion.w']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing required columns {missing}")

    # Parse timestamps (coerce errors) and drop rows without usable time
    t = pd.to_datetime(df['Timestamp'], utc=True, errors='coerce').astype('int64') / 1e9
    df = df.assign(_t=t).dropna(subset=['_t']).sort_values('_t')

    # Strictly enforce strictly increasing timeline (last-wins on ties)
    df = df.drop_duplicates(subset=['_t'], keep='last')
    tsec = df['_t'].to_numpy(np.float64)
    if len(tsec) < 3:
        # too short to resample/gradient meaningfully
        return pd.DataFrame(columns=['timestamp_s','yaw','pitch'])

    # quat -> euler
    yaw, pitch, roll = quat_to_euler_xyz(
        df['UnitQuaternion.x'].to_numpy(np.float32),
        df['UnitQuaternion.y'].to_numpy(np.float32),
        df['UnitQuaternion.z'].to_numpy(np.float32),
        df['UnitQuaternion.w'].to_numpy(np.float32),
    )
    # unwrap & bound
    yaw   = (np.unwrap(yaw) + np.pi) % (2*np.pi) - np.pi
    pitch = np.clip(np.unwrap(pitch), -np.pi/2, np.pi/2)

    # resample timeline
    step = float(1.0 / hz)
    t_new = np.arange(tsec[0], tsec[-1] + 1e-9, step, dtype=np.float64)

    # linear interp (handles monotone tsec); cast to f32
    yaw_i   = np.interp(t_new, tsec, yaw).astype(np.float32)
    pitch_i = np.interp(t_new, tsec, pitch).astype(np.float32)

    if kalman_filter:
        kf_yaw = KalmanFilter()
        kf_pitch = KalmanFilter()
        yaw_i = kf_yaw.filter(yaw_i)
        pitch_i = kf_pitch.filter(pitch_i)

    out = {
        'timestamp_s': t_new.astype(np.float32),
        'yaw':   yaw_i,
        'pitch': pitch_i,
    }
    for col in ('HmdPosition.x','HmdPosition.y','HmdPosition.z'):
        if col in df.columns:
            out[col] = np.interp(t_new, tsec, df[col].to_numpy(np.float32)).astype(np.float32)

    out_df = pd.DataFrame(out)
    # Final guard: drop any NaNs that may have crept in
    out_df = out_df.replace([np.inf, -np.inf], np.nan).dropna()
    return out_df


def make_windows(df, T=20, horizon=1, add_xyz_vel=False):
    if len(df) < T + horizon:
        return np.zeros((0, T, 2), np.float32), np.zeros((0, 3), np.float32)

    yaw   = df['yaw'].to_numpy(np.float32)
    pitch = df['pitch'].to_numpy(np.float32)
    t     = df['timestamp_s'].to_numpy(np.float32)

    # step from data, with floor at ~1/240s
    dt = np.diff(t)
    med = float(np.median(dt)) if len(dt) else 1/60
    step = max(med, 1/240)

    # finite differences
    yaw_d    = np.gradient(yaw, step).astype(np.float32)
    pitch_d  = np.gradient(pitch, step).astype(np.float32)
    yaw_dd   = np.gradient(yaw_d, step).astype(np.float32)
    pitch_dd = np.gradient(pitch_d, step).astype(np.float32)

    # robust clip the derivative channels (pre-zscore)
    def rclip(x, q=0.999):
        lo, hi = np.quantile(x, 1-q), np.quantile(x, q)
        return np.clip(x, lo, hi)
    yaw_d, pitch_d   = rclip(yaw_d),   rclip(pitch_d)
    yaw_dd, pitch_dd = rclip(yaw_dd), rclip(pitch_dd)

    feats = [np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch),
             yaw_d, pitch_d, yaw_dd, pitch_dd]

    if add_xyz_vel:
        for col in ('HmdPosition.x','HmdPosition.y','HmdPosition.z'):
            if col in df.columns:
                pos = df[col].to_numpy(np.float32)
                vel = np.gradient(pos, step).astype(np.float32)
                vel = rclip(vel)
                feats.append(vel)  # +3

    F = np.stack(feats, axis=1).astype(np.float32)

    X, Y = [], []
    for i in range(T, len(F)-horizon+1):
        X.append(F[i-T:i])
        y, p = yaw[i+horizon-1], pitch[i+horizon-1]
        cp, sp = np.cos(p), np.sin(p); cy, sy = np.cos(y), np.sin(y)
        vec = np.array([cp*cy, sp, cp*sy], dtype=np.float32)
        # assert unit-ish
        # (no exception at runtime; just safe guard)
        n = np.linalg.norm(vec) + 1e-8
        Y.append(vec / n)

    return np.asarray(X, np.float32), np.asarray(Y, np.float32)




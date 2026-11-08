"""Field Note #21 evaluation: stereo VO vs gyroscope (rotation) and vs accelerometer (translation), three real-work clips.
Inputs per clip: traj_<tag>.npy, vo_<tag>.json (from vo.py), the clip's SEI (video) and IMU CSV.
Outputs (fn21/): steps_all.csv, drift_curve.csv, accel_series.csv, summary_by_clip.csv, summary.csv"""
import sys, json, gzip, numpy as np, pandas as pd, cv2
from scipy.signal import savgol_filter
sys.path.insert(0, '/home/claude/fn/common'); sys.path.insert(0, '/home/claude/hero')
from odl_calib import device_calib
from sei import sei
CL = '/home/claude/odl-capture-analysis/clips/'
CLIPS = [('A', '20261104_104929_00002_s0060', '2026-11-04 · bagging carrots'), ('B', '20261102_104245_00004_s0060', '2026-11-02 · tying green onions'),
         ('C', '20261104_093804_00002_s0120', '2026-11-04 · bagging vegetables')]
C = device_calib(); KL, DL, KR, DR = C['cam0']['K'], C['cam0']['D'], C['cam1']['K'], C['cam1']['D']; T = C['T_c1_c0']
R1 = cv2.fisheye.stereoRectify(KL, DL, KR, DR, (1600, 1200), T[:3, :3], T[:3, 3], flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=(1600, 1200), balance=0.0)[0]
Rri = R1 @ C['cam0']['T_cam_imu'][:3, :3]; shift = C['cam0']['shift'] * 1e6
B14 = np.array([0.00277138, -0.00144135, -0.00546019])   # static gyro bias, rad/s, Field Note #14 (2026-10-26, mean of 5 static segments)
ang = lambda R: float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))
ST, DR_, AC, SU = [], [], [], []
for cid, clip, label in CLIPS:
    tag = f'{clip}_0_59'; pre = CL + clip
    P = np.load(f'traj_{tag}.npy'); fr = P[:, 0].astype(int); Tw = P[:, 1:].reshape(-1, 4, 4)
    V = json.load(open(f'vo_{tag}.json')); steps = V['steps']
    SL = sei(f'{pre}_left.mp4'); tcu = SL[:, 1] + shift + SL[:, 2] / 2; tc = tcu[fr] / 1e6
    I = np.loadtxt(gzip.open(f'{pre}_imu.csv.gz', 'rt'), delimiter=',', skiprows=1)
    G = I[I[:, 1] == 2]; A = I[I[:, 1] == 4]
    def gyro_rot(ta, tb, b=np.zeros(3)):
        m = (G[:, 0] > ta) & (G[:, 0] <= tb); ts = np.r_[ta, G[m, 0]]; Rm = np.eye(3)
        for k, w in enumerate(G[m, 2:5]): Rm = Rm @ cv2.Rodrigues((w - b) * (ts[k + 1] - ts[k]) / 1e6)[0]
        return Rm
    okb = {s['b']: s for s in steps if s['ok']}; t00 = tcu[steps[0]['a']]
    Rg0 = np.eye(3); Rg1 = np.eye(3); fit = []
    for k in range(len(fr)):
        if k:
            ra, rb = tcu[fr[k - 1]], tcu[fr[k]]
            G0 = Rri @ gyro_rot(ra, rb) @ Rri.T; G1 = Rri @ gyro_rot(ra, rb, B14) @ Rri.T
            Rg0 = Rg0 @ G0; Rg1 = Rg1 @ G1
            s_ = okb.get(fr[k]); Rv = Tw[k - 1][:3, :3].T @ Tw[k][:3, :3]
            if s_ is not None:
                fit.append(cv2.Rodrigues((Rri.T @ G0 @ Rri).T @ (Rri.T @ Rv @ Rri))[0].ravel() / ((rb - ra) / 1e6))
                ST.append(dict(clip=cid, t_s=(rb - t00) / 1e6, ok=True, inliers=s_['inl'], rot_gyro_deg=ang(G1), rot_vo_deg=ang(Rv), err_raw_deg=ang(Rv @ G0.T), err_deg=ang(Rv @ G1.T)))
            else: ST.append(dict(clip=cid, t_s=(rb - t00) / 1e6, ok=False, inliers=[x for x in steps if x['b'] == fr[k]][0].get('inl')))
        DR_.append(dict(clip=cid, t_s=tc[k] - tc[0], drift_raw_deg=ang(Tw[k][:3, :3] @ Rg0.T), drift_deg=ang(Tw[k][:3, :3] @ Rg1.T)))
    bfit = -np.median(np.array(fit), 0)            # gyro bias implied by vision (IMU frame, rad/s), for comparison with the static value
    # translation check: VO acceleration vs accelerometer (gravity removed), same 1.0 s Savitzky-Golay smoothing on both
    dt = float(np.median(np.diff(tc))); ta, fa = A[:, 0] / 1e6, A[:, 2:5]
    fw = np.array([Tw[k][:3, :3] @ (Rri @ fa[(ta >= t - dt / 2) & (ta < t + dt / 2)].mean(0)) for k, t in enumerate(tc)]); gw = fw.mean(0)
    # identical band-pass on both: 1 s Savitzky-Golay smoothing minus 4 s moving trend (removes slow gravity leakage from orientation drift)
    w = int(round(1.0 / dt)) | 1; w4 = int(round(4.0 / dt)) | 1
    BP = lambda X: savgol_filter(X, w, 3, axis=0) - savgol_filter(X, w4, 1, axis=0)
    Hp = BP(Tw[:, :3, 3]); avo = np.full_like(Hp, np.nan); avo[1:-1] = (Hp[2:] - 2 * Hp[1:-1] + Hp[:-2]) / dt ** 2
    aim = BP(fw - gw)
    ok = np.isfinite(avo).all(1) & np.isfinite(aim).all(1); ok[:w4] = ok[-w4:] = False
    for k in np.nonzero(ok)[0]: AC.append(dict(clip=cid, t_s=tc[k] - tc[0], imu_x=aim[k, 0], imu_y=aim[k, 1], imu_z=aim[k, 2], vo_x=avo[k, 0], vo_y=avo[k, 1], vo_z=avo[k, 2]))
    x, y = aim[ok].ravel(), avo[ok].ravel(); M = np.c_[x - x.mean(), y - y.mean()]; vt = np.linalg.svd(M, full_matrices=False)[2]
    cs = pd.DataFrame([r for r in ST if r['clip'] == cid and r['ok']]); e = cs.err_deg.values; e0 = cs.err_raw_deg.values; rg = cs.rot_gyro_deg.values; S = cs
    d = pd.DataFrame([r for r in DR_ if r['clip'] == cid]); traj = Tw[:, :3, 3]
    SU.append(dict(clip=cid, recording=clip, task=label, seconds=round(tc[-1] - tc[0], 1), steps=len(steps), solved=len(S), inliers_median=float(S.inliers.median()), err_raw_median_deg=float(np.median(e0)), drift_raw_end_deg=float(d.drift_raw_deg.iloc[-1]), bias_fit_rad_s=[round(float(x), 5) for x in bfit], bias_fit_deg_s=float(np.degrees(np.linalg.norm(bfit))),
                   rot_step_median_deg=float(np.median(rg)), err_median_deg=float(np.median(e)), err_p95_deg=float(np.percentile(e, 95)), err_max_deg=float(e.max()),
                   rel_err_median=float(np.median(e / np.maximum(rg, 0.2))), head_rotation_total_deg=float(rg.sum()),
                   drift_30s_deg=float(d[d.t_s <= 30].drift_deg.iloc[-1]), drift_end_deg=float(d.drift_deg.iloc[-1]),
                   accel_corr=float(np.corrcoef(x, y)[0, 1]), accel_corr_xyz=[float(np.corrcoef(aim[ok][:, i], avo[ok][:, i])[0, 1]) for i in range(3)],
                   accel_scale_tls=float(-vt[-1, 0] / vt[-1, 1]), accel_rms_ratio=float(y.std() / x.std()), accel_rms_imu=float(x.std()),
                   path_m=float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()), bbox_m=[round(v, 3) for v in (traj.max(0) - traj.min(0))]))
    print(cid, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in SU[-1].items()}, flush=True)
pd.DataFrame(ST).to_csv('fn21/steps_all.csv', index=False); pd.DataFrame(DR_).round(4).to_csv('fn21/drift_curve.csv', index=False)
pd.DataFrame(AC).round(4).to_csv('fn21/accel_series.csv', index=False); su = pd.DataFrame(SU); su.to_csv('fn21/summary_by_clip.csv', index=False)
st = pd.DataFrame(ST); st = st[st.ok]; ac = pd.DataFrame(AC)
x = ac[['imu_x', 'imu_y', 'imu_z']].values.ravel(); y = ac[['vo_x', 'vo_y', 'vo_z']].values.ravel(); vt = np.linalg.svd(np.c_[x - x.mean(), y - y.mean()], full_matrices=False)[2]
tot = dict(bias_static_fn14_rad_s=[round(float(x), 5) for x in B14], bias_static_deg_s=float(np.degrees(np.linalg.norm(B14))), err_raw_median_deg=float(st.err_raw_deg.median()), drift_raw_end_range=[float(su.drift_raw_end_deg.min()), float(su.drift_raw_end_deg.max())], clips=len(SU), seconds=float(su.seconds.sum()), steps=int(su.steps.sum()), solved=int(su.solved.sum()), err_median_deg=float(st.err_deg.median()), err_p95_deg=float(st.err_deg.quantile(.95)),
           rot_step_median_deg=float(st.rot_gyro_deg.median()), head_rotation_total_deg=float(st.rot_gyro_deg.sum()), drift_30s_range=[float(su.drift_30s_deg.min()), float(su.drift_30s_deg.max())],
           drift_end_range=[float(su.drift_end_deg.min()), float(su.drift_end_deg.max())], accel_corr=float(np.corrcoef(x, y)[0, 1]), accel_scale_tls=float(-vt[-1, 0] / vt[-1, 1]),
           accel_scale_range=[float(su.accel_scale_tls.min()), float(su.accel_scale_tls.max())])
pd.Series(tot).to_csv('fn21/summary.csv', header=['value']); print(json.dumps(tot, indent=1))

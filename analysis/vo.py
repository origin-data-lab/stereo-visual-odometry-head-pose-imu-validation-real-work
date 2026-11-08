"""Stereo visual odometry on a real-work clip (ODL-EGO S1), checked against the gyroscope.

Rectify both fisheye eyes with the factory calibration (pinhole f=240 px at 800x600), track Shi-Tomasi corners
left(t)->left(t+k), triangulate them from the left/right pair at t, and solve PnP+RANSAC for the camera motion.
The wearer's own jacket (blue) is masked. Rotation is cross-checked against gyro integration over the same
exposure-centre interval (independent sensor). Output: poses, per-step rotation comparison, coloured point cloud.
usage: python vo.py CLIP_PREFIX T0_S T1_S STEP
"""
import sys, json, numpy as np, cv2, gzip
sys.path.insert(0, '/home/claude/fn/common'); sys.path.insert(0, '/home/claude/hero')
from odl_calib import device_calib
from sei import sei
pre, T0, T1, STEP = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
C = device_calib(); KL, DL, KR, DR = C['cam0']['K'], C['cam0']['D'], C['cam1']['K'], C['cam1']['D']
T = C['T_c1_c0']; Rlr, tlr = T[:3, :3], T[:3, 3]; B = float(np.linalg.norm(tlr))
Rci = C['cam0']['T_cam_imu'][:3, :3]; shift = C['cam0']['shift'] * 1e6
size = (1600, 1200); W, H = 800, 600; F = 240.0; cx, cy = 400.0, 300.0
R1, R2, _, _, _ = cv2.fisheye.stereoRectify(KL, DL, KR, DR, size, Rlr, tlr, flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=size, balance=0.0)
P1 = np.array([[F, 0, cx, 0], [0, F, cy, 0], [0, 0, 1, 0]], float); P2 = P1.copy(); P2[0, 3] = -F * B
m1 = cv2.fisheye.initUndistortRectifyMap(KL, DL, R1, P1, (W, H), cv2.CV_32FC1); m2 = cv2.fisheye.initUndistortRectifyMap(KR, DR, R2, P2, (W, H), cv2.CV_32FC1)
K = P1[:3, :3]
# rotation from IMU frame to rectified-left frame: R1 @ R_cam_imu
Rri = R1 @ Rci
SL = sei(f'{pre}_left.mp4'); tc = SL[:, 1] + shift + SL[:, 2] / 2            # exposure centre, IMU clock (us)
G = np.loadtxt(gzip.open(f'{pre}_imu.csv.gz', 'rt'), delimiter=',', skiprows=1); G = G[G[:, 1] == 2]
gt, gw = G[:, 0], G[:, 2:5]
def gyro_rot(ta, tb):
    """integrate gyro (IMU frame) from ta to tb -> rotation of the IMU, as a matrix (body frame increments)"""
    m = (gt > ta) & (gt <= tb); ts = np.r_[ta, gt[m]]; ws = gw[m]; Rm = np.eye(3)
    for k in range(len(ws)):
        dt = (ts[k + 1] - ts[k]) / 1e6; Rm = Rm @ cv2.Rodrigues(ws[k] * dt)[0]
    return Rm
capL = cv2.VideoCapture(f'{pre}_left.mp4'); capR = cv2.VideoCapture(f'{pre}_right.mp4')
f0, f1 = int(T0 * 30), int(T1 * 30)
def body_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); m = cv2.inRange(hsv, (92, 120, 40), (107, 255, 255))
    m = cv2.dilate(m, np.ones((15, 15), np.uint8)); keep = np.full(m.shape, 255, np.uint8); keep[m > 0] = 0
    keep[H - 60:] = 0; return keep
frames = {}
i = 0
while True:
    okL, a = capL.read(); okR, b = capR.read()
    if not (okL and okR) or i > f1: break
    if i >= f0 and (i - f0) % STEP == 0:
        frames[i] = (cv2.remap(cv2.resize(a, (1600, 1200)), *m1, cv2.INTER_LINEAR), cv2.remap(cv2.resize(b, (1600, 1200)), *m2, cv2.INTER_LINEAR))
    i += 1
idx = sorted(frames); print('frames', len(idx), 'B', B)
lk = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
def stereo_points(L, R):
    gl, gr = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
    p = cv2.goodFeaturesToTrack(gl, 1500, 0.01, 8, mask=body_mask(L))
    if p is None: return None, None, gl
    q, st, _ = cv2.calcOpticalFlowPyrLK(gl, gr, p, None, **lk)
    pb, stb, _ = cv2.calcOpticalFlowPyrLK(gr, gl, q, None, **lk)
    p, q = p.reshape(-1, 2), q.reshape(-1, 2); fb = np.linalg.norm(pb.reshape(-1, 2) - p, axis=1)
    d = p[:, 0] - q[:, 0]; ok = (st.ravel() == 1) & (stb.ravel() == 1) & (fb < 0.7) & (np.abs(p[:, 1] - q[:, 1]) < 1.2) & (d > 1.0)
    p, d = p[ok], d[ok]; Z = F * B / d
    X = np.c_[(p[:, 0] - cx) * Z / F, (p[:, 1] - cy) * Z / F, Z]; keep = Z < 4.0
    return p[keep].astype(np.float32), X[keep], gl
Twc = np.eye(4); poses = [(idx[0], Twc.copy())]; steps = []
for a, b in zip(idx[:-1], idx[1:]):
    p, X, ga = stereo_points(*frames[a]); gb = cv2.cvtColor(frames[b][0], cv2.COLOR_BGR2GRAY)
    if p is None or len(p) < 30: steps.append(dict(a=a, b=b, ok=False)); poses.append((b, Twc.copy())); continue
    q, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, p.reshape(-1, 1, 2), None, **lk); q = q.reshape(-1, 2); s = st.ravel() == 1
    ok, rv, tv, inl = cv2.solvePnPRansac(X[s], q[s], K, None, reprojectionError=1.5, iterationsCount=300, flags=cv2.SOLVEPNP_ITERATIVE)
    nin = 0 if inl is None else len(inl)
    if not ok or nin < 25:
        steps.append(dict(a=a, b=b, ok=False, n=int(s.sum()), inl=nin)); poses.append((b, Twc.copy())); continue
    Rab = cv2.Rodrigues(rv)[0]; Tba = np.eye(4); Tba[:3, :3] = Rab; Tba[:3, 3] = tv.ravel()   # maps points in a-frame to b-frame
    Twc = Twc @ np.linalg.inv(Tba); poses.append((b, Twc.copy()))
    # gyro over the same interval, expressed in the rectified-left frame; camera motion a->b = inv(Tba) rotation
    Rg_imu = gyro_rot(tc[a], tc[b]); Rg = Rri @ Rg_imu @ Rri.T
    Rv = Rab.T
    err = np.degrees(np.linalg.norm(cv2.Rodrigues(Rv @ Rg.T)[0]))
    steps.append(dict(a=a, b=b, ok=True, n=int(s.sum()), inl=nin, rot_vo_deg=float(np.degrees(np.linalg.norm(cv2.Rodrigues(Rv)[0]))),
                      rot_gyro_deg=float(np.degrees(np.linalg.norm(cv2.Rodrigues(Rg)[0]))), err_deg=float(err), dt_ms=float((tc[b] - tc[a]) / 1000),
                      trans_m=float(np.linalg.norm(tv))))
# whole-window rotation drift: chain of VO vs chained gyro over the same span
Rv_tot = poses[-1][1][:3, :3]; Rg_tot = Rri @ gyro_rot(tc[idx[0]], tc[idx[-1]]) @ Rri.T
drift = float(np.degrees(np.linalg.norm(cv2.Rodrigues(Rv_tot @ Rg_tot.T)[0])))
# coloured point cloud from keyframes (sparse stereo points, world frame)
cloud = []
for k, (i_, Tw) in enumerate(poses):
    if k % max(1, 30 // STEP): continue                      # one keyframe per second
    L, R = frames[i_]; gl, gr = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
    sg = cv2.StereoSGBM_create(minDisparity=0, numDisparities=96, blockSize=5, P1=200, P2=800, uniquenessRatio=18, speckleWindowSize=150, speckleRange=1, disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_HH)
    d = sg.compute(gl, gr).astype(np.float32) / 16; mk = body_mask(L) > 0
    tex = cv2.Laplacian(cv2.GaussianBlur(gl, (3, 3), 0), cv2.CV_32F); tex = cv2.boxFilter(np.abs(tex), -1, (7, 7)) > 4     # textured pixels only
    v = (d > 8) & mk & tex; ys, xs = np.nonzero(v); ys, xs = ys[::2], xs[::2]; Z = F * B / d[ys, xs]; z_ok = Z < 1.6
    ys, xs, Z = ys[z_ok], xs[z_ok], Z[z_ok]
    Pc = np.c_[(xs - cx) * Z / F, (ys - cy) * Z / F, Z, np.ones_like(Z)]; Pw = (Tw @ Pc.T).T[:, :3]
    cloud.append(np.c_[Pw, L[ys, xs][:, ::-1]])
cloud = np.vstack(cloud)
S = [s for s in steps if s['ok']]; e = np.array([s['err_deg'] for s in S]); rv_ = np.array([s['rot_vo_deg'] for s in S]); rg_ = np.array([s['rot_gyro_deg'] for s in S])
traj = np.array([Tw[:3, 3] for _, Tw in poses])
res = dict(clip=pre.split('/')[-1], t0=T0, t1=T1, step=STEP, pairs=len(steps), ok=len(S), err_med_deg=float(np.median(e)), err_p95_deg=float(np.percentile(e, 95)),
           rot_med_deg=float(np.median(rg_)), rot_rms_vo_gyro=float(np.sqrt(np.mean((rv_ - rg_) ** 2))), drift_total_deg=drift,
           total_gyro_rotation_deg=float(np.sum(rg_)), path_m=float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))), inliers_med=float(np.median([s['inl'] for s in S])),
           bbox_m=(traj.max(0) - traj.min(0)).round(3).tolist(), cloud_pts=int(len(cloud)))
tag = res['clip'] + f'_{int(T0)}_{int(T1)}'
np.save(f'traj_{tag}.npy', np.array([np.r_[i_, Tw.ravel()] for i_, Tw in poses])); np.save(f'cloud_{tag}.npy', cloud.astype(np.float32))
json.dump(dict(res=res, steps=steps), open(f'vo_{tag}.json', 'w'))
print(json.dumps(res, indent=1))

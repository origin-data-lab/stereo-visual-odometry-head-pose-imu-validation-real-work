# NOTE: device-specific readers in this file are withheld (see CODE_AVAILABILITY.md).
import json, os, numpy as np, cv2
DATA = os.environ.get('ODL_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
W, H = 1600, 1200
BOARDS = {'a4': dict(nx=9, ny=6, sq=0.030), 'far': dict(nx=5, ny=3, sq=0.055)}

def obj(ids, kind='a4', scale=1.0):
    b = BOARDS[kind]; n = b['nx'] - 1; s = b['sq'] * scale
    ids = np.array(ids)
    return np.c_[(ids % n + 1) * s, (ids // n + 1) * s, np.zeros(len(ids))]

def load(rec, cam):
    return json.load(open(os.path.join(DATA, f'det_{rec}_{cam}.json')))

def best(r, th=8, ymin=None):
    bs = [b for b in r['boards'] if len(b.get('ids', [])) >= th]
    return max(bs, key=lambda b: len(b['ids'])) if bs else None

def pairs(rec, th=12, still=3.0):
    """stereo samples with stillness measured vs neighbours (same sampling step)"""
    L = load(rec, 'left'); R = load(rec, 'right'); out = []
    def disp(b1, b2):
        if b1 is None or b2 is None: return 99
        m1 = dict(zip(b1['ids'], b1['pts'])); m2 = dict(zip(b2['ids'], b2['pts'])); c = set(m1) & set(m2)
        if len(c) < 4: return 99
        return max(np.linalg.norm(np.subtract(m1[i], m2[i])) for i in c)
    n = min(len(L), len(R))
    for k in range(1, n - 1):
        bl = best(L[k], th); br = best(R[k], th)
        if bl is None or br is None: continue
        dl = max(disp(bl, best(L[k-1], th)), disp(bl, best(L[k+1], th)))
        dr = max(disp(br, best(R[k-1], th)), disp(br, best(R[k+1], th)))
        out.append(dict(rec=rec, f=L[k]['f'], L=bl, R=br, dl=dl, dr=dr))
    return [o for o in out if max(o['dl'], o['dr']) < still] if still else out

_fe = lambda n, v: getattr(cv2.fisheye, n, v)  # OpenCV 5 removed the cv2.fisheye.CALIB_* names; values are fisheye bit flags
FL = _fe('CALIB_RECOMPUTE_EXTRINSIC', 2) | _fe('CALIB_FIX_SKEW', 8) | _fe('CALIB_USE_INTRINSIC_GUESS', 1)
CRIT = (3, 200, 1e-10)

def mono(S, side, kind='a4'):
    dc = device_calib(); c = dc['cam0' if side == 'L' else 'cam1']
    K0 = c['K'].copy(); D0 = c['D'].copy()
    O = [obj(s[side]['ids'], s.get('kind', kind)).reshape(-1, 1, 3) for s in S]
    I = [np.array(s[side]['pts']).reshape(-1, 1, 2) for s in S]
    return cv2.fisheye.calibrate(O, I, (W, H), K0, D0, flags=FL, criteria=CRIT)

def per_view_err(S, side, K, D, kind='a4'):
    errs = []
    for s in S:
        o = obj(s[side]['ids'], s.get('kind', kind)); p = np.array(s[side]['pts'])
        u = cv2.fisheye.undistortPoints(p.reshape(-1, 1, 2), K, D).reshape(-1, 2)
        ok, rv, tv = cv2.solvePnP(o, u, np.eye(3), None)
        pr, _ = cv2.fisheye.projectPoints(o.reshape(-1, 1, 3), rv, tv, K, D)
        errs.append(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - p) ** 2, 1))))
    return np.array(errs)

def stereo(S, kind='a4', KD=None):
    if KD is None:
        rl, KL, DL, _, _ = mono(S, 'L', kind); rr, KR, DR, _, _ = mono(S, 'R', kind)
    else:
        KL, DL, KR, DR = KD; rl = rr = np.nan
    O = []; IL = []; IR = []
    for s in S:
        mL = dict(zip(s['L']['ids'], s['L']['pts'])); mR = dict(zip(s['R']['ids'], s['R']['pts'])); c = sorted(set(mL) & set(mR))
        if len(c) < 6: continue
        O.append(obj(c, s.get('kind', kind)).astype(np.float32))
        IL.append(cv2.fisheye.undistortPoints(np.array([mL[i] for i in c]).reshape(-1, 1, 2), KL, DL).reshape(-1, 2).astype(np.float32))
        IR.append(cv2.fisheye.undistortPoints(np.array([mR[i] for i in c]).reshape(-1, 1, 2), KR, DR).reshape(-1, 2).astype(np.float32))
    E = np.eye(3)
    rms, _, _, _, _, R, T, *_ = cv2.stereoCalibrate(O, IL, IR, E, np.zeros(5), E, np.zeros(5), (W, H), flags=cv2.CALIB_FIX_INTRINSIC, criteria=(3, 200, 1e-12))
    return dict(rms=rms * KL[0, 0], rl=rl, rr=rr, KL=KL, DL=DL, KR=KR, DR=DR, R=R, T=T, n=len(O))

def device_calib(path=None):
    raise NotImplementedError('Origin Data Lab device-specific reader withheld; results are in assets/tables. Contact founder@origindatalab.io')

def triang(r, bl, br, kind='a4'):
    mL = dict(zip(bl['ids'], bl['pts'])); mR = dict(zip(br['ids'], br['pts'])); c = sorted(set(mL) & set(mR))
    if len(c) < 4: return None, None
    ul = cv2.fisheye.undistortPoints(np.array([mL[i] for i in c]).reshape(-1, 1, 2), r['KL'], r['DL']).reshape(-1, 2).T
    ur = cv2.fisheye.undistortPoints(np.array([mR[i] for i in c]).reshape(-1, 1, 2), r['KR'], r['DR']).reshape(-1, 2).T
    X = cv2.triangulatePoints(np.c_[np.eye(3), np.zeros(3)], np.c_[r['R'], r['T'].reshape(3, 1)], ul, ur)
    return (X[:3] / X[3]).T, c

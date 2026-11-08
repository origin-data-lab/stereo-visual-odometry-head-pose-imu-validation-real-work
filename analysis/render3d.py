"""Render the VO trajectory + coloured point cloud (gravity-aligned) as a dark image, numpy splatting."""
import sys, json, gzip, numpy as np, cv2
sys.path.insert(0, '/home/claude/fn/common'); from odl_calib import device_calib
tag, pre = sys.argv[1], sys.argv[2]; out = sys.argv[3]
C = device_calib(); KL, DL, KR, DR = C['cam0']['K'], C['cam0']['D'], C['cam1']['K'], C['cam1']['D']
T = C['T_c1_c0']; R1 = cv2.fisheye.stereoRectify(KL, DL, KR, DR, (1600, 1200), T[:3, :3], T[:3, 3], flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=(1600, 1200), balance=0.0)[0]
Rri = R1 @ C['cam0']['T_cam_imu'][:3, :3]
P = np.load(f'traj_{tag}.npy'); Tw = P[:, 1:].reshape(-1, 4, 4); cloud = np.load(f'cloud_{tag}.npy')
A = np.loadtxt(gzip.open(f'{pre}_imu.csv.gz', 'rt'), delimiter=',', skiprows=1); A = A[A[:, 1] == 4]
up_c = Rri @ A[3 * 500:4 * 500, 2:5].mean(0); up_c /= np.linalg.norm(up_c)            # specific force ~ up, first frame camera coords
# world: Y up = up_c ; X = camera x projected horizontal ; Z completes
x = np.array([1, 0, 0.]) - up_c * up_c[0]; x /= np.linalg.norm(x); z = np.cross(x, up_c)
Rw = np.stack([x, up_c, z])                                                         # rows: world axes in camera-0 coords
pts = cloud[:, :3] @ Rw.T; col = cloud[:, 3:6]
traj = Tw[:, :3, 3] @ Rw.T
# voxel downsample 8 mm
key = np.floor(pts / 0.005).astype(np.int64); _, u = np.unique(key[:, 0] * 1_000_003 ** 2 + key[:, 1] * 1_000_003 + key[:, 2], return_index=True)
pts, col = pts[u], col[u]
c0 = traj.mean(0)
# virtual camera: looking from behind/above the wearer's start, toward the workspace
def look(eye, tgt):
    f = tgt - eye; f /= np.linalg.norm(f); r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r); u = np.cross(r, f); return np.stack([r, -u, f]), eye
fwd0 = Rw @ np.array([0, 0, 1.]); fwd0[1] = 0; fwd0 /= np.linalg.norm(fwd0)
W, H = int(sys.argv[4]), int(sys.argv[5]); fpx = W * 0.62
def render(az, el, dist, tgt_off):
    tgt = c0 + fwd0 * tgt_off + np.array([0, -0.55, 0])
    rot = np.array([[np.cos(az), 0, np.sin(az)], [0, 1, 0], [-np.sin(az), 0, np.cos(az)]])
    back = rot @ (-fwd0); eye = tgt + back * dist * np.cos(el) + np.array([0, dist * np.sin(el), 0])
    Rv, e = look(eye, tgt)
    def proj(X):
        Xc = (X - e) @ Rv.T; z = Xc[:, 2]; return np.c_[fpx * Xc[:, 0] / z + W / 2, fpx * Xc[:, 1] / z + H / 2], z
    img = np.zeros((H, W, 3), np.float32); zb = np.full((H, W), np.inf, np.float32)
    uv, z = proj(pts); ok = (z > 0.05) & (uv[:, 0] >= 1) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 1) & (uv[:, 1] < H - 1)
    o = np.argsort(-z[ok]); uu = uv[ok][o].astype(int); cc = col[ok][o]; zz = z[ok][o]
    shade = np.clip(1.25 - 0.28 * zz, 0.35, 1.0)[:, None]
    for dx in (0, 1):
        for dy in (0, 1):
            img[uu[:, 1] + dy, uu[:, 0] + dx] = cc * shade
    return img, proj
img, proj = render(np.radians(float(sys.argv[6])), np.radians(float(sys.argv[7])), float(sys.argv[8]), float(sys.argv[9]))
img = np.clip(img, 0, 255).astype(np.uint8)
bg = np.zeros_like(img); bg[:] = (22, 8, 5)[::-1][::-1]                               # #050816 in RGB order handled below
mask = img.sum(2) == 0; img[mask] = (5, 8, 22)
out_img = img.copy()
# trajectory (accent) + frustums every ~3 s
uv, z = proj(traj); acc = (56, 189, 248)
pts2 = uv[z > 0].astype(np.int32)
cv2.polylines(out_img, [pts2], False, (255, 255, 255), 5, cv2.LINE_AA); cv2.polylines(out_img, [pts2], False, acc, 3, cv2.LINE_AA)
for k in range(0, len(Tw), 90):
    Rc = Rw @ Tw[k][:3, :3]; o = traj[k]; s = 0.05
    corners = [o + Rc @ np.array([sx * s * 0.75, sy * s * 0.56, s]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    q, zq = proj(np.vstack([o] + corners)); q = q.astype(np.int32)
    if (zq <= 0).any(): continue
    for c in q[1:]: cv2.line(out_img, tuple(q[0]), tuple(c), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.polylines(out_img, [q[1:]], True, (255, 255, 255), 2, cv2.LINE_AA)
cv2.circle(out_img, tuple(pts2[0]), 9, (255, 255, 255), -1, cv2.LINE_AA); cv2.circle(out_img, tuple(pts2[-1]), 9, acc, -1, cv2.LINE_AA)
cv2.imwrite(out, out_img[:, :, ::-1])
json.dump(dict(start=pts2[0].tolist(), end=pts2[-1].tolist(), n_pts=int(len(pts))), open(out + '.json', 'w')); print(out, len(pts))

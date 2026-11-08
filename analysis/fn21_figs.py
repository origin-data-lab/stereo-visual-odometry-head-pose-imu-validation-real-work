"""Figures for Field Note #21 (stereo VO head pose checked against the IMU). Run after vo_eval.py."""
import numpy as np, pandas as pd, cv2, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, os
FIG = 'fn21/figures'; os.makedirs(FIG, exist_ok=True)
INK, GR, C1, C2, C3 = '#1d232b', '#8a8f98', '#1f5fbf', '#d9622b', '#2a9d6f'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e7eaee', 'axes.edgecolor': '#c9ced6', 'axes.axisbelow': True})
def save(fig, name): fig.savefig(f'{FIG}/{name}.png', dpi=170, bbox_inches='tight'); fig.savefig(f'{FIG}/{name}.svg', bbox_inches='tight'); plt.close(fig)
ST = pd.read_csv('fn21/steps_all.csv'); DR = pd.read_csv('fn21/drift_curve.csv'); AC = pd.read_csv('fn21/accel_series.csv'); SU = pd.read_csv('fn21/summary_by_clip.csv')
COL = {'A': C1, 'B': C2, 'C': C3}; LAB = {r.clip: f'{r.clip}: {r.task}' for r in SU.itertuples()}
# 1) rotation: per-step vision vs gyro (clip A) + error distribution (all clips)
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={'width_ratios': [1.6, 1]})
a = ST[(ST['clip'] == 'A') & ST.ok]
ax = axs[0]; ax.plot(a.t_s, a.rot_gyro_deg, color=INK, lw=1.3, label='gyroscope (IMU)'); ax.plot(a.t_s, a.rot_vo_deg, color=C2, lw=1, ls='--', label='stereo vision')
ax.set_xlabel('Time in clip A (s)'); ax.set_ylabel('Rotation per step (°)'); ax.set_title('Head rotation in each 67 ms step, clip A', loc='left', fontsize=10.5); ax.legend(frameon=False, fontsize=9, loc='upper left')
ax = axs[1]; ok = ST[ST.ok]; bins = np.logspace(-3, 0.5, 50)
for c in 'ABC': ax.hist(ok[ok['clip'] == c].err_deg, bins=bins, histtype='step', lw=1.4, color=COL[c], label=c)
md, p95 = ok.err_deg.median(), ok.err_deg.quantile(.95)
ax.axvline(md, color=INK, lw=1); ax.axvline(p95, color=INK, lw=1, ls=':'); ax.set_xscale('log')
ax.text(md * 0.9, ax.get_ylim()[1] * 0.92, f'median {md:.2f}°', ha='right', fontsize=9); ax.text(p95 * 1.1, ax.get_ylim()[1] * 0.92, f'p95 {p95:.2f}°', fontsize=9)
ax.set_xlabel('Vision − gyro difference per step (°)'); ax.set_ylabel('Steps'); ax.set_title(f'All {len(ok):,} solved steps, 3 clips', loc='left', fontsize=10.5); ax.legend(frameon=False, fontsize=9, title='clip', title_fontsize=9, loc='center left')
save(fig, 'fn21_rotation_check')
# 2) translation: acceleration from VO positions vs accelerometer
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={'width_ratios': [1.6, 1]})
a = AC[AC['clip'] == 'A']; k = max('xyz', key=lambda c: a[f'imu_{c}'].std())
ax = axs[0]; ax.plot(a.t_s, a[f'imu_{k}'], color=INK, lw=1.3, label='accelerometer, gravity removed'); ax.plot(a.t_s, a[f'vo_{k}'], color=C2, lw=1, ls='--', label='2nd derivative of stereo position')
ax.set_xlabel('Time in clip A (s)'); ax.set_ylabel(f'Acceleration, {k} axis (m/s²)'); ax.set_title('Camera acceleration, same band-pass on both', loc='left', fontsize=10.5); ax.legend(frameon=False, fontsize=9, loc='upper left', ncol=2)
ax = axs[1]
for c in 'ABC':
    b = AC[AC['clip'] == c]; ax.scatter(b[['imu_x', 'imu_y', 'imu_z']].values.ravel(), b[['vo_x', 'vo_y', 'vo_z']].values.ravel(), s=3, alpha=.35, color=COL[c], label=c, lw=0)
lim = np.nanpercentile(np.abs(AC[['imu_x', 'imu_y', 'imu_z']].values), 99.5); ax.plot([-lim, lim], [-lim, lim], color=INK, lw=1); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
ax.set_xlabel('Accelerometer (m/s²)'); ax.set_ylabel('Stereo vision (m/s²)'); ax.set_title('All clips, all axes; line = equal', loc='left', fontsize=10.5); ax.legend(frameon=False, fontsize=9, markerscale=3, title='clip', title_fontsize=9)
save(fig, 'fn21_accel_check')
# 3) rotation drift over the clip: raw gyro vs gyro with the static bias (Field Note #14) removed
fig, ax = plt.subplots(figsize=(8, 3.6))
for c in 'ABC':
    d = DR[DR['clip'] == c]; ax.plot(d.t_s, d.drift_raw_deg, color=COL[c], lw=1, alpha=.35, ls='--'); ax.plot(d.t_s, d.drift_deg, color=COL[c], lw=1.6, label=LAB[c])
ax.plot([], [], color=GR, ls='--', lw=1, label='same, gyro bias not removed')
ax.annotate('A: 4 failed steps in a fast turn\nand back (pose held)', xy=(32.5, 23), xytext=(38, 26), fontsize=8.5, color=INK, arrowprops=dict(arrowstyle='->', color=GR, lw=.8))
ax.set_xlabel('Time from clip start (s)'); ax.set_ylabel('Vision vs gyro orientation (°)'); ax.set_title('Accumulated orientation difference (no loop closure)', loc='left', fontsize=10.5); ax.legend(frameon=False, fontsize=8.5, loc='upper left')
save(fig, 'fn21_drift')
# 4) trajectory render (clip A, first 36 s; rendered with render3d.py)
im = cv2.imread('/home/claude/vo/r_main.jpg'); cv2.imwrite(f'{FIG}/fn21_trajectory.png', im)
print('figures ok', k)

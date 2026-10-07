"""
Thu vien OMA (Operational Modal Analysis) bang ma nguon mo - thay the MACEC 3.4.

Gom:
  - tien xu ly (detrend, decimate, bandpass)
  - SSI-COV (Covariance-driven Stochastic Subspace Identification)
  - bieu do on dinh (stabilization diagram) + gom cum cuc (pole clustering)
  - FDD/SVD cua ma tran PSD de doi chung bang phuong phap khac
  - chi so chat luong mode shape (MPC, MPD) va MAC

Chi phu thuoc numpy + scipy.
"""

import os
import numpy as np
from scipy import signal, linalg


# ----------------------------------------------------------------------------
# 1. TIEN XU LY
# ----------------------------------------------------------------------------

def preprocess(x, fs, dec_factors=(4, 8), f_high=0.3, f_low=None, order=4):
    """Detrend -> decimate (nhieu buoc, co loc chong aliasing) -> loc thong cao.

    x           : (N, nch) tin hieu gia toc
    dec_factors : cac he so decimate lien tiep, vd (4, 8) => giam 32 lan
    f_high      : tan so cat thong cao [Hz], loai troi va thanh phan DC
    f_low       : tan so cat thong thap [Hz]; None = de bo loc chong aliasing lo
    """
    y = signal.detrend(np.asarray(x, dtype=float), axis=0, type='linear')

    fs_out = float(fs)
    for q in dec_factors:
        # Chebyshev type I zero-phase; decimate da bao gom loc chong aliasing
        y = signal.decimate(y, q, ftype='iir', zero_phase=True, axis=0)
        fs_out /= q

    nyq = fs_out / 2.0
    if f_high:
        b, a = signal.butter(order, f_high / nyq, btype='high')
        y = signal.filtfilt(b, a, y, axis=0)
    if f_low:
        b, a = signal.butter(order, min(f_low / nyq, 0.99), btype='low')
        y = signal.filtfilt(b, a, y, axis=0)

    y = signal.detrend(y, axis=0, type='constant')
    return y, fs_out


# ----------------------------------------------------------------------------
# 2. SSI-COV
# ----------------------------------------------------------------------------

def output_covariances(y, n_lags):
    """R[k] = E[y(t+k) y(t)^T], k = 0..n_lags. y: (N, nch) -> (n_lags+1, nch, nch)"""
    N, nch = y.shape
    R = np.empty((n_lags + 1, nch, nch))
    for k in range(n_lags + 1):
        R[k] = y[k:].T @ y[:N - k] / (N - k)
    return R


def block_toeplitz(R, i, refs):
    """Ma tran Toeplitz khoi T_{1|i} dung cho SSI-COV.

    T[p, q] = R[i + p - q],  p, q = 0..i-1  (cot lay theo kenh tham chieu refs)
    """
    nch = R.shape[1]
    nref = len(refs)
    T = np.empty((i * nch, i * nref))
    for p in range(i):
        for q in range(i):
            T[p * nch:(p + 1) * nch, q * nref:(q + 1) * nref] = R[i + p - q][:, refs]
    return T


def ssi_cov(y, dt, i_blockrows, orders, refs=None):
    """SSI-COV: tra ve tham so modal cho tung bac mo hinh.

    y           : (N, nch)
    dt          : buoc thoi gian [s]
    i_blockrows : so hang khoi i (cua so tre = i*dt giay)
    orders      : danh sach bac mo hinh (so chan)
    refs        : chi so kenh tham chieu (mac dinh: tat ca)

    Tra ve dict {order: dict(f=, xi=, phi=, lam=)}
    """
    N, nch = y.shape
    if refs is None:
        refs = np.arange(nch)
    refs = np.asarray(refs)

    R = output_covariances(y, 2 * i_blockrows)
    T = block_toeplitz(R, i_blockrows, refs)

    U, S, Vt = linalg.svd(T, full_matrices=False)

    results = {}
    for n in orders:
        if n > len(S):
            continue
        U1 = U[:, :n]
        sqS = np.sqrt(S[:n])
        Oi = U1 * sqS                      # ma tran quan sat mo rong
        # bat bien dich chuyen: Oi(1:end-nch) * A = Oi(nch+1:end)
        A, *_ = linalg.lstsq(Oi[:-nch, :], Oi[nch:, :])
        C = Oi[:nch, :]

        mu, V = linalg.eig(A)
        lam = np.log(mu.astype(complex)) / dt      # cuc lien tuc
        phi = C @ V                                 # dang mode tai cac kenh

        # chi giu cuc co tan so duong (mot nua cap lien hop phuc)
        keep = lam.imag > 0
        lam, phi = lam[keep], phi[:, keep]

        f = np.abs(lam) / (2 * np.pi)
        xi = -lam.real / np.abs(lam)

        idx = np.argsort(f)
        results[n] = dict(f=f[idx], xi=xi[idx], phi=phi[:, idx], lam=lam[idx])
    return results


# ----------------------------------------------------------------------------
# 3. LOC CUC + BIEU DO ON DINH + GOM CUM
# ----------------------------------------------------------------------------

def mac(p1, p2):
    """Modal Assurance Criterion giua hai dang mode phuc."""
    num = np.abs(np.vdot(p1, p2)) ** 2
    den = np.real(np.vdot(p1, p1) * np.vdot(p2, p2))
    return float(num / den) if den > 0 else 0.0


def mpc(phi):
    """Modal Phase Collinearity (0..1). Gan 1 = mode thuc, dang mode nhat quan."""
    re, im = phi.real, phi.imag
    re = re - re.mean()
    im = im - im.mean()
    sxx, syy, sxy = re @ re, im @ im, re @ im
    eta = (syy - sxx) / (2 * sxy) if sxy != 0 else 0.0
    lam1 = (sxx + syy) / 2 + sxy * np.sqrt(eta ** 2 + 1) * np.sign(eta if eta else 1)
    lam2 = (sxx + syy) - lam1
    s = lam1 + lam2
    return float(((lam1 - lam2) / s) ** 2) if s > 0 else 0.0


def mpd(phi):
    """Mean Phase Deviation [do]. Cang nho cang tot."""
    ph = np.angle(phi)
    ph = np.unwrap(np.sort(ph))
    w = np.abs(phi)
    m = np.average(ph, weights=w)
    return float(np.degrees(np.sqrt(np.average((ph - m) ** 2, weights=w))))


def filter_poles(res, f_range, xi_range=(0.001, 0.20), mpc_min=0.0):
    """Loai cuc phi vat ly: ngoai dai tan, cản am hoac qua lon."""
    out = {}
    for n, r in res.items():
        m = ((r['f'] >= f_range[0]) & (r['f'] <= f_range[1]) &
             (r['xi'] > xi_range[0]) & (r['xi'] < xi_range[1]))
        if mpc_min > 0:
            m &= np.array([mpc(r['phi'][:, k]) >= mpc_min for k in range(len(r['f']))])
        out[n] = dict(f=r['f'][m], xi=r['xi'][m], phi=r['phi'][:, m])
    return out


def stable_poles(res, tol_f=0.01, tol_xi=0.05, mac_min=0.98):
    """Danh dau cuc on dinh: so khop voi bac lien truoc theo f, xi va MAC."""
    orders = sorted(res)
    stab = {n: np.zeros(len(res[n]['f']), dtype=bool) for n in orders}
    for a, b in zip(orders[:-1], orders[1:]):
        ra, rb = res[a], res[b]
        if len(ra['f']) == 0:
            continue
        for k in range(len(rb['f'])):
            fk, xk, pk = rb['f'][k], rb['xi'][k], rb['phi'][:, k]
            j = int(np.argmin(np.abs(ra['f'] - fk)))
            df = abs(ra['f'][j] - fk) / fk
            dxi = abs(ra['xi'][j] - xk) / max(xk, 1e-9)
            if df < tol_f and dxi < tol_xi and mac(ra['phi'][:, j], pk) > mac_min:
                stab[b][k] = True
    return stab


def cluster_modes(res, stab, tol_f=0.02, mac_min=0.90, min_count=8):
    """Gom cac cuc on dinh thanh mode vat ly. Tra ve list dict da sap theo f."""
    pts = []
    for n in sorted(res):
        r = res[n]
        for k in np.where(stab[n])[0]:
            pts.append(dict(order=n, f=r['f'][k], xi=r['xi'][k], phi=r['phi'][:, k]))
    pts.sort(key=lambda p: p['f'])

    clusters = []
    for p in pts:
        placed = False
        for c in clusters:
            if abs(p['f'] - c['f_ref']) / c['f_ref'] < tol_f and mac(p['phi'], c['phi_ref']) > mac_min:
                c['members'].append(p)
                c['f_ref'] = float(np.mean([m['f'] for m in c['members']]))
                placed = True
                break
        if not placed:
            clusters.append(dict(f_ref=p['f'], phi_ref=p['phi'], members=[p]))

    modes = []
    for c in clusters:
        if len(c['members']) < min_count:
            continue
        f = np.array([m['f'] for m in c['members']])
        xi = np.array([m['xi'] for m in c['members']])
        # dang mode dai dien: lay tai bac cao nhat trong cum
        rep = max(c['members'], key=lambda m: m['order'])
        phi = rep['phi'] / rep['phi'][np.argmax(np.abs(rep['phi']))]
        modes.append(dict(
            f=float(f.mean()), f_std=float(f.std()),
            xi=float(np.median(xi)), xi_std=float(xi.std()),
            count=len(c['members']), order=rep['order'],
            phi=phi, mpc=mpc(phi), mpd=mpd(phi),
        ))
    modes.sort(key=lambda m: m['f'])
    return modes


# ----------------------------------------------------------------------------
# 4. FDD - doi chung bang phuong phap mien tan so
# ----------------------------------------------------------------------------

def fdd(y, fs, nperseg=None):
    """Frequency Domain Decomposition: SVD cua ma tran PSD tai tung tan so.

    Tra ve (f, sv) voi sv: (n_freq, nch) cac gia tri ky di giam dan.
    """
    N, nch = y.shape
    if nperseg is None:
        nperseg = min(2 ** int(np.log2(N / 8)), 4096)

    f, Pxy = signal.csd(y[:, :, None], y[:, None, :], fs=fs,
                        nperseg=nperseg, axis=0)
    sv = np.empty((len(f), nch))
    for k in range(len(f)):
        sv[k] = linalg.svdvals(Pxy[k])
    return f, sv


def peak_pick(f, sv0, f_range, prominence_db=3.0, n_max=15):
    """Chon dinh tren duong gia tri ky di thu nhat (dB)."""
    m = (f >= f_range[0]) & (f <= f_range[1])
    ff, pp = f[m], 10 * np.log10(sv0[m] + 1e-30)
    idx, props = signal.find_peaks(pp, prominence=prominence_db)
    if len(idx) == 0:
        return np.array([]), np.array([])
    order = np.argsort(props['prominences'])[::-1][:n_max]
    sel = np.sort(idx[order])
    return ff[sel], props['prominences'][np.argsort(props['prominences'])[::-1][:n_max]]


# ----------------------------------------------------------------------------
# 5. KIEM TRA THANH PHAN DIEU HOA (HARMONIC)
# ----------------------------------------------------------------------------

def harmonic_indicator(y, fs, f0, bw=0.4):
    """Chi so phan biet mode ket cau / thanh phan dieu hoa.

    Loc dai hep quanh f0 roi tinh kurtosis cua tin hieu.
    - Harmonic thuan (sin) -> kurtosis ~ 1.5
    - Mode ket cau kich thich ngau nhien -> kurtosis ~ 3
    """
    nyq = fs / 2
    lo, hi = max(f0 - bw, 0.05) / nyq, min((f0 + bw) / nyq, 0.99)
    if lo >= hi:
        return np.nan
    b, a = signal.butter(4, [lo, hi], btype='band')
    z = signal.filtfilt(b, a, y, axis=0)
    k = [np.mean(c ** 4) / np.mean(c ** 2) ** 2 for c in z.T]
    return float(np.mean(k))


# ----------------------------------------------------------------------------
# Cac cuc dai thap theo setup, doc tu ket qua run_dot1.py / run_dot2.py
# ----------------------------------------------------------------------------
def band_poles_by_setup(results_dir, band=(1.0, 1.4)):
    """Tra ve (C1, C2): {ten setup ngan: [cac tan so trong dai]} cho hai dot do."""
    import csv
    out = []
    for camp in (1, 2):
        path = os.path.join(results_dir, f"dot{camp}", f"modes_dot{camp}.csv")
        d = {}
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                name = r["setup"].split(", campaign")[0].replace(" setup", "")
                d.setdefault(name, [])
                f = float(r["f_Hz"])
                if band[0] <= f <= band[1]:
                    d[name].append(f)
        out.append(d)
    return out[0], out[1]

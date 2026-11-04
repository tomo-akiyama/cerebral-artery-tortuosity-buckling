#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: sin trend fitting."""

import numpy as np
from scipy.fft import dct
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.ndimage import binary_closing
from scipy.spatial import cKDTree

                                                         


# ===== Utilities for sin+trend fitting =====
def arclength(P_3darr):
    ds = np.linalg.norm(np.diff(P_3darr, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return s, ds


def trime_ends(x, removal):
    return x[removal: -removal]


def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)                   

                     
def bishop_frame(T_3darr):
    T = np.asarray(T_3darr, float)
    N = len(T)
    U = np.zeros_like(T)
    seed = np.array([1.0, 0.0, 0.0])
                                                
    if np.abs(np.dot(seed, T[0])) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    u0 = seed - np.dot(seed, T[0]) * T[0]
    U[0] = u0 / (np.linalg.norm(u0) + 1e-12)
    for i in range(1, N):
        u = U[i - 1] - np.dot(U[i - 1], T[i]) * T[i]
        U[i] = u / (np.linalg.norm(u) + 1e-12)
    V = np.cross(T, U)
    V = normalize(V)
                                                 
    for i in range(1, N):
        if np.dot(U[i], U[i - 1]) < 0:
            U[i] = -U[i]
            V[i] = -V[i]
    return T, U, V


def lateral_components(P_3darr, P0_3darr, T_3darr, U_3darr, V_3darr):
                   
    Delta_3darr = P_3darr - P0_3darr
                 
    Delta_parT_3darr = np.einsum('ij,ij->i', Delta_3darr, T_3darr)[:, np.newaxis] * T_3darr
                             
    Delta_perpT_3darr = Delta_3darr - Delta_parT_3darr

            
    uU_arr = np.einsum('ij,ij->i', Delta_perpT_3darr, U_3darr)
    uV_arr = np.einsum('ij,ij->i', Delta_perpT_3darr, V_3darr)

               
    DeltaU_3darr = uU_arr[:, None] * U_3darr
    DeltaV_3darr = uV_arr[:, None] * V_3darr
    return uU_arr, uV_arr, Delta_3darr, Delta_parT_3darr, Delta_perpT_3darr, DeltaU_3darr, DeltaV_3darr


def kmeans_1d_two_clusters(x, n_iter=50):
                                 
    x = x.reshape(-1, 1)
    c1, c2 = np.percentile(x, [33, 66]).reshape(2, 1)
    for _ in range(n_iter):
        d1 = np.abs(x - c1)
        d2 = np.abs(x - c2)
        labels = (d2 < d1).astype(int)
        if np.any(labels == 0):
            c1_new = x[labels == 0].mean()
        else:
            c1_new = c1
        if np.any(labels == 1):
            c2_new = x[labels == 1].mean()
        else:
            c2_new = c2
        if np.isclose(c1, c1_new).all() and np.isclose(c2, c2_new).all():
            break
        c1, c2 = np.array([[c1_new]]), np.array([[c2_new]])
    centers = np.array([c1.item(), c2.item()])
    labels = labels.ravel()
    return labels, centers

"""
# ===== 信号処理・FFT関連関数 =====
def dct_power_sum(uU_arr, uV_arr, detrend_deg=2):
    # DCT-II (orthonormal) を uU,uV に別々に適用し、パワー和 S[m] を返す。
    uU_arr = np.asarray(uU_arr, float)
    uV_arr = np.asarray(uV_arr, float)
    N = len(uU_arr)

    # デトレンド
    uU_detr, metaU = poly_trend(uU_arr, np.arange(N), degree=detrend_deg)
    uV_detr, metaV = poly_trend(uV_arr, np.arange(N), degree=detrend_deg)

    # DCT-II
    Uc_arr = dct(uU_detr, type=2, norm='ortho')
    Vc_arr = dct(uV_detr, type=2, norm='ortho')

    S_arr = Uc_arr**2 + Vc_arr**2
    m_arr = np.arange(N)
    return m_arr, S_arr, Uc_arr, Vc_arr, N, dict(metaU=metaU, metaV=metaV)
"""

"""
def dct_index_to_k(m_arr, N, ds):
    # DCT-II のインデックスmを波数k[rad/length] に換算（目安）
    return np.pi * m_arr / (N * ds)
"""

"""
def pick_peak_in_band(m_arr, S_arr, k_arr, lambda_prior=None, mult_range=(0.5, 2.0), exclude_dc=True):
    # バンド制限（k_band=(kmin,kmax)）内で S のピークを取る。
    if lambda_prior is None:
        sel = (m_arr >= 1) if exclude_dc else np.ones_like(m_arr, dtype=bool)
    else:
        lam_min = mult_range[0] * lambda_prior
        lam_max = mult_range[1] * lambda_prior
        k_min = 2 * np.pi / lam_max
        k_max = 2 * np.pi / lam_min
        sel = (k_arr >= k_min) & (k_arr <= k_max)
        if exclude_dc:
            sel = sel & (m_arr >= 1)

    if not np.any(sel):
        return None, dict(found=False, reason="no_selection")

    rel_idx = int(np.argmax(S_arr[sel]))
    idx = np.where(sel)[0][rel_idx]
    m_star = int(m_arr[idx])
    k_star = float(k_arr[idx])
    lam_star = float(2*np.pi / max(k_star, 1e-12))
    return m_star, dict(found=True, m=m_star, k=k_star, lambda_est=lam_star, idx=idx)
"""
"""
def find_candidate_peaks_in_band(
    mode_arr, power_arr, k_arr,
    lambda_prior,
    mult_range=(0.5, 2.0),
    top_n=3,
    peak_power_frac=None,
    min_band_frac=0.10,
    min_separation=2,
    exclude_dc=True,
    include_edge=True,
    force_take_argmax=True):
    
    # DCTパワー和 S = |U|^2 + |V|^2から、事前帯域内の局所ピークを"最大top_n個"抽出
    mode_arr = np.asarray(mode_arr)
    power_arr = np.asarray(power_arr, float)
    k_arr = np.asarray(k_arr, float)

    # 事前帯域
    lam_min = mult_range[0] * lambda_prior
    lam_max = mult_range[1] * lambda_prior
    k_min = 2 * np.pi / lam_max
    k_max = 2 * np.pi / lam_min
    sel = (k_arr >= k_min) & (k_arr <= k_max)
    if exclude_dc:
        sel = sel & (mode_arr >= 1)

    if not np.any(sel):
        return [], dict(reason="no_band", band=None)

    band_idx = np.where(sel)[0]
    band_power = power_arr[band_idx]

    max_power_band = float(np.max(band_power))
    total_power_band = float(np.sum(band_power))

    threshold_list = []
    if (peak_power_frac is not None) and (max_power_band > 0):
        threshold_local = max_power_band * peak_power_frac
        threshold_list.append(threshold_local)
    else:
        threshold_local = None

    if (min_band_frac is not None) and (total_power_band > 0):
        threshold_global = total_power_band * min_band_frac
        threshold_list.append(threshold_global)
    else:
        threshold_global = None

    threshold_power = max(threshold_list) if len(threshold_list) > 0 else 0.0

    idx_peaks_local, props = find_peaks(
        band_power, height=threshold_power, distance=min_separation
    )
    idx_peaks_abs = band_idx[idx_peaks_local]

    extra = []
    if include_edge and band_power.size >= 2:
        if (band_power[0] >= threshold_power) and (band_power[0] >= band_power[1]):
            extra.append(band_idx[0])
        if (band_power[-1] >= threshold_power) and (band_power[-1] >= band_power[-2]):
            extra.append(band_idx[-1])

    if force_take_argmax:
        extra.append(band_idx[int(np.argmax(band_power))])

    if len(extra):
        idx_peaks_abs = np.unique(np.concatenate([idx_peaks_abs, extra]).astype(int))

    if idx_peaks_abs.size == 0:
        order = np.argsort(band_power)[::-1]
        idx_candidates = band_idx[order[:top_n]]
    else:
        idx_candidates = idx_peaks_abs

    if idx_candidates.size > 1 and min_separation > 0:
        order = np.argsort(power_arr[idx_candidates])[::-1]
        chosen = []
        for j in idx_candidates[order]:
            if all(abs(mode_arr[j] - mode_arr[q]) >= min_separation for q in chosen):
                chosen.append(j)
        idx_candidates = np.array(sorted(chosen), dtype=int)

    idx_candidates = idx_candidates[np.argsort(power_arr[idx_candidates])[::-1]]
    if idx_candidates.size > top_n:
        idx_candidates = idx_candidates[:top_n]

    candidates = []
    for j in idx_candidates:
        lamb = (2 * np.pi / k_arr[j]) if k_arr[j] > 0 else np.inf
        candidates.append(dict(
            idx=int(j),
            mode=int(mode_arr[j]),
            k=float(k_arr[j]),
            lamb=float(lamb),
            power=float(power_arr[j]),
            power_frac_band=float(power_arr[j] / total_power_band),
            in_band=True
        ))

    meta = dict(
        k_band=(k_min, k_max),
        lambda_band=(lam_min, lam_max),
        max_power_band=max_power_band,
        total_power_band=total_power_band,
        threshold_local=threshold_local,
        threshold_global=threshold_global,
        threshold_used=threshold_power,
        n_in_band=len(band_power),
        n_candidates=len(candidates),
        min_separation=min_separation,
        top_n=top_n,
    )

    return candidates, meta
"""

"""
def k_to_dct_index(k, N, ds):
    # k [rad/voxel] から DCT モード番号 m に変換
    m = k * N * ds / np.pi
    return int(np.floor(m + 0.5))
"""

"""
def harmonic_ratio_dct(m, S):
    # DCT-IIの近似ビンでE3/E1を評価（CoreのQC用）
    m1 = int(m)
    m3 = 3 * m1

    if m1 <= 0 or m3 >= len(S):
        return float('nan'), dict(m1=m1, m3=m3, reason="out_of_range")
    else:
        E1 = float(S[m1])
        E3 = float(S[m3])
        return float(E3/(E1 + 1e-12)), dict(m1=m1, m3=m3, reason="ok")
"""


def harmonic_ratio_from_fit(uU, uV, s, k, ds, weights=None, trend_degree=1):
                                                              
    uU = np.asarray(uU, float)
    uV = np.asarray(uV, float)
    s = np.asarray(s, float)

                                
    fit = fit_sin_trend_two_component(uU, uV, s, k,
                                      weights=weights,
                                      trend_degree=trend_degree)

    A1 = float(fit['A_comb'])

    if 3.0 * k >= np.pi / ds:
        return dict(hr_energy=np.nan, hr_amp=np.nan, A1=A1, A3=np.nan, fit=fit)

                                 
    rU = uU - fit['yhat_U']
    rV = uV - fit['yhat_V']
    X3 = design_matrix(s, 3*k, trend_degree=0)

    (a3U, b3U, cU3), y3U = solve_linear(X3, rU, weights=weights)
    (a3V, b3V, cV3), y3V = solve_linear(X3, rV, weights=weights)

    A3 = float(np.sqrt(a3U**2 + b3U**2 + a3V**2 + b3V**2))

    hr_energy = (A3*A3) / (A1*A1 + 1e-12)
    hr_amp = A3 / (A1 + 1e-12)

    return dict(hr_energy=float(hr_energy), hr_amp=float(hr_amp), A1=A1, A3=A3, fit=fit)

"""
def dk_bin_from_s(N, ds):
    # 1 binの波数幅 Δk_bin [rad/voxel] を計算
    dk_bin = np.pi / (N * ds)
    return dk_bin
"""

"""
def grid_around(k0, N, ds, B_bins, q_rel, step_frac):
    # k_seedsの周りを"bin幅"と"％幅"の大きい方で半幅を決めて等間隔グリッドを返す
    k0 = float(k0)
    dk_bin = dk_bin_from_s(N, ds)

    half_bin = float(B_bins) * dk_bin
    half_pct = float(q_rel) * k0
    half = max(half_bin, half_pct)

    dk = max(step_frac * dk_bin, 1e-12)
    lo, hi = max(k0 - half, 1e-12), k0 + half
    n = int(np.ceil((hi - lo)/dk)) + 1
    ks = lo + np.arange(n) * dk
    return ks
"""


                         
def poly_trend(P_3darr, s_arr, degree, return_residual=True):
    Y = np.asarray(P_3darr, float)
    s = np.asarray(s_arr, float)

    is_1d = (Y.ndim == 1)
    if is_1d:
        Y = Y[:, None]

    if degree == 0:
        mean_val = np.mean(Y, axis=0)
        trend = np.tile(mean_val, (len(Y), 1))

        info = {
            "coef": mean_val,
            "degree": 0,
            "s_minmax": (float(s[0]), float(s[-1]))
        }

        if return_residual:
            resid = Y - trend
            info["residual"] = resid[:, 0] if is_1d else resid

        trend_out = trend[:, 0] if is_1d else trend
        return trend_out, info

    s0 = (s - s[0]) / max(s[-1] - s[0], 1e-12)
    coef = np.polyfit(s0, Y, deg=degree)

    if is_1d:
        trend_1d = np.polyval(coef, s0)
        trend = trend_1d[:, None]
    else:
        trend = np.polyval(coef, s0[:, None])

    info = {
        "coef": coef,
        "degree": degree,
        "s_minmax": (float(s[0]), float(s[-1]))
    }

    if return_residual:
        resid = Y - trend
        info["residual"] = resid[:, 0] if is_1d else resid

    trend_out = trend[:, 0] if is_1d else trend
    return trend_out, info


def design_matrix(s_arr, k_arr, trend_degree=1):
    s = np.asarray(s_arr, float)
    s0 = s - s.mean()
    X = [np.cos(k_arr*s), np.sin(k_arr*s), np.ones_like(s)]
    if trend_degree >= 1:
        X.append(s0)
    if trend_degree >= 2:
        X.append(s0**2)
    return np.column_stack(X)


def design_matrix_trend_only(s_arr, trend_degree=1):
    s = np.asarray(s_arr, float)
    s0 = s - s.mean()
    X = [np.ones_like(s)]
    if trend_degree >= 1:
        X.append(s0)
    if trend_degree >= 2:
        X.append(s0**2)
    return np.column_stack(X)


def solve_linear(X, y, weights=None, return_info=False):
    if weights is not None:
        weights = np.asarray(weights, float)
        W = np.sqrt(np.clip(weights, 0.0, np.inf))[:, None]
        Xw = X * W
        yw = y * W.squeeze()
        coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    else:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef

    resid = y - yhat
    sse = float(np.sum((resid if weights is None else np.sqrt(weights)*resid)**2))
    if return_info:
        return coef, yhat, dict(sse=sse, n=X.shape[0], p=X.shape[1])
    
    return coef, yhat


def r2_unweighted(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1.0 - ss_res / ss_tot)


def r2_weighted(y_true, y_pred, weights):
    weights = np.asarray(weights, float)
    wsum = np.sum(weights)
    ybar_w = np.sum(weights * y_true) / wsum
    ss_res_w = np.sum(weights * (y_true - y_pred)**2)
    ss_tot_w = np.sum(weights * (y_true - ybar_w)**2)
    return float(1.0 - ss_res_w / ss_tot_w)


def fit_sin_trend_two_component(uU_arr, uV_arr, s_arr, k_arr,
                                weights=None,
                                trend_degree=1):
    uU = np.asarray(uU_arr, float)
    uV = np.asarray(uV_arr, float)
    s = np.asarray(s_arr, float)

    X = design_matrix(s, k_arr, trend_degree=trend_degree)

    cU, yU = solve_linear(X, uU, weights=weights)
    cV, yV = solve_linear(X, uV, weights=weights)

    def amp_from_coef(c):
        a, b = float(c[0]), float(c[1])
        return a, b, float(np.sqrt(a**2 + b**2))

    aU, bU, AU = amp_from_coef(cU)
    aV, bV, AV = amp_from_coef(cV)
    A = float(np.sqrt(aU**2 + bU**2 + aV**2 + bV**2))

    y_true_cat = np.concatenate([uU, uV])
    y_pred_cat = np.concatenate([yU, yV])
    R2_tot = r2_unweighted(y_true_cat, y_pred_cat)

    resid_full = y_true_cat - y_pred_cat
    if weights is None:
        SSE_full = float(np.sum(resid_full**2))
    else:
        w = np.asarray(weights, float)
        w_cat = np.concatenate([w, w])
        SSE_full = float(np.sum(w_cat * resid_full**2))

    X_tr = design_matrix_trend_only(s, trend_degree=trend_degree)
    _, yU_tr = solve_linear(X_tr, uU, weights=weights)
    _, yV_tr = solve_linear(X_tr, uV, weights=weights)
    y_pred_tr_cat = np.concatenate([yU_tr, yV_tr])
    resid_tr = y_true_cat - y_pred_tr_cat
    if weights is None:
        SSE_tr = float(np.sum(resid_tr**2))
    else:
        w_cat = np.concatenate([w, w])
        SSE_tr = float(np.sum(w_cat * resid_tr**2))

    partial_R2 = 1.0 - SSE_full / SSE_tr

    R2_w = None
    if weights is not None:
        w_cat = np.concatenate([w, w])
        R2_w = r2_weighted(y_true_cat, y_pred_cat, w_cat)

    return dict(
        k=float(k_arr),
        lamb=float(2.0*np.pi / max(k_arr, 1e-12)),
        A_comb=float(A), A_U=AU, A_V=AV,
        uU_cos=aU, uU_sin=bU,
        uV_cos=aV, uV_sin=bV,
        coef_U=cU, coef_V=cV,
        yhat_U=yU, yhat_V=yV,
        R2=R2_tot,
        R2_weighted=R2_w,
        partial_R2=partial_R2,
        resid_full=resid_full,
        resid_tr=resid_tr,
        SSE_full=SSE_full,
        SSE_tr=SSE_tr
    )


                                             
                                                                                                                                                                 
                                                                                      
                                                                
"""
def choose_k_from_tie_set(candidates, 
                          lambda_prior=None,
                          dR2_tol=0.006, 
                          dhr_amp_tol=0.05):
    # 統計学的に同等のkをもつTie setから1つに絞る
    R2_vals = np.array([c['R2'] for c in candidates], float)
    R2_max = R2_vals.max()
    tie = [c for c in candidates if (R2_max - c['R2']) <= dR2_tol]
    print("複数のkが統計学的に同等（ΔR² ≤ {:.4f}）: {} 個".format(dR2_tol, len(tie)))
    print("k values in tie set:", [c['k'] for c in tie])

    if len(tie) == 1:
        return tie[0], tie
    
    hr_amp_vals = np.array([c['hr_amp'] for c in tie])
    hr_amp_min = float(hr_amp_vals.min())
    tie2 = [c for c in tie if (c['hr_amp'] - hr_amp_min) <= dhr_amp_tol]
    print("さらにtie set内での複数のkでhr_ampに大きな差がない（Δhr_amp ≤ {:.4f}）: {} 個".format(dhr_amp_tol, len(tie2)))
    print("k values in tie set2:", [c['k'] for c in tie2])

    if len(tie2) == 1:
        return tie2[0], tie2

    if lambda_prior is not None:
        k_prior = 2 * np.pi / lambda_prior
        tie3 = sorted(tie2, key=lambda c: abs(c['k'] - k_prior))
        print("k_priorに近い順:", [c['k'] for c in tie3])
        return tie3[0], tie3
    else:
        tie3 = sorted(tie2, key=lambda c: c['k'])
        print("k小さい順:", [c['k'] for c in tie3])
        return tie3[0], tie3
"""

"""
def grid_search_k(uU, uV, s, N, ds,
                  k_seeds,
                  pct_coarse=0.30, pct_fine=0.10,
                  B_bins_coarse=1.5, B_bins_fine=0.5,
                  step_frac=0.25,
                  weights=None,
                  trend_degree=1,
                  lambda_prior=None
                  ):
    # 最適な波数 k を「粗→細」で探す
    k_seeds = np.atleast_1d(np.asarray(k_seeds, float))
    s = np.asarray(s, float)

    # 1) 粗探索（k_seedsの ±30% or ±1.5bin）
    # k_seedsの各seed k_0の周りを分割(_grid_around) -> 複数seedがあれば結合して重複を除去。
    # ※近い複数のピークなら粗グリッドが自然に重なってくれるので、そのままで問題なし。離れた複数のpeakでも、粗探索でどちらがデータに合うかを自動で判定。

    coarse_ks = np.unique(np.concatenate([grid_around(k0=k0, 
                                                       N=N, 
                                                       ds=ds, 
                                                       B_bins=B_bins_coarse, 
                                                       q_rel=pct_coarse, 
                                                       step_frac=step_frac) for k0 in k_seeds]))
    cand = []
    for k in coarse_ks:
        fit = fit_sin_trend_two_component(uU, uV, s, k,
                                          weights=weights,
                                          trend_degree=trend_degree)
        lam = fit["lamb"]
        r2 = fit['R2']
        partial_r2 = fit["partial_R2"]

        hr = harmonic_ratio_from_fit(uU=uU, uV=uV, s=s, k=k, ds=ds,
                                     weights=weights,
                                     trend_degree=trend_degree)
        hr_energy = hr['hr_energy']
        hr_amp = hr['hr_amp']
        cand.append({'k': float(k), 'lambda': float(lam), 'R2': float(r2), 
                     "partial_R2": float(partial_r2), 'hr_energy': float(hr_energy), 
                     'hr_amp': float(hr_amp), 'fit': fit})
    best_cand, _ = choose_k_from_tie_set(cand, 
                                        lambda_prior=lambda_prior)
    k_best = best_cand['k']
    lambda_best = best_cand['lambda']
    R2_best = best_cand['R2']
    partial_R2_best = best_cand['partial_R2']
    hr_amp_best = best_cand['hr_amp']
    hr_energy_best = best_cand['hr_energy']
    print(f"Coarse search best k={k_best:.4f} rad/length: lambda={lambda_best:.4f}, R2={R2_best:.4f}, partial R2={partial_R2_best:.4f}, hr(E3/E1)_amp={hr_amp_best:4f}, hr(E3/E1)_energy={hr_energy_best:.4f}")

    # 2) 細探索（best の ±10% or ±0.5bin）
    fine_ks = grid_around(k0=k_best, 
                           N=N, 
                           ds=ds, 
                           B_bins=B_bins_fine, 
                           q_rel=pct_fine, 
                           step_frac=step_frac)

    cand2 = []
    for k in fine_ks:
        fit = fit_sin_trend_two_component(uU, uV, s, k,
                                          weights=weights,
                                          trend_degree=trend_degree)
        lam = fit["lamb"]
        r2 = fit['R2']
        partial_r2 = fit["partial_R2"]

        hr = harmonic_ratio_from_fit(uU=uU, uV=uV, s=s, k=k, ds=ds,
                                     weights=weights,
                                     trend_degree=trend_degree)
        hr_energy = hr['hr_energy']
        hr_amp = hr['hr_amp']
        cand2.append({'k': float(k), 'lambda': float(lam), 'R2': float(r2), 
                      "partial_R2": float(partial_r2), 'hr_energy': float(hr_energy), 
                      'hr_amp': float(hr_amp), 'fit': fit})

    best_cand2, _ = choose_k_from_tie_set(cand2, 
                                         lambda_prior=lambda_prior)
    k_best2 = best_cand2['k']
    lambda_best2 = best_cand2['lambda']
    R2_best2 = best_cand2['R2']
    partial_R2_best2 = best_cand2['partial_R2']
    hr_amp_best2 = best_cand2['hr_amp']
    hr_energy_best2 = best_cand2['hr_energy']
    fit_best2 = best_cand2['fit']
    print(f"Fine search best k={k_best2:.4f} rad/length: lambda={lambda_best2:.4f}, R2={R2_best2:.4f}, partial R2={partial_R2_best2:.4f}, hr(E3/E1)_amp={hr_amp_best2:4f}, hr(E3/E1)_energy={hr_energy_best2:.4f}")

    return dict(
        k_hat=k_best2,
        lambda_hat=lambda_best2,
        R2=R2_best2,
        partial_R2=partial_R2_best2,
        hr_amp=hr_amp_best2,
        hr_energy=hr_energy_best2,
        A_comb=fit_best2['A_comb'],
        A_U=fit_best2['A_U'],
        A_V=fit_best2['A_V'],
        uU_cos=fit_best2['uU_cos'],
        uU_sin=fit_best2['uU_sin'],
        uV_cos=fit_best2['uV_cos'],
        uV_sin=fit_best2['uV_sin'],
        coef_U=fit_best2['coef_U'],
        coef_V=fit_best2['coef_V'],
        yhat_U=fit_best2['yhat_U'],
        yhat_V=fit_best2['yhat_V'],
        fit=fit_best2,

        k_coarse=float(k_best),
        lambda_coarse=float(lambda_best),
        R2_coarse=float(R2_best),
        partial_R2_coarse=float(partial_R2_best),
        hr_amp_coarse=float(hr_amp_best),
        hr_energy_coarse=float(hr_energy_best),
        candidates_coarse=cand,
        candidates_fine=cand2
    )
"""

                                        
                                  
                                     
                                

def make_k_grid(L, 
                N_cyc=0.5, 
                N_pp=20, 
                ds_eff=1,                                    
                q_fine=4,
                verbose=True):
                                          
    L = float(L)
                                
    k_max = 2.0 * np.pi / (N_pp * ds_eff)
                                
    k_min = 2*np.pi * N_cyc / L
    if k_max <= k_min:
        raise ValueError("Invalid band: k_max must be > k_min.")
                               
    dk = np.pi / (q_fine * L)
                    
    n = int(np.floor((k_max - k_min) / dk)) + 1
    ks = k_min + dk * np.arange(n, dtype=float)
    if ks[-1] < k_max - 1e-12:
        ks = np.append(ks, k_max)
    if verbose:
        print(f"探索波数 grid: k_min={k_min:.4f}, k_max={k_max:.4f}, dk={dk:.4f}, 点数={len(ks)}")

    return ks, k_min, k_max, dk                          

def choose_k_from_tie_set(candidates, 
                          dR2_tol=0.006, 
                          dhr_amp_tol=0.05,                  
                          verbose=True): 
    ###
    # candidates: list of dict
                                                                                                 
    ###
                            
    R2_vals = np.array([c['R2'] for c in candidates], float)
    R2_max = R2_vals.max()
    tie = [c for c in candidates if (R2_max - c['R2']) <= dR2_tol]
    if verbose:
        print("複数のkが統計学的に同等（ΔR² ≤ {:.4f}）: {} 個".format(dR2_tol, len(tie)))
        print("k values in tie set:", [c['k'] for c in tie])
                                                    
    if len(tie) == 1:
                        
        return tie[0], tie    
    hr_amp_vals = np.array([c['hr_amp'] for c in tie])
    if np.all(~np.isfinite(hr_amp_vals)):
                                            
        tie2 = tie
        hr_amp_min = np.nan
    else:        
        hr_amp_min = float(np.nanmin(hr_amp_vals))
        tie2 = [c for c in tie if (c['hr_amp'] - hr_amp_min) <= dhr_amp_tol]
        if len(tie2) == 0:
                                 
            tie2 = tie
    if verbose:
        print("さらにtie set内での複数のkでhr_ampに大きな差がない（Δhr_amp ≤ {:.4f}）: {} 個".format(dhr_amp_tol, len(tie2)))
        print("k values in tie set2:", [c['k'] for c in tie2])
               
    tie_sorted = sorted(tie2, key=lambda c: float(c['k']))             
    best = tie_sorted[0]
    return best, tie_sorted


def grid_search_k_band(uU, uV, s, ds,
                       N_cyc=1.0,
                       N_pp=20,
                       ds_eff=1,                                                                             
                       q_fine=4,                 # Δk = π/(q_fine*L).  q_fine=4 -> π/(4L)
                       weights=None,
                       trend_degree=1,
                       dR2_tol=0.006,
                       dhr_amp_tol=0.05,                  
                       boundary_margin_bins=2,                         
                       verbose=True
):
    ###
                                                

    # Args:
                                   
                               
                            
    #   q_fine: Δk = π/(q_fine*L)
                                                               
    ###

    uU = np.asarray(uU, float)
    uV = np.asarray(uV, float)
    s  = np.asarray(s, float)

    L = float(s[-1] - s[0])
    if L <= 0:
        raise ValueError("Invalid s: non-positive length.")

                             
    ks, k_min, k_max, dk = make_k_grid(L, 
                N_cyc=N_cyc, 
                N_pp=N_pp, 
                ds_eff=ds_eff,
                q_fine=q_fine,
                verbose=verbose)
        

    cand = []
    for k in ks:
        fit = fit_sin_trend_two_component(
            uU, uV, s, k,
            weights=weights,
            trend_degree=trend_degree
        )

        hr = harmonic_ratio_from_fit(
            uU=uU, uV=uV, s=s, k=k, ds=ds,
            weights=weights,
            trend_degree=trend_degree
        )

        cand.append({
            "k": float(k),
            "lambda": float(fit["lamb"]),
            "R2": float(fit["R2"]),
            "partial_R2": float(fit["partial_R2"]),
            "hr_energy": float(hr["hr_energy"]),
            "hr_amp": float(hr["hr_amp"]),
            "fit": fit
        })

    best, tie_sorted = choose_k_from_tie_set(
        cand,
        dR2_tol=dR2_tol,
        dhr_amp_tol=dhr_amp_tol,
        verbose=verbose
    )

    k_hat = float(best["k"])
    fit_best = best["fit"]

                                              
                     
    idx_hat = int(np.argmin(np.abs(ks - k_hat)))                      
    boundary_hit_low  = (idx_hat <= boundary_margin_bins)
    boundary_hit_high = (idx_hat >= (len(ks) - 1 - boundary_margin_bins))

    if verbose:
        print("Fine-only grid search")
        print(f"  L = {L:.4f} (same unit as s)")
        print(f"  dk = {dk:.6g} (same unit as k)")
        print(f"  Number of grid = {len(ks)}")
        print(f"  best k = {k_hat:.6g}, lambda = {best['lambda']:.4f}, "
              f"R2 = {best['R2']:.4f}, partial_R2 = {best['partial_R2']:.4f}, "
              f"hr_amp = {best['hr_amp']:.4f}")
        if boundary_hit_low:
            print("  boundary_hit_low = True")
        if boundary_hit_high:
            print("  boundary_hit_high = True")

    out = dict(
        k_hat=k_hat,
        lambda_hat=float(best["lambda"]),
        R2=float(best["R2"]),
        partial_R2=float(best["partial_R2"]),
        hr_amp=float(best["hr_amp"]),
        hr_energy=float(best["hr_energy"]),
        A_comb=float(fit_best["A_comb"]),
        A_U=float(fit_best["A_U"]),
        A_V=float(fit_best["A_V"]),
        uU_cos=float(fit_best["uU_cos"]), uU_sin=float(fit_best["uU_sin"]),
        uV_cos=float(fit_best["uV_cos"]), uV_sin=float(fit_best["uV_sin"]),
        coef_U=fit_best["coef_U"],
        coef_V=fit_best["coef_V"],
        yhat_U=fit_best["yhat_U"],
        yhat_V=fit_best["yhat_V"],
        fit=fit_best,

                
        k_min=float(k_min),
        k_max=float(k_max),
        ks=ks,
        dk=float(dk),
        q_fine=int(q_fine),
        L=float(L),
        n_grid=int(len(ks)),
        N_pp=int(N_pp),
        N_cyc=float(N_cyc),
        ds_eff=float(ds_eff),
        trend_degree=int(trend_degree),
        boundary_hit_low=bool(boundary_hit_low),
        boundary_hit_high=bool(boundary_hit_high),
        candidates=cand,
        tie_sorted=tie_sorted
    )
    return out


                     
def clip_mesh_to_coords(P_3darr, mesh, margin=10):
                          
    # Args:
                            
                        
                                                     
    # Returns:
                                 
    
    P_3darr = np.asarray(P_3darr, float)
    x, y, z = P_3darr.T
    bounds = (x.min()-margin, x.max()+margin,
              y.min()-margin, y.max()+margin,
              z.min()-margin, z.max()+margin)

                
    mesh = mesh.compute_normals(
        point_normals=True, cell_normals=False,
        split_vertices=False,
        auto_orient_normals=True,
        consistent_normals=True,
        inplace=False
    )

                                             
    cropped_mesh = mesh.clip_box(bounds, invert=False)

    return cropped_mesh

"""
# PCA法：近傍の「脳表点座標（位置）」の広がりを PCA -> "平面っぽいか/トンネルっぽいか"と法線を推定
# 「2面が入って“厚み”が主成分になって法線が壊れる」問題がPCA法の弱点
def analyze_surface_constraint_pca(
    P_3darr,                         # (N,3) voxel座標
    dPsin_3darr,                     # (N,3) 変位
    g1_3d,                           # bucklingモード主方向（N,3）単位ベクトル推(QC)
    ds_vox,                          # 1点あたり弧長ステップ（voxel）
    tree,                            # cKDTree(face_centers_cropped_mesh)
    face_centers_cropped_mesh,       # (M,3) voxel座標（面中心 or 頂点）
    r_patch_vox,                     # パッチ探索半径（voxel）
    stride=10,                       # 0.1mm samplingなら 1mmごと=10
    min_neighbors=5,                 # PCAに必要な最低点数
    flat_thresh=0.5,                 # flatness=lam3/lam2 の閾値（トンネル判定）
    tunnel_ratio_thresh=0.5,         # トンネル割合が過半数なら Excluded
    sheet_coverage_thresh=0.5,       # sheet-like の点が過半数未満なら Excluded
    squared_displacement_coverage_thresh=0.6,      # sheet-like で説明できる変位エネルギー割合、変位の大半は「壁が曖昧／評価不能な場所」で起きている
    R_out_thresh=0.25,               # 面外率の閾値（R_out < 0.25 → predominantly tangential; 面との角で sin²θ、θ_eq=30°）
    verbose=True
):
    # 目的：局所PCAで「脳表の実効法線」を推定（flatness_ratio=lam3/lam2）
    #       tunnel_ratio（flatnessが大きい=壁が定義しにくい点の割合）
    #       sheet_coverage（flatnessが小さい=壁が定義できる点の割合）
    #       R_out = (法線方向エネルギー)/(全エネルギー) を sheet-like 点だけで計算
    #       判定：Excluded / In-plane Buckling / Terrain Following
    #       ※重要：R_out は flatness <= flat_thresh の点だけで計算する（物理的に意味がある点に限定）

    P_3darr = np.asarray(P_3darr, float)
    dPsin_3darr = np.asarray(dPsin_3darr, float)

    assert P_3darr.shape == dPsin_3darr.shape
    n_points = len(P_3darr)

    # --- 0. アンカー作成 ---
    anchor_indices = np.arange(0, n_points, stride)
    if anchor_indices[-1] != n_points - 1:
        anchor_indices = np.append(anchor_indices, n_points - 1)

    anchor_pts = P_3darr[anchor_indices]
    s_full = np.arange(n_points) * ds_vox
    s_anchors = s_full[anchor_indices]

    if verbose:
        print(f"Loop start: Total {len(anchor_pts)} anchors.")

    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)

    anchor_normals = []
    anchor_flatness = []
    anchor_has_cortex = []
    neighbors_list = []
    prev_normal = None

    # --- 1. アンカー毎に局所PCA ---
    for pt in anchor_pts:
        idx = tree.query_ball_point(pt, r=r_patch_vox)

        if len(idx) < min_neighbors:
            # パッチ空/少なすぎる → cortex法線が信頼できない点
            has_cortex = False
            v3 = prev_normal.copy() if prev_normal is not None else np.array([0.0, 0.0, 1.0])
            flatness_ratio = np.nan  # ★ 値なし（トンネル扱いにせず、後で除外）
        else:
            has_cortex = True
            neighbors = face_centers_cropped_mesh[idx]
            neighbors_list.append(neighbors)
            pca.fit(neighbors)

            lam1, lam2, lam3 = pca.explained_variance_  # lam1>=lam2>=lam3
            v3 = pca.components_[2]                    # 最小固有値方向＝法線

            if lam2 > 1e-9:
                flatness_ratio = lam3 / lam2
            else:
                flatness_ratio = np.nan

        # 法線符号の整合性（急な反転防止）
        if prev_normal is not None and np.dot(prev_normal, v3) < 0:
            v3 = -v3
        prev_normal = v3

        anchor_normals.append(v3)
        anchor_flatness.append(flatness_ratio)
        anchor_has_cortex.append(has_cortex)

    anchor_normals = np.array(anchor_normals, float)
    anchor_flatness = np.array(anchor_flatness, float)
    anchor_has_cortex = np.array(anchor_has_cortex, float)  # 補間のため float

    # --- 2. 補間（アンカー→全点） ---
    func_n = interp1d(s_anchors, anchor_normals, axis=0, kind='linear', fill_value="extrapolate")
    func_f = interp1d(s_anchors, anchor_flatness, axis=0, kind='linear', fill_value="extrapolate")
    func_c = interp1d(s_anchors, anchor_has_cortex, kind='nearest', fill_value="extrapolate")  # True/False は nearest

    normals_full = func_n(s_full)
    flatness_full = func_f(s_full)
    has_cortex_full = func_c(s_full) > 0.5

    # 補間後の法線は再正規化
    norms = np.linalg.norm(normals_full, axis=1, keepdims=True)
    normals_full = normals_full / (norms + 1e-9)

    # --- 3. 有効マスク整理 ---
    # flatness が finite で、かつ has_cortex=True の点だけを「cortex評価可能点」とする
    valid_flat = np.isfinite(flatness_full)
    mask_eval = has_cortex_full & valid_flat

    cortex_coverage = float(np.mean(mask_eval))  # どれくらいの点で cortex 評価できたか（空パッチなどの影響を見る）

    # R_out は sheet-like 点のみ（flatness <= flat_thresh）
    mask_sheet = mask_eval & (flatness_full <= flat_thresh)

    sheet_coverage = float(np.mean(mask_sheet))  # 全点に対する sheet-like の割合（“壁が定義できる長さ”）

    # tunnel_ratio は「評価可能点の中」で flatness > flat_thresh の割合
    # これが “壁が定義できない点がどれだけ支配的か” の指標
    if np.any(mask_eval):
        tunnel_ratio = float(np.mean(flatness_full[mask_eval] > flat_thresh))
    else:
        tunnel_ratio = np.nan

    # --- 4. R_out の計算（sheet-like点のみ） ---
    # 注意：mask_sheet が少ない場合は R_out を計算しても不安定なので判定保留する
    displ = dPsin_3darr
    E_tot_all = float(np.sum(np.sum(displ**2, axis=1)))  # 全点でのエネルギー

    if np.any(mask_sheet):
        displ_sheet = displ[mask_sheet]
        normals_sheet = normals_full[mask_sheet]

        d_perp = np.sum(displ_sheet * normals_sheet, axis=1)
        E_sheet = float(np.sum(np.sum(displ_sheet**2, axis=1)))
        E_perp = float(np.sum(d_perp**2))

        R_out = E_perp / (E_sheet + 1e-9)    # sheet-like点に限定した面外率
        squared_displacement_coverage = E_sheet / (E_tot_all + 1e-9)  # 変位エネルギーのうち、何割がsheet-like点で説明できるか
    else:
        R_out = np.nan
        squared_displacement_coverage = 0.0
    
    # --- 4.1 点ごとの “面外率” r_out_local(s) を計算（★追加） ---
    # 注意：normals_full は全点で定義されているが、mask_eval=False の点は信頼性が低いので、
    # r_out_local_masked では NaN に落とす
    d2 = np.sum(displ**2, axis=1) + 1e-12
    d_dot = np.sum(displ * normals_full, axis=1)
    r_out_local = (d_dot**2) / d2  # 0～1（点ごとの面外率）

    # 評価可能点だけ残す版（mask_eval=False は NaN）
    r_out_local_eval = r_out_local.copy()
    r_out_local_eval[~mask_eval] = np.nan

    # sheet-like点だけ残す版（mask_sheet=False は NaN）
    r_out_local_sheet = r_out_local.copy()
    r_out_local_sheet[~mask_sheet] = np.nan

    # --- 4.2 R_dir = mean(cos^2)（g1_3d と法線の直交性） ---
    # 判定には使わない（QC用）
    g1_3d = np.asarray(g1_3d, float)
    # 長さ合わせ
    min_len_dir = min(len(g1_3d), len(normals_full))
    g1_use = g1_3d[:min_len_dir]
    n_use  = normals_full[:min_len_dir]
    # sheet-like 点だけで評価（mask_sheet と整合）
    mask_sheet_dir = mask_sheet[:min_len_dir]
    if np.any(mask_sheet_dir):
        cos2 = (np.sum(g1_use * n_use, axis=1))**2
        cos2[~mask_sheet_dir] = np.nan
        R_dir = float(np.nanmean(cos2))
    else:
        R_dir = np.nan

    # --- 5. Summary ---
    if verbose:
        print("-" * 30)
        print("解析結果 Summary")
        print(f"flat_thresh                 : {flat_thresh}")
        if np.any(mask_eval):
            fv = flatness_full[mask_eval]
            print(f"Flatness Ratio max/min/mean : {np.max(fv):.4f} / {np.min(fv):.4f} / {np.mean(fv):.4f}")
        else:
            print("Flatness Ratio              : (no valid cortex points)")
        print(f"Cortex coverage (評価可能点): {cortex_coverage:.2%}")
        print(f"Sheet coverage              : {sheet_coverage:.2%}  (flatness <= {flat_thresh})")
        print(f"Tunnel ratio                : {tunnel_ratio:.2%}    (flatness > {flat_thresh} among valid)")
        print(f"Energy coverage (sheet)     : {squared_displacement_coverage:.2%}")
        print(f"R_out (面外率; sheet-only)   : {R_out}")
        print(f"R_dir (=mean(cos^2)): {R_dir:.2f} => 接平面内成分が優勢（約{1-R_dir:.1%}）")
        # R_out ≈ R_dir -> モードがきれい
        # 乖離 -> 局所不安定／境界／登録誤差

    # --- 6. 判定 ---
    # (A) cortex 評価可能点が少なすぎる
    if cortex_coverage < 0.3:
        category = "Excluded (No cortex constraint evaluated)"

    # (B) トンネル的な点が過半数（壁法線が安定して定義できない領域が支配的）
    elif tunnel_ratio_thresh is not None and np.isfinite(tunnel_ratio) and (tunnel_ratio > tunnel_ratio_thresh):
        category = "Excluded (Tunnel/Enclosed)"

    # (C) sheet-like点が少なすぎる（点数割合 or エネルギー割合でゲート）
    elif (sheet_coverage < sheet_coverage_thresh) or (squared_displacement_coverage < squared_displacement_coverage_thresh):
        category = "Excluded (Insufficient sheet)"

    # (D) ここまで来れば sheet-like 領域は十分ある。R_outで in-plane / terrain を判定
    elif np.isfinite(R_out) and (R_out < R_out_thresh):
        category = "In-plane Buckling"
    else:
        category = "Terrain Following / Artifact"

    if verbose:
        print(f"判定: {category}")

    return dict(
        # 判定・主要指標
        category=category,
        R_out=R_out,
        R_dir=R_dir,
        cortex_coverage=cortex_coverage,
        sheet_coverage=sheet_coverage,
        squared_displacement_coverage=squared_displacement_coverage,
        tunnel_ratio=tunnel_ratio,

        # エネルギー内訳（再計算不要にする）
        total_energy=E_tot_all,
        energy_sheet=E_sheet,
        perp_energy=E_perp,

        # 点数（割合だけだと危ない）
        n_points_total=int(n_points), # 全点数
        n_eval=int(mask_eval.sum()), # 評価できた点の数（脳表法線が定義できた点）
        n_sheet=int(mask_sheet.sum()), # その中で「壁（シート）」として信頼できた点の数

        # 閾値・設定（再現性）
        flat_thresh=flat_thresh,
        R_out_thresh=R_out_thresh,
        tunnel_ratio_thresh=tunnel_ratio_thresh,
        sheet_coverage_thresh=sheet_coverage_thresh,
        squared_displacement_coverage_thresh=squared_displacement_coverage_thresh,
        stride=int(stride),
        min_neighbors=int(min_neighbors),
        r_patch_vox=float(r_patch_vox),

        # 可視化用（あなたの今の構成でOK）
        r_out_local=r_out_local,
        r_out_local_eval=r_out_local_eval,
        r_out_local_sheet=r_out_local_sheet,
        mask_eval=mask_eval,
        mask_sheet=mask_sheet,
        flatness_full=flatness_full,
        normals_full=normals_full,
        neighbors_list=neighbors_list
    )
"""

                                                                  
def estimate_effective_normal_from_mesh_normals(
    pt, tree_pts, surf_normals, r_patch, min_neighbors=30, prev_n=None):
                                                                    
    
    idx = tree_pts.query_ball_point(pt, r=r_patch)
    idx = np.array(idx, dtype=int)
    if len(idx) < min_neighbors:
        return None, np.nan, len(idx), idx 

    N = normalize(surf_normals[idx])

                                 
    T = (N.T @ N) / N.shape[0]        # = (1/m) Σ n_i n_i^T
    w, V = np.linalg.eigh(T)                         
    coherence = float(w[-1])                                   
    n_eff = V[:, -1]                             

                          
    if prev_n is not None and np.dot(prev_n, n_eff) < 0:
        n_eff = -n_eff

    return n_eff, coherence, len(idx), idx

def fill_short_gaps_1d(mask, gap_fill_size):
    ###
                                                      
                                                  
                                     
    ###
    mask = np.asarray(mask, dtype=bool)

    if gap_fill_size is None or gap_fill_size <= 1:
        return mask

    structure = np.ones(int(gap_fill_size), dtype=bool)

                        
    mask_padded = np.pad(mask, pad_width=1, mode='edge')
    mask_clean_padded = binary_closing(mask_padded, structure=structure).astype(bool)
    return mask_clean_padded[1:-1]

def analyze_surface_constraint_coherence(
    P_3darr,                                                 
    dPsin_3darr,                                        
    g1_3d,                                
    ds_vox,                                              
    cropped_mesh,                                              
    r_patch_vox,                                     
    stride=10,
    min_neighbors=30,
                           
                                          
                                       
                                                              
                               
    coh_thr=0.75,
    constraint_coverage_thresh=0.5,                                                                             
    squared_displacement_coverage_thresh=0.5,                                                                                        
    # R_out < 0.25 ⇔ R_tan > 0.75 → Predominantly tangential（θ_eq=30°）
    R_out_thresh=0.25,
    gap_fill_size=3,                                                             
    bridge_max_gap_mm=5.0,                                   
    bridge_axis_max_deg=45.0,                                                 
    voxel_size_mm=0.5,
    verbose=True
):
    ### 
         
                                                                     
                                           
                                                             
    #
         
                                                   
                                         
    ###

    P_3darr = np.asarray(P_3darr, float)
    displ = np.asarray(dPsin_3darr, float)
    assert P_3darr.shape == displ.shape
    n_points = len(P_3darr)

    if cropped_mesh is None:
        raise ValueError("cropped_mesh（compute_normals済み）が必要!!")

                       
    anchor_indices = np.arange(0, n_points, stride)
    if anchor_indices[-1] != n_points - 1:
        anchor_indices = np.append(anchor_indices, n_points - 1)

    anchor_pts = P_3darr[anchor_indices]
    s_full = np.arange(n_points) * ds_vox
    s_anchors = s_full[anchor_indices]

                         
    surf_pts = cropped_mesh.points
    if "Normals" in cropped_mesh.point_data:
        surf_normals = cropped_mesh.point_data["Normals"]
    else:
                                  
        surf_normals = cropped_mesh.point_normals
    tree_pts = cKDTree(surf_pts)

    if verbose:
        print(f"Loop start: Total {len(anchor_pts)} anchors.")

                                         
    anchor_normals = []
    anchor_coherence = []
    anchor_has_cortex = [] 
    anchor_has_constraint = []
    anchor_nnb = []
    anchor_neighbors_idx = []

    prev_n = None
    default_n = np.array([0.0, 0.0, 1.0])
    for pt in anchor_pts:
        n_eff, coh, nnb, idx = estimate_effective_normal_from_mesh_normals(
            pt, tree_pts, surf_normals, r_patch_vox, min_neighbors=min_neighbors, prev_n=prev_n
        )
        anchor_neighbors_idx.append(idx)
        anchor_nnb.append(nnb)

                              
        has_cortex = (nnb >= min_neighbors)
        anchor_has_cortex.append(has_cortex)

               
        if not has_cortex:
                                                                              
            n_use = prev_n.copy() if prev_n is not None else default_n.copy()
            coh_use = np.nan
        else:
                                 
            n_use = n_eff
            coh_use = coh

                          
        if prev_n is not None and np.dot(prev_n, n_use) < 0:
            n_use = -n_use
        
                   
        prev_n = n_use

                                        
        anchor_normals.append(n_use)
        anchor_coherence.append(coh_use)
                                                   
        is_good = bool(has_cortex and np.isfinite(coh_use) and coh_use >= coh_thr)
        anchor_has_constraint.append(is_good)

    anchor_normals = np.asarray(anchor_normals, float)
    anchor_coherence = np.asarray(anchor_coherence, float)
    anchor_has_cortex = np.asarray(anchor_has_cortex, bool)
    anchor_has_constraint = np.asarray(anchor_has_constraint, bool)
    anchor_nnb = np.asarray(anchor_nnb, int)
                                        
    _ = gap_fill_size
    anchor_has_constraint_clean = anchor_has_constraint.copy()

                                               
    from sin_trend_fitting_helpers import interpolate_good_anchor_normals_with_bridge

    anchor_normals_good = np.full_like(anchor_normals, np.nan)
    anchor_normals_good[anchor_has_constraint] = anchor_normals[anchor_has_constraint]
    bridged = interpolate_good_anchor_normals_with_bridge(
        s_full,
        s_anchors,
        anchor_normals_good,
        anchor_has_constraint,
        voxel_size_mm=float(voxel_size_mm),
        bridge_max_gap_mm=float(bridge_max_gap_mm),
        bridge_axis_max_deg=float(bridge_axis_max_deg),
    )
    normals_full = np.asarray(bridged["normals_eval"], float)
    has_constraint_full = np.asarray(bridged["has_constraint_full"], bool)
    eval_normal_source = np.asarray(bridged["eval_normal_source"], int)

                                                    
    interp_cortex = interp1d(s_anchors, anchor_has_cortex.astype(float), kind="nearest", fill_value="extrapolate")
    interp_coherence = interp1d(s_anchors, anchor_coherence, kind="nearest", fill_value="extrapolate")
    has_cortex_full = interp_cortex(s_full) > 0.5
    coherence_full = interp_coherence(s_full)

    okn = np.isfinite(normals_full).all(axis=1)
    nn = np.linalg.norm(normals_full[okn], axis=1, keepdims=True)
    ok_u = (nn > 0).ravel()
    idx_ok = np.flatnonzero(okn)
    normals_full[idx_ok[ok_u]] = normals_full[idx_ok[ok_u]] / nn[ok_u]

                      
    mask_cortex_eval = has_cortex_full
    cortex_coverage = float(np.mean(mask_cortex_eval))
    mask_eval = has_constraint_full & okn
    constraint_coverage = float(np.mean(mask_eval))

                                           
    E_tot_all = float(np.sum(displ * displ))

    E_constrained = 0.0           
    E_perp = 0.0

    if np.any(mask_eval):
        d = displ[mask_eval]
        n = normals_full[mask_eval]
        d_dot = np.sum(d * n, axis=1)

        E_constrained = float(np.sum(d * d))
        E_perp = float(np.sum(d_dot * d_dot))
        R_out = E_perp / (E_constrained + 1e-12)
        squared_displacement_coverage = E_constrained / (E_tot_all + 1e-12)
    else:
        R_out = np.nan
        squared_displacement_coverage = 0.0

                               
    d2 = np.sum(displ * displ, axis=1) + 1e-12
    d_dot_full = np.full(n_points, np.nan, float)
    d_dot_full[okn] = np.sum(displ[okn] * normals_full[okn], axis=1)
    r_out_local = (d_dot_full * d_dot_full) / d2
    r_out_local_eval = r_out_local.copy()
    r_out_local_eval[~mask_eval] = np.nan

                             
    R_dir = np.nan
    g1_3d = np.asarray(g1_3d, float)
    m = min(len(g1_3d), len(normals_full))
    g1_use = g1_3d[:m]
    n_use = normals_full[:m]
    mm = mask_eval[:m]
    if np.any(mm):
        cos2 = (np.sum(g1_use * n_use, axis=1))**2
        cos2[~mm] = np.nan
        R_dir = float(np.nanmean(cos2))                                                  

    # --- 5. Summary ---
    if verbose:
        print("-" * 30)
        print("解析結果 Summary (coherence-based surface constraint)")
        print(f"coherence threshold (coh_thr)               : {coh_thr}")
        print(f"cortex_coverage (評価可能点).                 : {cortex_coverage:.2%}")
        print(f"constraint_coverage                         : {constraint_coverage:.2%} (good/bridged; C≥{coh_thr})")
        print(f"bridge_max_gap_mm / axis_max_deg            : {bridge_max_gap_mm:g} / {bridge_axis_max_deg:g}")
        print(f"squared_displacement_coverage (constraint)                : {squared_displacement_coverage:.2%}")
        print(f"R_out (面外率; constraint)                   : {R_out:.2f}")
        print(f"R_dir (=mean(cos^2), 脳表法線方向成分)         : {R_dir:.1%} <=> 接平面内成分（{1 - R_dir:.1%}）")

    if verbose:
        print(f"E_tot_all: {E_tot_all:.2f}")
        print(f"E_constrained: {E_constrained:.2f}")
        print(f"E_perp: {E_perp:.2f}")
        print(f"R_out: {R_out:.2f}")
        print(f"squared_displacement_coverage: {squared_displacement_coverage:.2f}")

    return dict(
                               
        R_out=R_out,
        R_dir=R_dir,
        cortex_coverage=cortex_coverage,
        constraint_coverage=constraint_coverage,
        squared_displacement_coverage=squared_displacement_coverage,

                 
        total_energy=E_tot_all,
        energy_constrained=E_constrained,
        perp_energy=E_perp,

            
        n_points_total=int(n_points),
        n_eval=int(mask_eval.sum()),
        
                    
        coh_thr=float(coh_thr),
        constraint_coverage_thresh=float(constraint_coverage_thresh),
        squared_displacement_coverage_thresh=float(squared_displacement_coverage_thresh),
        R_out_thresh=float(R_out_thresh),
        stride=int(stride),
        min_neighbors=int(min_neighbors),
        r_patch_vox=float(r_patch_vox),
        gap_fill_size=gap_fill_size,
        bridge_max_gap_mm=float(bridge_max_gap_mm),
        bridge_axis_max_deg=float(bridge_axis_max_deg),
        eval_normal_source=eval_normal_source,

                        
        anchor_indices=anchor_indices,
        anchor_pts=anchor_pts,
        anchor_normals=anchor_normals,
        anchor_coherence=anchor_coherence,
        anchor_has_cortex=anchor_has_cortex,
        anchor_has_constraint_raw=anchor_has_constraint,
        anchor_has_constraint_clean=anchor_has_constraint_clean,
        anchor_nnb=anchor_nnb,
        anchor_neighbors_idx=anchor_neighbors_idx,

                   
        normals_full=normals_full,
        coherence_full=coherence_full,
        mask_eval=mask_eval,
        r_out_local=r_out_local,
        r_out_local_eval=r_out_local_eval
    )


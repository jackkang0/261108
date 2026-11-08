"""
Flicker KPI — AI 偵測（路線 A：以現有 metric 為特徵的分類器）
============================================================
作法
  1. 以「格子」為樣本單位（格大小跟人工框的單位一致）。
  2. 每格抽出一個特徵向量：
       - 20 項 metric 的格內 max 與 p99，各自除以該序列整張圖的中位數
         （相對於自身噪聲底的倍率，跨素材才可移植）
       - 比值特徵：block/pixel variance（空間同相性）、色度 vs 亮度
       - 時間型態：von Neumann ratio、正負號翻轉次數、最大單幀跳動佔比
       - 內容特徵：亮度、梯度能量、局部對比
  3. 標籤：與人工框重疊的格 = 1；其餘格 = 0。
  4. 模型：邏輯迴歸（純 numpy/scipy，分數 = w·x + b，可攜、可解釋）
     或 gradient boosting（需要 scikit-learn，可學非線性交互作用）。
  5. 輸出每格一個 0~1 分數，可當成第 21 項 metric 使用。

與門檻法的差別：門檻法是 20 個軸向切面的聯集，無法表達跨項目的條件；
本模組在同一組數字上學一個判定邊界，可表達「A 高且 B 低才算閃」這類規則。
"""

import json
import math

import numpy as np
import cv2

from flicker_kpi_core import METRIC_NAMES, derive_channels, DEFAULT_Y_MODE

LEVELS = ("pixel", "block")
KEYS = [(lv, m) for lv in LEVELS for m in METRIC_NAMES]
EPS = 1e-6
MODEL_VERSION = 1


# --------------------------------------------------------------------------
# 特徵
# --------------------------------------------------------------------------
def _tiles_of(arr, tile):
    """(H,W) -> (n_tile, tile*tile)；不足一格的邊緣捨去"""
    H, W = arr.shape
    th, tw = H // tile, W // tile
    if th == 0 or tw == 0:
        return np.zeros((0, tile * tile), np.float32), (0, 0)
    sub = arr[:th * tile, :tw * tile].reshape(th, tile, tw, tile)
    return sub.transpose(0, 2, 1, 3).reshape(th * tw, tile * tile), (th, tw)


def _kth(flat, frac):
    """每列第 frac 分位（由大到小）的值；frac=0 -> 最大值"""
    n = flat.shape[1]
    k = max(1, min(n, int(round(n * frac)) or 1))
    part = np.partition(flat, n - k, axis=1)[:, n - k:]
    return part.min(axis=1)


def _temporal_maps(stack, y_mode=DEFAULT_Y_MODE):
    """由原始 6 幀算出時間型態的 per-pixel 圖"""
    Y = derive_channels(stack, y_mode)["Y"]            # (N,H,W)
    d = np.diff(Y, axis=0)
    p2p = Y.max(axis=0) - Y.min(axis=0)
    cen = Y - Y.mean(axis=0, keepdims=True)
    fi = (d ** 2).sum(axis=0) / ((cen ** 2).sum(axis=0) + EPS)   # von Neumann ratio
    sign = np.sign(d)
    chg = (np.diff(sign, axis=0) != 0).sum(axis=0).astype(np.float32)
    jump = np.abs(d).max(axis=0) / (p2p + EPS)
    return {"p2p": p2p.astype(np.float32), "fi": fi.astype(np.float32),
            "chg": chg, "jump": jump.astype(np.float32)}


def feature_names():
    names = []
    for lv, m in KEYS:
        names += [f"{lv}:{m}:max", f"{lv}:{m}:p99"]
    for c in ("R", "G", "B", "Y", "L1"):
        names.append(f"coh:{c}")                       # block/pixel variance
    names += ["chroma:RB_over_Y", "chroma:Y_over_L1", "ratio:blockY_over_pixelY_diff"]
    names += ["temporal:fi_at_peak", "temporal:fi_median",
              "temporal:signchg_at_peak", "temporal:jump_at_peak"]
    names += ["content:luma", "content:grad", "content:contrast"]
    return names


def tile_features(result, tile, y_mode=DEFAULT_Y_MODE, with_raw=False):
    """
    回傳 (X, grid)，with_raw=True 時回傳 (X, grid, R)
      X = (n_tile, n_feature) float32，特徵順序與 feature_names() 一致
      R = (n_tile, 20) 每格 20 項 metric 的原始最大值（給傳統門檻法當對照用）
    """
    bs = result.get("block_size", 2)
    cols, grid = [], None
    raw_max = {}
    for lv, blk in (("pixel", 1), ("block", bs)):
        t_lv = max(1, tile // blk)
        for m in METRIC_NAMES:
            arr = result[lv]["maps"][m]
            flat, g = _tiles_of(arr, t_lv)
            if grid is None or (g[0] * g[1]) < (grid[0] * grid[1]):
                grid = g
            med = float(np.median(arr)) + EPS           # 該序列自身的噪聲底
            mx, p99 = _kth(flat, 0.0), _kth(flat, 0.01)
            raw_max[(lv, m)] = mx
            cols += [np.log1p(mx / med), np.log1p(p99 / med)]

    n = min(len(c) for c in cols)
    cols = [c[:n] for c in cols]
    raw_max = {k: v[:n] for k, v in raw_max.items()}

    for c in ("R", "G", "B", "Y", "L1"):               # 空間同相性
        cols.append(np.log1p(raw_max[("block", f"{c}_variance")] /
                             (raw_max[("pixel", f"{c}_variance")] + EPS)))
    rb = np.maximum(raw_max[("pixel", "R_variance")], raw_max[("pixel", "B_variance")])
    cols.append(np.log1p(rb / (raw_max[("pixel", "Y_variance")] + EPS)))
    cols.append(np.log1p(raw_max[("pixel", "Y_variance")] /
                         (raw_max[("pixel", "L1_variance")] + EPS)))
    cols.append(np.log1p(raw_max[("block", "Y_difference")] /
                         (raw_max[("pixel", "Y_difference")] + EPS)))

    tm = _temporal_maps(result["stack"], y_mode)        # 時間型態
    p2p_flat, _ = _tiles_of(tm["p2p"], tile)
    idx = p2p_flat.argmax(axis=1)                       # 格內擺幅最大的那個 pixel
    rows = np.arange(p2p_flat.shape[0])
    for key in ("fi", "chg", "jump"):
        flat, _ = _tiles_of(tm[key], tile)
        v = flat[rows, idx]
        if key == "fi":
            cols.append(v[:n])
            cols.append(np.median(flat, axis=1)[:n])
        else:
            cols.append(v[:n])

    gray = result["stack"].mean(axis=0)                 # 內容
    gray = (0.299 * gray[..., 0] + 0.587 * gray[..., 1] + 0.114 * gray[..., 2])
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gflat, _ = _tiles_of(gray.astype(np.float32), tile)
    eflat, _ = _tiles_of(np.abs(gx) + np.abs(gy), tile)
    cols.append(np.log1p(gflat.mean(axis=1))[:n])
    cols.append(np.log1p(eflat.mean(axis=1))[:n])
    cols.append(np.log1p(gflat.std(axis=1))[:n])

    X = np.stack([np.asarray(c[:n], np.float32) for c in cols], axis=1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if not with_raw:
        return X, grid
    R = np.stack([raw_max[k] for k in KEYS], axis=1).astype(np.float64)
    return X, grid, R


def tile_labels(boxes, grid, tile):
    """與任一框重疊的格 = 1，其餘 = 0"""
    th, tw = grid
    y = np.zeros(th * tw, np.int8)
    for (x0, y0, x1, y1) in boxes or ():
        for r in range(max(0, y0 // tile), min(th, -(-y1 // tile))):
            for c in range(max(0, x0 // tile), min(tw, -(-x1 // tile))):
                y[r * tw + c] = 1
    return y


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------
def _standardize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0
    return (X - mean) / std, mean, std


def fit_logreg(X, y, l2=1.0, max_iter=300):
    """L2 邏輯迴歸（class weight balanced），回傳 (w, b)"""
    from scipy.optimize import minimize
    Xs, mean, std = _standardize(X)
    y = np.asarray(y, float)
    npos, nneg = max(y.sum(), 1), max((1 - y).sum(), 1)
    wgt = np.where(y > 0, 0.5 / npos, 0.5 / nneg) * len(y)

    def obj(p):
        w, b = p[:-1], p[-1]
        z = Xs @ w + b
        # 穩定版 log(1+exp(z))
        ll = np.logaddexp(0, z) - y * z
        loss = float((wgt * ll).sum() / len(y) + 0.5 * l2 * (w @ w) / len(y))
        s = 1.0 / (1.0 + np.exp(-z))
        g = (wgt * (s - y)) / len(y)
        return loss, np.concatenate([Xs.T @ g + l2 * w / len(y), [g.sum()]])

    p0 = np.zeros(Xs.shape[1] + 1)
    res = minimize(obj, p0, jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter})
    return {"type": "logreg", "mean": mean, "std": std,
            "coef": res.x[:-1], "intercept": float(res.x[-1])}


def fit_gbdt(X, y, max_depth=3, max_iter=200, l2=1.0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        max_depth=max_depth, max_iter=max_iter, l2_regularization=l2,
        learning_rate=0.08, class_weight="balanced", random_state=0)
    clf.fit(X, y)
    return {"type": "gbdt", "clf": clf}


def predict(model, X):
    if model["type"] == "logreg":
        Xs = (X - model["mean"]) / model["std"]
        z = Xs @ model["coef"] + model["intercept"]
        return 1.0 / (1.0 + np.exp(-z))
    return model["clf"].predict_proba(X)[:, 1]


def fit(X, y, kind="logreg", **kw):
    if kind == "gbdt":
        try:
            return fit_gbdt(X, y, **kw)
        except ImportError:
            pass
    return fit_logreg(X, y, **kw)


# --------------------------------------------------------------------------
# 訓練 / 驗證（以序列為單位切分，避免同一張圖的格子互相洩漏）
# --------------------------------------------------------------------------
def group_cv_scores(X, y, groups, kind="logreg", n_splits=5, **kw):
    """回傳 out-of-fold 分數（每個樣本都由沒看過它所屬序列的模型預測）"""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    n_splits = min(n_splits, len(uniq))
    oof = np.zeros(len(y), float)
    if n_splits < 2:
        return oof, 0
    folds = np.array_split(np.random.default_rng(0).permutation(uniq), n_splits)
    for te in folds:
        te_mask = np.isin(groups, te)
        if te_mask.all() or not te_mask.any():
            continue
        m = fit(X[~te_mask], y[~te_mask], kind, **kw)
        oof[te_mask] = predict(m, X[te_mask])
    return oof, n_splits


def _sweep(scores, y, groups):
    """
    門檻掃描的共用核心：把所有格子依分數由大到小排一次，回傳四個等長陣列
      u[i]    第 i 個相異分數（由大到小）
      fmax[i] 門檻取 u[i] 時「單一序列的框外誤判格數」最大值   （單調不減）
      rec[i]  抓到的框數                                       （單調不減）
      fp[i]   框外誤判格總數                                   （單調不減）

    三個量都單調，是因為門檻下降時 pred = (scores >= t) 只會變大。
    這個單調性讓「每序列誤判 <= budget」的可行解恰好是掃描序列的一個 prefix，
    於是不必像舊版那樣對每個候選門檻重算一次 mask（那是 O(相異分數 x 序列數)）。
    同分數的格子會在同一個 i 一起被計入，與 scores >= t 的語意一致。
    """
    scores = np.asarray(scores, float)
    y = np.asarray(y)
    groups = np.asarray(groups)
    n = len(scores)
    z = np.zeros(0)
    if n == 0:
        return z, z, z, z

    order = np.argsort(-scores, kind="stable")
    sc = scores[order]
    yo = np.asarray(y)[order]
    _, gc = np.unique(groups, return_inverse=True)
    gco = gc[order]

    # 每個框外格在「它所屬序列的框外格」之中是第幾個被加入的；
    # 取 prefix 最大值即為「任一序列的誤判數上限」
    neg = np.flatnonzero(yo == 0)
    occ = np.zeros(n, np.int64)
    if neg.size:
        gneg = gco[neg]
        og = np.argsort(gneg, kind="stable")
        blk = np.flatnonzero(np.r_[True, np.diff(gneg[og]) != 0])
        cnt = np.diff(np.r_[blk, len(og)])
        occ[neg[og]] = np.arange(len(og)) - np.repeat(blk, cnt) + 1

    last = np.flatnonzero(np.r_[np.diff(sc) != 0, True])   # 每個相異分數的最後位置
    return (sc[last],
            np.maximum.accumulate(occ)[last],
            np.cumsum(yo == 1)[last],
            np.cumsum(yo == 0)[last])


def pick_thresholds(scores, y, groups, budgets):
    """
    一次掃描求出所有 budget 的分數門檻。回傳 {budget: threshold}。
    規則與舊版逐一試候選門檻的版本完全相同：每序列框外誤判 <= budget 之下
    框召回最大，召回相同時取門檻較高者；抓不到任何框時給 inf（等於不抓）。
    """
    u, fmax, rec, _ = _sweep(scores, y, groups)
    out = {int(b): np.inf for b in budgets}
    if not len(u):
        return out
    for b in budgets:
        b = int(b)
        n_ok = int(np.searchsorted(fmax, b, side="right"))   # 可行的 prefix 長度
        if n_ok == 0:
            continue
        best = int(rec[n_ok - 1])                            # rec 單調 -> 最佳在尾端
        if best <= 0:
            continue
        i = int(np.searchsorted(rec[:n_ok], best, side="left"))
        out[b] = float(u[i])
    return out


def recall_at_budget(scores, y, groups, budgets):
    """
    以每組序列允許幾個「框外格被判成閃」為預算，回報框的召回率。
    門檻取全域同一個值（與實際使用方式一致）。
    """
    y = np.asarray(y)
    n_seq = max(len(np.unique(np.asarray(groups))), 1)
    npos = max(int((y == 1).sum()), 1)
    u, fmax, rec, fp = _sweep(scores, y, groups)
    out = []
    for bud in budgets:
        b = int(bud)
        r, t, f = 0.0, None, 0
        if len(u):
            n_ok = int(np.searchsorted(fmax, b, side="right"))
            if n_ok and rec[n_ok - 1] > 0:
                best = int(rec[n_ok - 1])
                i = int(np.searchsorted(rec[:n_ok], best, side="left"))
                r, t, f = best / npos, float(u[i]), int(fp[i])
        out.append({"budget": b, "recall": r, "threshold": t,
                    "fp_total": f, "n_seq": n_seq})
    return out


def _pick_threshold(scores, y, groups, budget):
    """在訓練資料上挑分數門檻：每組序列的框外誤判格 <= budget，且框召回最大"""
    return pick_thresholds(scores, y, groups, [budget])[int(budget)]


def cv_evaluate(X, y, groups, budgets, kind="logreg", n_splits=5, **kw):
    """
    與實際部署一致的評估：每一折只用訓練序列訓練模型、並在訓練序列上決定分數門檻，
    再套到沒看過的序列。回傳 ([{budget, recall, fp_total, n_seq}], 平均 AUC, 折數)
    """
    X = np.asarray(X, float)
    y = np.asarray(y)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    n_splits = min(n_splits, len(uniq))
    if n_splits < 2:
        return [], float("nan"), 0
    folds = np.array_split(np.random.default_rng(0).permutation(uniq), n_splits)
    aucs = []
    hits = {int(b): 0 for b in budgets}
    fps = {int(b): 0 for b in budgets}
    for te in folds:
        te_mask = np.isin(groups, te)
        tr = ~te_mask
        if not tr.any() or not te_mask.any() or not y[tr].any():
            continue
        m = fit(X[tr], y[tr], kind, **kw)
        s_tr, s_te = predict(m, X[tr]), predict(m, X[te_mask])
        aucs.append(auc(s_te, y[te_mask]))
        thr = pick_thresholds(s_tr, y[tr], groups[tr], budgets)   # 一折掃一次
        for b in budgets:
            t = thr[int(b)]
            pred = s_te >= t
            hits[int(b)] += int((pred & (y[te_mask] == 1)).sum())
            fps[int(b)] += int((pred & (y[te_mask] == 0)).sum())
    npos = max(int((y == 1).sum()), 1)
    out = [{"budget": int(b), "recall": hits[int(b)] / npos,
            "fp_total": fps[int(b)], "n_seq": len(uniq)} for b in budgets]
    return out, float(np.nanmean(aucs)) if aucs else float("nan"), n_splits


L2_GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def tune_l2(X, y, groups, grid=L2_GRID, n_splits=4):
    """以序列切分的 CV AUC 選正則化強度（資料少時影響很大）"""
    best, best_auc = grid[0], -1.0
    for l2 in grid:
        _, a, ns = cv_evaluate(X, y, groups, [0], "logreg", n_splits=n_splits, l2=l2)
        if ns and a == a and a > best_auc:
            best, best_auc = l2, a
    return best, best_auc


def _rank_avg(a):
    """平均排名（同分取平均）。np.argsort(np.argsort(.)) 只給序位排名，
    所有值相同時會依出現順序給 1..n，使 AUC 變成 0 而非 0.5。"""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    gid = np.cumsum(np.r_[True, s[1:] != s[:-1]]) - 1      # 同分的歸為一組
    cnt = np.bincount(gid)
    ends = np.cumsum(cnt)
    avg = (ends - cnt + ends + 1) / 2.0                    # 該組佔的序位取平均
    r = np.empty(len(a), float)
    r[order] = avg[gid]
    return r


def auc(scores, y):
    y = np.asarray(y)
    s = np.asarray(scores, float)
    if not np.isfinite(s).all():        # 模型發散時分數會是 nan/inf
        return float("nan")
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = _rank_avg(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def top_features(model, names, k=10):
    """邏輯迴歸：權重最大的幾項（標準化後，可直接比較大小）"""
    if model["type"] != "logreg":
        return []
    idx = np.argsort(-np.abs(model["coef"]))[:k]
    return [(names[i], float(model["coef"][i])) for i in idx]


# --------------------------------------------------------------------------
# 存檔 / 載入
# --------------------------------------------------------------------------
def save_model(path, model, meta):
    """logreg 存成可攜的 JSON；gbdt 需要 joblib（載入端也要有 scikit-learn）"""
    if model["type"] == "logreg":
        data = {"version": MODEL_VERSION, "type": "logreg", "meta": meta,
                "mean": list(map(float, model["mean"])),
                "std": list(map(float, model["std"])),
                "coef": list(map(float, model["coef"])),
                "intercept": model["intercept"]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    else:
        import joblib
        joblib.dump({"version": MODEL_VERSION, "type": "gbdt",
                     "meta": meta, "clf": model["clf"]}, path)
    return path


def load_model(path):
    if str(path).lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("type") != "logreg":
            raise ValueError("JSON 模型檔的 type 不是 logreg")
        model = {"type": "logreg", "mean": np.array(d["mean"], float),
                 "std": np.array(d["std"], float),
                 "coef": np.array(d["coef"], float),
                 "intercept": float(d["intercept"])}
        return model, d.get("meta", {})
    import joblib
    d = joblib.load(path)
    return {"type": "gbdt", "clf": d["clf"]}, d.get("meta", {})


def score_map(model, result, tile, y_mode=DEFAULT_Y_MODE):
    """回傳 (th, tw) 的分數圖"""
    X, grid = tile_features(result, tile, y_mode)
    s = predict(model, X)
    return s.reshape(grid), grid


def baseline_cv(R, y, groups, budgets, n_splits=5, progress=None):
    """
    傳統門檻法的同條件對照：每一折只用訓練序列求門檻（flicker_kpi_opt.optimize_roi），
    再到沒看過的序列上評估，與 AI 的 out-of-fold 分數可直接比較。
    回傳 [{budget, recall, fp_total, n_seq}]

    每一折只呼叫 optimize_roi 一次、把所有 budget 一起傳進去（它內部本來就在
    掃 budget 列表，候選門檻與 set cover 的前置計算因此只做一次）。
    折數 x budget 數次求解 -> 折數次求解，結果完全相同。
    """
    import flicker_kpi_opt as kopt
    groups = np.asarray(groups)
    y = np.asarray(y)
    uniq = np.unique(groups)
    n_splits = min(n_splits, len(uniq))
    if n_splits < 2:
        return []
    folds = np.array_split(np.random.default_rng(0).permutation(uniq), n_splits)
    buds = [int(b) for b in budgets]
    hit = {b: 0 for b in buds}
    fp = {b: 0 for b in buds}
    for fi, te in enumerate(folds):
        if progress:
            progress(fi, len(folds), f"第 {fi+1}/{len(folds)} 折")
        te_mask = np.isin(groups, te)
        tr = ~te_mask
        B, N = R[tr & (y == 1)], R[tr & (y == 0)]
        if not len(B):
            continue
        # 訓練折的預算按序列數等比例換算
        n_tr_seq = max(len(np.unique(groups[tr])), 1)
        res = kopt.optimize_roi(B, N, [b * n_tr_seq for b in buds])
        yt = y[te_mask]
        Rte = R[te_mask].astype(np.float32)
        for b, r in zip(buds, res):
            t = np.asarray(r["thresholds"], np.float64).astype(np.float32)
            pred = (Rte > t).any(axis=1)
            hit[b] += int((pred & (yt == 1)).sum())
            fp[b] += int((pred & (yt == 0)).sum())
    npos = max(int((y == 1).sum()), 1)
    return [{"budget": b, "recall": hit[b] / npos, "fp_total": fp[b],
             "n_seq": len(uniq)} for b in buds]


# ==========================================================================
# 統一介面：logreg / gbdt（路線 A）與 cnn（路線 B）共用同一組存取與評分函式
# ==========================================================================
def summarize_scores(scores, thr):
    """AI 分數圖的序列層級摘要：任一格 >= 門檻即判為閃"""
    scores = np.asarray(scores, float)
    if scores.size == 0:
        return {"verdict": "—", "n_hi": 0, "n_tile": 0, "max": 0.0, "median": 0.0}
    hi = int((scores >= thr).sum())
    return {"verdict": "FAIL" if hi else "PASS", "n_hi": hi,
            "n_tile": int(scores.size), "max": float(scores.max()),
            "median": float(np.median(scores))}


def roi_summary_scores(scores, thr, boxes, tile):
    """
    roi_summary 的 AI 版本：判定依據換成格的 AI 分數，其餘語意完全相同
      n_box / n_caught / miss   框的總數、抓到數、沒抓到的框
      n_tile / n_fp / fp_tiles  與框不重疊的格子總數、被判成閃的格數與位置 (row, col)
    一個框只要有任一重疊的格 >= 門檻就算抓到（與傳統法「框內任一 metric 超標」對應）。
    """
    scores = np.asarray(scores, float)
    boxes = list(boxes or ())
    if scores.size == 0:
        return {"n_box": len(boxes), "n_caught": 0, "miss": boxes,
                "n_tile": 0, "n_fp": 0, "fp_tiles": []}
    th, tw = scores.shape
    hit = scores >= thr
    cover = np.zeros((th, tw), bool)
    caught, miss = [], []
    for b in boxes:
        x0, y0, x1, y1 = b
        r0, r1 = max(0, y0 // tile), min(th, -(-y1 // tile))
        c0, c1 = max(0, x0 // tile), min(tw, -(-x1 // tile))
        sub = hit[r0:r1, c0:c1]
        (caught if sub.size and sub.any() else miss).append(b)
        cover[r0:r1, c0:c1] = True            # 與框重疊的格不算框外
    out = hit & ~cover
    return {"n_box": len(boxes), "n_caught": len(caught), "miss": miss,
            "n_tile": int((~cover).sum()), "n_fp": int(out.sum()),
            "fp_tiles": [(int(r), int(c)) for r, c in zip(*np.nonzero(out))]}


def save_any(path, model, meta):
    if model["type"] == "cnn":
        import flicker_kpi_cnn as kcnn
        return kcnn.save(path, model["net"], meta)
    return save_model(path, model, meta)


def load_any(path):
    if str(path).lower().endswith(".npz"):
        import flicker_kpi_cnn as kcnn
        net, meta = kcnn.load(path)
        return {"type": "cnn", "net": net}, meta
    return load_model(path)


def default_ext(kind):
    return {"logreg": ".json", "gbdt": ".joblib", "cnn": ".npz"}.get(kind, ".json")


def score_result(model, result, tile, y_mode=DEFAULT_Y_MODE, meta=None):
    """對一組已分析的序列輸出每格分數 (th, tw)"""
    if model["type"] == "cnn":
        import flicker_kpi_cnn as kcnn
        content = bool((meta or {}).get("content", False))   # 由模型檔決定輸入通道
        sc, _ = kcnn.score_tiles(model["net"], result["paths"], tile, content)
        return sc
    sc, _ = score_map(model, result, tile, y_mode)
    return sc

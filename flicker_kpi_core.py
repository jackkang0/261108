"""
Flicker KPI 核心引擎
====================
規格（依 KPI 投影片）：
  1 張圖片 = 6 個 frames，8 bits dither +1

  Pixel-based:
      R/G/B/Y_difference, L1_difference
      R/G/B/Y_variance,   L1_variance      (6 張 frame 計算，pixel 沒變化 -> var = 0)

  Block-based (2x2):
      mean_R/G/B/Y_difference, mean_L1_difference
      R/G/B/Y_variance,        L1_variance
      -> 先對每個 frame 做 2x2 mean pooling，再沿 frame 軸算 difference / variance

定義：
  Y  可選: BT.601  = 0.299R + 0.587G + 0.114B   (預設)
           (R+2G+B)/4 = 0.25R + 0.5G + 0.25B      (與 BT.601 同量級，threshold 相近)
           R+2G+B     = 1.0R + 2.0G + 1.0B        (為前者的 4 倍，Y_* threshold 需 x4;
                                                   variance 需 x16)
  L1 = R + G + B                     (三通道 L1 norm，pixel 值皆為非負)

  difference = max(frames) - min(frames)        # peak-to-peak，抓整段序列最大擺幅
               (可切換為 max adjacent |delta|)
  variance   = sigma((xi - mean)^2) / N         # 母體變異數，N=6，與投影片左下角公式一致

KPI 判定：
  某位置的某項 metric > 該項 threshold  ->  該位置該項 fail
  KPI = fail 位置數 / 總位置數  (百分比)
"""

import os
import re
import glob
import json

import numpy as np
import cv2


# --------------------------------------------------------------------------
# 預設 threshold（來自 KPI 投影片 Default 欄）
# --------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    "pixel": {
        "R_difference": 133.0,
        "G_difference": 75.0,
        "B_difference": 140.0,
        "Y_difference": 75.0,
        "L1_difference": 220.0,
        "R_variance": 2500.0,
        "G_variance": 1150.0,
        "B_variance": 1750.0,
        "Y_variance": 1200.0,
        "L1_variance": 9000.0,
    },
    "block": {
        "R_difference": 43.0,
        "G_difference": 30.0,
        "B_difference": 40.0,
        "Y_difference": 25.0,
        "L1_difference": 70.0,
        "R_variance": 240.0,
        "G_variance": 200.0,
        "B_variance": 360.0,
        "Y_variance": 115.0,
        "L1_variance": 1200.0,
    },
}

CHANNELS = ["R", "G", "B", "Y", "L1"]

# Y 的定義（權重 R, G, B）
Y_MODES = {
    "BT.601": (0.299, 0.587, 0.114),
    "(R+2G+B)/4": (0.25, 0.5, 0.25),
    "R+2G+B": (1.0, 2.0, 1.0),
}
DEFAULT_Y_MODE = "BT.601"


def k_for_level(K, level, block_size=2):
    """
    序列判定規則「某 metric 超標位置數 >= K」的 K 以 pixel 計；
    block-based 換算為 ceil(K / block_size^2) 個 block（面積一致）。
    """
    import math
    return int(K) if level == "pixel" else max(1, math.ceil(K / block_size ** 2))


def metric_is_fail(stat, level, min_fail=1, block_size=2):
    return stat["fail_count"] >= k_for_level(min_fail, level, block_size)


def all_enabled():
    """每項 metric 的啟用狀態（預設全開）；停用的 metric 不列入判定、風險圖與分格"""
    return {lv: {m: True for m in METRIC_NAMES} for lv in ("pixel", "block")}


def is_enabled(enabled, level, metric):
    if enabled is None:
        return True
    return bool(enabled.get(level, {}).get(metric, True))
METRIC_NAMES = [f"{c}_{k}" for k in ("difference", "variance") for c in CHANNELS]


# --------------------------------------------------------------------------
# 讀檔
# --------------------------------------------------------------------------
def imread_unicode(path):
    """cv2.imread 在 Windows 不支援非 ASCII 路徑，改用 fromfile + imdecode。回傳 BGR uint8"""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"無法讀取: {path}")
    return img


def load_frames(paths, progress=None):
    """依給定順序讀入 frames，回傳 (N, H, W, 3) float32 RGB

    progress: 可選的 callback(done, total, text)，逐幀回報讀取進度
    """
    frames = []
    for i, p in enumerate(paths):
        if progress:
            progress(i, len(paths), f"讀圖 {i+1}/{len(paths)}")
        frames.append(imread_unicode(p)[:, :, ::-1])
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise ValueError(f"序列中幀尺寸不一致: {shapes}")
    return np.stack(frames, axis=0).astype(np.float32)


def load_sequence(folder, pattern="*.bmp"):
    """
    讀入資料夾內所有符合 pattern 的檔案為一組序列（舊介面，CLI 使用）。
    回傳 stack: (N, H, W, 3) float32 RGB, paths
    """
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise IOError(f"{folder} 找不到符合 {pattern} 的檔案")
    return load_frames(paths), paths


_FRAME_RE = re.compile(r"^(.*?)(\d{4})$")   # 檔名(不含副檔名) = <prefix><4 位 frame 編號>


def discover_sequences(folder, pattern="*.bmp", n_frames=6):
    """
    掃描單一資料夾，把 <prefix>0001 ~ <prefix>000N 的檔案依 prefix 分組成多組序列。
    例: sceneA_0001.bmp ~ sceneA_0006.bmp, sceneB_0001.bmp ~ sceneB_0006.bmp -> 2 組

    回傳 (seqs, skipped)
      seqs    : [{"name", "prefix", "paths"(依 0001..000N 排序), "extra"(超出 N 的編號)}]
      skipped : [(prefix, 原因)]  —— 缺幀的 prefix 不分析
    """
    groups = {}
    for p in sorted(glob.glob(os.path.join(folder, pattern))):
        stem = os.path.splitext(os.path.basename(p))[0]
        m = _FRAME_RE.match(stem)
        if not m:
            continue
        groups.setdefault(m.group(1), {})[int(m.group(2))] = p

    need = list(range(1, n_frames + 1))
    seqs, skipped, used = [], [], set()
    for prefix in sorted(groups):
        fr = groups[prefix]
        missing = [i for i in need if i not in fr]
        if missing:
            skipped.append((prefix, "缺 frame " + ",".join(f"{i:04d}" for i in missing)))
            continue
        name = prefix.rstrip("_-. ") or os.path.basename(os.path.normpath(folder))
        if name in used:                       # 例如 "a" 與 "a_" 同時存在
            name = prefix
        used.add(name)
        seqs.append({"name": name, "prefix": prefix,
                     "paths": [fr[i] for i in need],
                     "extra": sorted(set(fr) - set(need))})
    return seqs, skipped


# --------------------------------------------------------------------------
# 通道推導
# --------------------------------------------------------------------------
def derive_channels(stack, y_mode=DEFAULT_Y_MODE):
    """
    stack: (N, H, W, 3) RGB float32
    y_mode: Y_MODES 的鍵，決定 Y 的權重
    回傳 dict，每項為 (N, H, W) float32
    """
    try:
        wr, wg, wb = Y_MODES[y_mode]
    except KeyError:
        raise ValueError(f"未知的 Y 定義: {y_mode}（可用: {', '.join(Y_MODES)}）") from None
    R = stack[..., 0]
    G = stack[..., 1]
    B = stack[..., 2]
    Y = wr * R + wg * G + wb * B
    L1 = R + G + B
    return {"R": R, "G": G, "B": B, "Y": Y, "L1": L1}


def block_mean_pool(arr, block=2):
    """
    arr: (N, H, W) -> 每個 frame 做 block x block 平均 -> (N, H//block, W//block)
    邊緣不足一個 block 的部分直接裁掉（保證每個 block 統計量的樣本數一致）。
    """
    N, H, W = arr.shape
    h, w = H // block, W // block
    cropped = arr[:, :h * block, :w * block]
    return cropped.reshape(N, h, block, w, block).mean(axis=(2, 4))


# --------------------------------------------------------------------------
# metric 計算
# --------------------------------------------------------------------------
def frame_difference(arr, mode="peak_to_peak"):
    """
    arr: (N, H, W)
    peak_to_peak : max - min，抓整段序列最大擺幅（預設，對應 KPI 的 difference）
    max_adjacent : max(|frame[i+1] - frame[i]|)，只抓相鄰幀跳動（閃爍感較直接相關）
    """
    if mode == "peak_to_peak":
        return arr.max(axis=0) - arr.min(axis=0)
    if mode == "max_adjacent":
        return np.abs(np.diff(arr, axis=0)).max(axis=0)
    raise ValueError(f"未知 difference mode: {mode}")


def frame_variance(arr):
    """Var = sigma((xi - mean)^2) / N，母體變異數；pixel 完全沒變化 -> 0"""
    return arr.var(axis=0)


def compute_metric_maps(stack, block=1, diff_mode="peak_to_peak",
                        y_mode=DEFAULT_Y_MODE):
    """
    計算一組序列的 10 張 metric map。
    block=1 -> pixel-based；block=2 -> 2x2 block-based。
    回傳 {metric_name: 2D float32 array}
    """
    ch = derive_channels(stack, y_mode)
    if block > 1:
        ch = {k: block_mean_pool(v, block) for k, v in ch.items()}

    maps = {}
    for name, arr in ch.items():
        maps[f"{name}_difference"] = frame_difference(arr, diff_mode)
        maps[f"{name}_variance"] = frame_variance(arr)
    return maps


# --------------------------------------------------------------------------
# KPI 統計
# --------------------------------------------------------------------------
def evaluate_kpi(metric_maps, thresholds):
    """
    對每個 metric 計算超標比例與分布統計。
    回傳 {metric_name: {...stats...}}
    """
    report = {}
    for name, m in metric_maps.items():
        thr = float(thresholds[name])
        fail = m > thr
        total = fail.size
        n_fail = int(fail.sum())
        report[name] = {
            "threshold": thr,
            "fail_count": n_fail,
            "total": total,
            "fail_ratio": n_fail / total,          # 超標比例（KPI 主值）
            "max": float(m.max()),
            "mean": float(m.mean()),
            "p99": float(np.percentile(m, 99)),
            "p999": float(np.percentile(m, 99.9)),
            "margin": float(m.max()) / thr if thr > 0 else float("inf"),  # 峰值 / threshold
            "fail_mask": fail,
        }
    return report


def combined_fail_map(report, selected=None):
    """
    綜合圖：每個位置有幾項 metric 超標（0 ~ len(selected)）。
    可用來找「同時違反多項指標」的重災區。
    """
    names = selected if selected else list(report.keys())
    acc = None
    for n in names:
        f = report[n]["fail_mask"].astype(np.int32)
        acc = f if acc is None else acc + f
    return acc


def normalized_map(metric_map, threshold):
    """value / threshold，1.0 就是及格線。方便用同一色階比較不同量級的 metric。"""
    if threshold <= 0:
        return np.zeros_like(metric_map)
    return metric_map / threshold


# --------------------------------------------------------------------------
# 視覺化（不依賴 GUI，可單獨輸出檔案）
# --------------------------------------------------------------------------
def render_heatmap(metric_map, threshold, mode="ratio", vmax_ratio=2.0):
    """
    回傳 BGR uint8 heatmap，可直接 cv2.imwrite。

    mode:
      "ratio"  - 以 value/threshold 上色，色階固定 0 ~ vmax_ratio，
                 1.0 (及格線) 落在色階中段，跨圖可比較
      "raw"    - 以 metric 自身最大值正規化
      "binary" - 只顯示超標 / 未超標
    """
    if mode == "binary":
        mask = (metric_map > threshold).astype(np.uint8) * 255
        vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
        vis[..., 2] = mask                      # 紅色 = fail
        return vis

    if mode == "ratio":
        norm = np.clip(normalized_map(metric_map, threshold) / vmax_ratio, 0, 1)
    else:
        mx = metric_map.max()
        norm = metric_map / mx if mx > 0 else np.zeros_like(metric_map)

    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)


def upscale(img, factor):
    """block-based 的圖比較小，放大回原尺寸方便對照（nearest 保持 block 邊界銳利）"""
    if factor <= 1:
        return img
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------
# 風險分級總圖：20 項 metric 合併成單張，疊在原圖上
# --------------------------------------------------------------------------
# 分界為 value/threshold；1.0 = 及格線（metric > threshold 才算 fail）
DEFAULT_RISK_BOUNDS = (0.5, 0.8, 1.0, 2.0)
RISK_LEVELS = [              # (label, BGR)
    ("SAFE",   (80, 175, 0)),    # 綠  ratio <= b0
    ("WATCH",  (0, 215, 255)),   # 黃  b0 < ratio <= b1
    ("NEAR",   (0, 130, 255)),   # 橘  b1 < ratio <= b2  (接近門檻，仍 PASS)
    ("FAIL",   (0, 0, 230)),     # 紅  b2 < ratio <= b3
    ("SEVERE", (200, 0, 200)),   # 紫  ratio > b3
]


def _to_full_res(m, block, H, W):
    """block map 以 nearest 還原到原解析度；被裁掉的邊緣補 0（由 pixel-based 覆蓋）"""
    if block > 1:
        m = np.repeat(np.repeat(m, block, axis=0), block, axis=1)
    out = np.zeros((H, W), np.float32)
    h, w = min(H, m.shape[0]), min(W, m.shape[1])
    out[:h, :w] = m[:h, :w]
    return out


def parse_risk_bounds(text):
    b = tuple(float(x) for x in str(text).replace(" ", "").split(","))
    if len(b) != len(RISK_LEVELS) - 1 or any(v <= 0 for v in b) \
            or any(b[i] >= b[i + 1] for i in range(len(b) - 1)):
        raise ValueError(f"風險分界需為 {len(RISK_LEVELS)-1} 個遞增正數，例如 0.5,0.8,1.0,2.0")
    return b


def compute_risk(result, bounds=DEFAULT_RISK_BOUNDS, levels=("pixel", "block"),
                 enabled=None):
    """
    每個原圖 pixel 的 risk ratio = max over (pixel 10 項 + block 10 項) of value/threshold。
    block metric 映射回其覆蓋的 2x2 pixel。
    回傳 {
      ratio : (H,W) float32   最大 value/threshold
      best  : (H,W) int16     造成最大值的 metric index（對應 names）
      names : ["pixel:R_difference", ...]
      level : (H,W) uint8     0~4 風險等級
      frac  : [5]             各等級面積比例
    }
    """
    H, W = result["shape"]
    names, ratio, best = [], None, None
    zero = np.zeros((H, W), np.float32)
    for lv in levels:
        blk = 1 if lv == "pixel" else result.get("block_size", 2)
        rep, maps = result[lv]["report"], result[lv]["maps"]
        for n in METRIC_NAMES:
            thr = rep[n]["threshold"]
            if thr <= 0 or not is_enabled(enabled, lv, n):
                continue
            r = _to_full_res(maps[n] / thr, blk, H, W)
            idx = len(names)
            names.append(f"{lv}:{n}")
            if ratio is None:
                ratio, best = r, np.zeros((H, W), np.int16)
            else:
                upd = r > ratio
                ratio = np.where(upd, r, ratio)
                best[upd] = idx
    if ratio is None:                       # 全部 metric 都停用
        ratio, best, names = zero, np.zeros((H, W), np.int16), ["(none)"]
    level = np.searchsorted(np.asarray(bounds, np.float32), ratio,
                            side="left").astype(np.uint8)
    frac = np.bincount(level.ravel(), minlength=len(RISK_LEVELS)) / level.size
    return {"ratio": ratio, "best": best, "names": names, "level": level,
            "frac": frac, "bounds": tuple(bounds)}


def summarize_result(result, risk, tiles=None, min_fail=1, enabled=None):
    """
    單組序列的一行摘要（批次表格 / summary CSV 用）；給 tiles 時附上分格統計。
    verdict: 任一 metric 超標位置數 >= K (min_fail，block 以面積換算) 即 FAIL。
    """
    bs = result.get("block_size", 2)
    pf = [n for n in METRIC_NAMES
          if is_enabled(enabled, "pixel", n)
          and metric_is_fail(result["pixel"]["report"][n], "pixel", min_fail, bs)]
    bf = [n for n in METRIC_NAMES
          if is_enabled(enabled, "block", n)
          and metric_is_fail(result["block"]["report"][n], "block", min_fail, bs)]
    flat = int(np.argmax(risk["ratio"]))
    y, x = divmod(flat, risk["ratio"].shape[1])
    worst = int(risk["level"].max())
    return {
        "name": result["name"],
        "verdict": "FAIL" if (pf or bf) else "PASS",
        "pixel_fail_metrics": len(pf),
        "block_fail_metrics": len(bf),
        "danger_pct": float(risk["frac"][3:].sum() * 100),   # FAIL + SEVERE 面積
        "worst_level": RISK_LEVELS[worst][0],
        "max_ratio": float(risk["ratio"][y, x]),
        "max_metric": risk["names"][int(risk["best"][y, x])],
        "max_xy": (int(x), int(y)),
        **({"tiles_red": tiles["counts"]["RED"],
            "tiles_orange": tiles["counts"]["ORANGE"],
            "tiles_yellow": tiles["counts"]["YELLOW"],
            "tiles_fail": tiles["n_fail"],
            "tiles_total": tiles["n_tiles"],
            "tile_fail_pct": tiles["n_fail"] / tiles["n_tiles"] * 100,
            "tile_size": tiles["tile"]} if tiles is not None else {}),
    }


def risk_range_labels(bounds):
    b = bounds
    return ([f"<= {b[0]:g}x"] +
            [f"{b[i]:g}-{b[i+1]:g}x" for i in range(len(b) - 1)] +
            [f"> {b[-1]:g}x"])


LEGEND_ROW_H, LEGEND_PAD, LEGEND_MIN_W = 24, 8, 460
LEGEND_H = LEGEND_PAD * 2 + LEGEND_ROW_H * (len(RISK_LEVELS) + 1)


def max_pool_fit(arr, max_w, max_h):
    """
    縮小顯示用：以整數倍 max-pool 取代 nearest 縮圖，
    保證 1x1 的超標點在縮圖中不會被跳過（nearest 會直接丟掉非取樣點）。
    arr 需為非負值（level / metric map）。回傳 (pooled, factor)
    """
    H, W = arr.shape[:2]
    s = int(np.ceil(max(W / max(max_w, 1), H / max(max_h, 1), 1.0)))
    if s == 1:
        return arr, 1
    h, w = -(-H // s), -(-W // s)
    pad = np.zeros((h * s, w * s) + arr.shape[2:], arr.dtype)
    pad[:H, :W] = arr
    return pad.reshape(h, s, w, s, *arr.shape[2:]).max(axis=(1, 3)), s


def render_risk_map(result, risk, alpha=0.5, bg_frame=0, min_width=640, fit=None):
    """
    單張輸出：原圖（灰階）疊 5 色風險等級 + 下方圖例（含各等級面積 %）。
    fit=(max_w, max_h)：螢幕顯示用，等級圖以 max-pool 縮小（小閃爍點不會消失）；
    fit=None：原解析度輸出（匯出檔用）。
    回傳 (BGR uint8, scale)；scale = 輸出圖 pixel / 原圖 pixel
    """
    rgb = result["stack"][bg_frame]
    gray = np.clip(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2],
                   0, 255).astype(np.uint8)
    lv = risk["level"]
    scale = 1.0
    if fit is not None:
        lv, f = max_pool_fit(lv, fit[0], fit[1] - LEGEND_H)
        scale = 1.0 / f
        if f > 1:
            gray = cv2.resize(gray, (lv.shape[1], lv.shape[0]), interpolation=cv2.INTER_AREA)
    bg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)

    lut = np.array([c for _, c in RISK_LEVELS], np.float32)
    # 疊色強度隨風險遞增：安全區淡、危險區濃，alpha 為 WATCH 的基準
    a_lut = np.array([0.4, 1.0, 1.1, 1.4, 1.4], np.float32) * alpha
    a = np.clip(a_lut[lv], 0, 0.85)[..., None]
    img = (bg * (1 - a) + lut[lv] * a).clip(0, 255).astype(np.uint8)

    if fit is None:
        up = max(1, int(np.ceil(min_width / img.shape[1])))
        img = upscale(img, up)
        scale = float(up)
    if img.shape[1] < LEGEND_MIN_W:                      # 圖例文字需要最小寬度
        padw = LEGEND_MIN_W - img.shape[1]
        img = cv2.copyMakeBorder(img, 0, 0, padw // 2, padw - padw // 2,
                                 cv2.BORDER_CONSTANT, value=(32, 32, 32))
    W = img.shape[1]

    # 圖例
    row_h, pad = LEGEND_ROW_H, LEGEND_PAD
    legend = np.full((LEGEND_H, W, 3), 255, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(legend, "risk = max(value / threshold) over pixel+block metrics",
                (pad, pad + 16), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    for i, ((label, color), rng) in enumerate(zip(RISK_LEVELS,
                                                   risk_range_labels(risk["bounds"]))):
        y = pad + row_h * (i + 1)
        cv2.rectangle(legend, (pad, y + 4), (pad + 28, y + row_h - 4), color, -1)
        cv2.putText(legend, f"{label:<7}{rng:<12}{risk['frac'][i]*100:8.3f}%",
                    (pad + 38, y + 17), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([img, legend]), scale


def imwrite_unicode(path, img):
    """cv2.imwrite 在 Windows 不支援非 ASCII 路徑，改用 imencode + tofile"""
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if not ok:
        raise IOError(f"編碼失敗: {path}")
    buf.tofile(path)


# --------------------------------------------------------------------------
# 分格總圖：把 pixel 級的超標點彙整到 tile x tile 格子，供人工複查
# --------------------------------------------------------------------------
DEFAULT_TILE = 40
# 每一級的升級條件 (max_ratio, pixel 超標數, block 超標數)，任一條件成立即升級
DEFAULT_TILE_RULES = {"RED": (2.0, 16, 4), "ORANGE": (1.5, 4, 1)}
TILE_LEVELS = [          # (label, BGR)；index 0 = 無超標，不畫
    ("NONE",   None),
    ("YELLOW", (0, 220, 255)),
    ("ORANGE", (0, 140, 255)),
    ("RED",    (0, 0, 255)),
]


def parse_tile_size(text, shape=None, block_size=2):
    t = int(str(text).strip())
    if t < block_size or t % block_size:
        raise ValueError(f"分格大小需為 block 尺寸 ({block_size}) 的倍數且 >= {block_size}")
    return t


def parse_tile_rule(text):
    v = [float(x) for x in str(text).replace(" ", "").split(",")]
    if len(v) != 3 or v[0] <= 0 or v[1] < 0 or v[2] < 0:
        raise ValueError("條件格式: max_ratio,pixel超標數,block超標數  例如 2,16,4")
    return (v[0], v[1], v[2])


def _tile_reduce(arr, t, op):
    """(H,W) -> (ceil(H/t), ceil(W/t))，邊緣不足一格者補 0 後照算"""
    H, W = arr.shape
    th, tw = -(-H // t), -(-W // t)
    pad = np.zeros((th * t, tw * t), arr.dtype)
    pad[:H, :W] = arr
    return op(pad.reshape(th, t, tw, t), axis=(1, 3))


def compute_tiles(result, risk, tile=DEFAULT_TILE, rules=DEFAULT_TILE_RULES):
    """
    每格統計：
      max_ratio   : 格內 max(value/threshold)，pixel 10 項 + block 10 項（= risk ratio）
      max_x/max_y : 最大值所在 pixel；max_metric: 造成最大值的 metric index（risk["names"]）
      pixel_fail  : pixel-based 任一 metric 超標的 pixel 數
      block_fail  : block-based 任一 metric 超標的 2x2 block 數（block 以左上角 pixel 歸格）
      main_metric : 格內超標最多的 metric index（block 的超標數 x block 面積換算成 pixel 數比較）
      level       : 0 無超標 / 1 YELLOW / 2 ORANGE / 3 RED
    """
    H, W = result["shape"]
    bs = result.get("block_size", 2)
    if tile % bs:
        raise ValueError(f"tile ({tile}) 需為 block 尺寸 ({bs}) 的倍數")
    tb = tile // bs

    ratio = risk["ratio"]
    max_ratio = _tile_reduce(ratio, tile, np.max)

    # 最大值位置：以 tile 內 argmax 求得
    th, tw = max_ratio.shape
    pad = np.zeros((th * tile, tw * tile), np.float32)
    pad[:H, :W] = ratio
    blk = pad.reshape(th, tile, tw, tile).transpose(0, 2, 1, 3).reshape(th, tw, tile * tile)
    am = blk.argmax(axis=2)
    ry, rx = np.divmod(am, tile)
    max_y = np.arange(th)[:, None] * tile + ry
    max_x = np.arange(tw)[None, :] * tile + rx
    best = np.zeros((th * tile, tw * tile), np.int16)
    best[:H, :W] = risk["best"]
    max_metric = best[np.clip(max_y, 0, th * tile - 1), np.clip(max_x, 0, tw * tile - 1)]

    names = risk["names"]
    pix_any = np.zeros((H, W), bool)
    blk_any = None
    per_metric = []
    for nm in names:
        if ":" not in nm:                      # 全部 metric 停用 -> 沒有可統計的項目
            continue
        lv, m = nm.split(":")
        f = result[lv]["report"][m]["fail_mask"]
        if lv == "pixel":
            pix_any |= f
            per_metric.append(_tile_reduce(f.astype(np.int32), tile, np.sum))
        else:
            blk_any = f.copy() if blk_any is None else (blk_any | f)
            c = _tile_reduce(f.astype(np.int32), tb, np.sum)[:th, :tw]
            per_metric.append(c * bs * bs)
    pixel_fail = _tile_reduce(pix_any.astype(np.int32), tile, np.sum)
    block_fail = (_tile_reduce(blk_any.astype(np.int32), tb, np.sum)[:th, :tw]
                  if blk_any is not None else np.zeros_like(pixel_fail))
    # block map 比 H/bs 小時（邊緣被裁），補 0 讓 shape 一致
    if block_fail.shape != pixel_fail.shape:
        bf = np.zeros_like(pixel_fail)
        bf[:block_fail.shape[0], :block_fail.shape[1]] = block_fail
        block_fail = bf
        per_metric = [pm if pm.shape == pixel_fail.shape else
                      np.pad(pm, ((0, th - pm.shape[0]), (0, tw - pm.shape[1])))
                      for pm in per_metric]
    main_metric = (np.stack(per_metric, 0).argmax(axis=0) if per_metric
                   else np.zeros_like(pixel_fail))

    fail = max_ratio > 1.0
    def hit(rule):
        r, px, bk = rule
        return fail & ((max_ratio >= r) | (pixel_fail >= px) | (block_fail >= bk))
    level = np.zeros((th, tw), np.uint8)
    level[fail] = 1
    level[hit(rules["ORANGE"])] = 2
    level[hit(rules["RED"])] = 3

    counts = {lab: int((level == i).sum()) for i, (lab, _) in enumerate(TILE_LEVELS) if i}
    return {"tile": tile, "shape": (H, W), "level": level, "max_ratio": max_ratio,
            "max_x": max_x, "max_y": max_y, "max_metric": max_metric,
            "pixel_fail": pixel_fail, "block_fail": block_fail,
            "main_metric": main_metric, "names": names, "rules": dict(rules),
            "counts": counts, "n_fail": int(fail.sum()), "n_tiles": th * tw}


def tile_detail(tiles, row, col):
    """單一格的明細 dict（GUI 點選用）"""
    t, (H, W) = tiles["tile"], tiles["shape"]
    n = tiles["names"]
    row, col = int(row), int(col)
    return {"row": row, "col": col,
            "x0": col * t, "y0": row * t,
            "x1": min((col + 1) * t, W), "y1": min((row + 1) * t, H),
            "level": TILE_LEVELS[int(tiles["level"][row, col])][0],
            "max_ratio": float(tiles["max_ratio"][row, col]),
            "max_metric": n[int(tiles["max_metric"][row, col])],
            "max_x": int(tiles["max_x"][row, col]), "max_y": int(tiles["max_y"][row, col]),
            "pixel_fail": int(tiles["pixel_fail"][row, col]),
            "block_fail": int(tiles["block_fail"][row, col]),
            "main_metric": n[int(tiles["main_metric"][row, col])]}


def tile_rule_text(rule):
    r, px, bk = rule
    return f"ratio>={r:g} | px>={px:g} | blk>={bk:g}"


def render_tile_map(result, tiles, bg_frame=0, fit=None, min_width=640, thickness=2):
    """
    分格總圖：灰階原圖 + RED/ORANGE/YELLOW 格子外框（無文字）+ 圖例。
    fit=(max_w, max_h) 為螢幕顯示；None 為原解析度輸出。
    回傳 (BGR uint8, scale)；scale = 輸出圖 pixel / 原圖 pixel（GUI 換算點擊座標用）
    """
    H, W = tiles["shape"]
    rgb = result["stack"][bg_frame]
    gray = np.clip(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2],
                   0, 255).astype(np.uint8)
    if fit is not None:
        sc = min(fit[0] / W, (fit[1] - LEGEND_H) / H, 1.0)
    else:
        sc = float(max(1, int(np.ceil(min_width / W))))
    dw, dh = max(1, int(round(W * sc))), max(1, int(round(H * sc)))
    interp = cv2.INTER_AREA if sc < 1 else cv2.INTER_NEAREST
    img = cv2.cvtColor(cv2.resize(gray, (dw, dh), interpolation=interp),
                       cv2.COLOR_GRAY2BGR)
    img = (img.astype(np.float32) * 0.75).astype(np.uint8)   # 背景壓暗，突顯外框

    t, lv = tiles["tile"], tiles["level"]
    for li in (1, 2, 3):                                     # 高等級後畫，共用邊以高等級為準
        color = TILE_LEVELS[li][1]
        for r, c in zip(*np.nonzero(lv == li)):
            x0, y0 = int(round(c * t * sc)), int(round(r * t * sc))
            x1 = int(round(min((c + 1) * t, W) * sc)) - 1
            y1 = int(round(min((r + 1) * t, H) * sc)) - 1
            cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)

    if img.shape[1] < LEGEND_MIN_W:
        padw = LEGEND_MIN_W - img.shape[1]
        img = cv2.copyMakeBorder(img, 0, 0, 0, padw, cv2.BORDER_CONSTANT, value=(32, 32, 32))
    Wd = img.shape[1]
    row_h, pad = LEGEND_ROW_H, LEGEND_PAD
    legend = np.full((LEGEND_H, Wd, 3), 255, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    th, tw = lv.shape
    cv2.putText(legend, f"tile {t}x{t}: fail tiles {tiles['n_fail']}/{tiles['n_tiles']}"
                        f" ({tiles['n_fail']/tiles['n_tiles']*100:.2f}%)",
                (pad, pad + 16), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    rows = [("RED", tile_rule_text(tiles["rules"]["RED"])),
            ("ORANGE", tile_rule_text(tiles["rules"]["ORANGE"])),
            ("YELLOW", "other fail tiles (ratio > 1)")]
    lut = dict(TILE_LEVELS[1:])
    for i, (lab, txt) in enumerate(rows):
        y = pad + row_h * (i + 1)
        cv2.rectangle(legend, (pad, y + 4), (pad + 28, y + row_h - 4), lut[lab], 2)
        cv2.putText(legend, f"{lab:<7}{tiles['counts'][lab]:>5}   {txt}",
                    (pad + 38, y + 17), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([img, legend]), sc


# --------------------------------------------------------------------------
# ROI（人工框出「這裡會閃」的位置）存取
# --------------------------------------------------------------------------
ROI_FILE = "flicker_rois.csv"
ROI_HEADER = ["sequence", "x0", "y0", "x1", "y1"]


def load_rois(folder):
    """回傳 {sequence: [(x0, y0, x1, y1), ...]}；x1/y1 為不含端點（half-open）"""
    path = os.path.join(folder, ROI_FILE)
    out = {}
    if not os.path.isfile(path):
        return out
    import csv as _csv
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in _csv.DictReader(f):
            try:
                box = tuple(int(float(row[k])) for k in ("x0", "y0", "x1", "y1"))
                name = row["sequence"]
            except (KeyError, TypeError, ValueError):
                continue
            if box[2] > box[0] and box[3] > box[1]:
                out.setdefault(name, []).append(box)
    return out


def save_rois(folder, rois):
    """rois: {sequence: [(x0, y0, x1, y1), ...]}；空的序列不寫入"""
    import csv as _csv
    path = os.path.join(folder, ROI_FILE)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = _csv.writer(f)
        wr.writerow(ROI_HEADER)
        for name in sorted(rois):
            for b in rois[name]:
                wr.writerow([name, *b])
    return path


def clip_box(box, shape, min_size=1):
    """(x0,y0,x1,y1) 正規化到影像範圍內；太小回傳 None"""
    H, W = shape
    x0, x1 = sorted((int(box[0]), int(box[2])))
    y0, y1 = sorted((int(box[1]), int(box[3])))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < min_size or y1 - y0 < min_size:
        return None
    return (x0, y0, x1, y1)


def box_metric_stats(result, box, enabled=None, min_fail=1):
    """
    單一框內，20 項 metric 各自的統計。
    回傳 [{level, metric, max, mean, thr, ratio, count, total, enabled, fail}]
      count : 框內超過門檻的位置數（block 層級以 block 計）
      ratio : 框內最大值 / threshold
      fail  : 框內超標數 >= K（K 取 min(K, 框內位置數)）
    """
    x0, y0, x1, y1 = box
    bs = result.get("block_size", 2)
    out = []
    for lv, blk in (("pixel", 1), ("block", bs)):
        rep, maps = result[lv]["report"], result[lv]["maps"]
        if blk > 1:
            bx0, by0 = x0 // blk, y0 // blk
            bx1, by1 = -(-x1 // blk), -(-y1 // blk)
        else:
            bx0, by0, bx1, by1 = x0, y0, x1, y1
        for m in METRIC_NAMES:
            arr = maps[m]
            sub = arr[by0:min(by1, arr.shape[0]), bx0:min(bx1, arr.shape[1])]
            thr = float(rep[m]["threshold"])
            if sub.size == 0:
                out.append({"level": lv, "metric": m, "max": 0.0, "mean": 0.0,
                            "thr": thr, "ratio": 0.0, "count": 0, "total": 0,
                            "enabled": is_enabled(enabled, lv, m), "fail": False})
                continue
            mx = float(sub.max())
            cnt = int((sub > thr).sum())
            k = min(k_for_level(min_fail, lv, bs), sub.size)
            out.append({"level": lv, "metric": m, "max": mx, "mean": float(sub.mean()),
                        "thr": thr, "ratio": mx / thr if thr > 0 else float("inf"),
                        "count": cnt, "total": int(sub.size),
                        "enabled": is_enabled(enabled, lv, m), "fail": cnt >= k})
    return out


def tile_kth(arr, tile, k):
    """(H,W) -> (th, tw) 每格第 k 大的值；不足一格的邊緣捨去"""
    H, W = arr.shape
    th, tw = H // tile, W // tile
    if th == 0 or tw == 0:
        return np.zeros((0, 0), np.float32)
    sub = arr[:th * tile, :tw * tile].reshape(th, tile, tw, tile)
    flat = sub.transpose(0, 2, 1, 3).reshape(th * tw, tile * tile)
    kk = min(int(k), flat.shape[1])
    part = np.partition(flat, flat.shape[1] - kk, axis=1)[:, flat.shape[1] - kk:]
    return part.min(axis=1).astype(np.float32).reshape(th, tw)


def tile_flags(result, tile=120, k_box=1, enabled=None):
    """
    傳統門檻法的逐格判定：(th, tw) bool，True = 該格被判成閃。
    規則與框相同：格內有 >= K_box 個位置超過門檻（任一啟用的 metric）即為閃。
    """
    H, W = result["shape"]
    bs = result.get("block_size", 2)
    th, tw = H // tile, W // tile
    flag = np.zeros((th, tw), bool) if th and tw else np.zeros((0, 0), bool)
    for lv, blk in (("pixel", 1), ("block", bs)):
        rep, maps = result[lv]["report"], result[lv]["maps"]
        t_lv = max(1, tile // blk)
        k = k_for_level(k_box, lv, bs)
        for m in METRIC_NAMES:
            if not is_enabled(enabled, lv, m):
                continue
            thr = float(rep[m]["threshold"])
            km = tile_kth(maps[m], t_lv, k)
            if km.size:
                flag[:km.shape[0], :km.shape[1]] |= km > np.float32(thr)
    return flag


def roi_summary(result, boxes, tile=120, k_box=1, enabled=None):
    """
    以目前門檻評估人工框：
      n_box / n_caught / miss   框的總數、抓到數、沒抓到的框
      n_tile / n_fp / fp_tiles  與框不重疊的格子總數、被判成閃的格數與位置 (row, col)
    判定規則與框相同：格內有 >= K_box 個位置超過門檻即視為該格被判成閃。
    """
    H, W = result["shape"]
    bs = result.get("block_size", 2)
    boxes = list(boxes or ())

    caught, miss = [], []
    for b in boxes:
        rows = box_metric_stats(result, b, enabled, k_box)
        hit = any(r["enabled"] and r["fail"] for r in rows)
        (caught if hit else miss).append(b)

    th, tw = H // tile, W // tile
    flag = tile_flags(result, tile, k_box, enabled)      # 與三法疊圖共用同一套判定
    for (x0, y0, x1, y1) in boxes:                       # 與框重疊的格不算框外
        flag[max(0, y0 // tile):min(th, -(-y1 // tile)),
             max(0, x0 // tile):min(tw, -(-x1 // tile))] = False
    fp = [(int(r), int(c)) for r, c in zip(*np.nonzero(flag))]
    n_tile = th * tw - sum(
        (min(th, -(-y1 // tile)) - max(0, y0 // tile)) *
        (min(tw, -(-x1 // tile)) - max(0, x0 // tile)) for (x0, y0, x1, y1) in boxes) \
        if boxes else th * tw
    return {"tile": tile, "k_box": k_box, "n_box": len(boxes), "n_caught": len(caught),
            "miss": miss, "n_tile": max(n_tile, 0), "n_fp": len(fp), "fp_tiles": fp}


# --------------------------------------------------------------------------
# 設定檔（threshold 等參數）存取
# --------------------------------------------------------------------------
SETTINGS_VERSION = 1
SETTINGS_FILE = "flicker_kpi_settings.json"


def default_settings_path():
    """預設設定檔：與程式同目錄的 flicker_kpi_settings.json"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), SETTINGS_FILE)


def save_settings(path, settings):
    data = dict(settings)
    data["version"] = SETTINGS_VERSION
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def load_settings(path):
    """回傳 (settings dict, warnings list)。缺少或格式錯誤的項目會被忽略並列入 warnings。"""
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"不是有效的 JSON: {e}") from None
    if not isinstance(data, dict):
        raise ValueError("設定檔格式錯誤（最外層需為物件）")
    ver = data.get("version")
    warn = []
    if ver != SETTINGS_VERSION:
        warn.append(f"設定檔版本為 {ver}，本程式為 {SETTINGS_VERSION}；只套用可辨識的項目")

    out = {}
    thr = data.get("thresholds")
    if isinstance(thr, dict):
        clean = {}
        for lv in ("pixel", "block"):
            d = thr.get(lv)
            if not isinstance(d, dict):
                warn.append(f"thresholds.{lv} 缺少或格式錯誤，沿用目前值")
                continue
            vals, missing = {}, []
            for m in METRIC_NAMES:
                try:
                    vals[m] = float(d[m])
                except (KeyError, TypeError, ValueError):
                    missing.append(m)
            if missing:
                warn.append(f"thresholds.{lv} 缺少或無法解析: {', '.join(missing)}，這些沿用目前值")
            if vals:
                clean[lv] = vals
        if clean:
            out["thresholds"] = clean
    elif thr is not None:
        warn.append("thresholds 格式錯誤，沿用目前值")

    def take(key, cast, check=lambda v: True):
        if key in data:
            try:
                v = cast(data[key])
            except (TypeError, ValueError):
                warn.append(f"{key} 無法解析，沿用目前值")
                return
            if check(v):
                out[key] = v
            else:
                warn.append(f"{key} 數值不合法，沿用目前值")

    take("min_fail", int, lambda v: v >= 1)
    take("k_box", int, lambda v: v >= 1)
    take("tile_size", int, lambda v: v >= 2)
    take("diff_mode", str, lambda v: v in ("peak_to_peak", "max_adjacent"))
    take("y_mode", str, lambda v: v in Y_MODES)
    en = data.get("enabled")
    if isinstance(en, dict):
        clean = all_enabled()
        for lv in ("pixel", "block"):
            d = en.get(lv, {})
            if not isinstance(d, dict):
                warn.append(f"enabled.{lv} 格式錯誤，該層級視為全開")
                continue
            for m in METRIC_NAMES:
                if m in d:
                    clean[lv][m] = bool(d[m])
        out["enabled"] = clean
    elif en is not None:
        warn.append("enabled 格式錯誤，沿用目前值")
    take("pattern", str, lambda v: bool(v))
    if "risk_bounds" in data:
        try:
            out["risk_bounds"] = parse_risk_bounds(
                ",".join(str(x) for x in data["risk_bounds"]))
        except (ValueError, TypeError):
            warn.append("risk_bounds 不合法，沿用目前值")
    rules = data.get("tile_rules")
    if isinstance(rules, dict):
        clean = {}
        for k in ("RED", "ORANGE"):
            try:
                clean[k] = parse_tile_rule(",".join(str(x) for x in rules[k]))
            except (KeyError, ValueError, TypeError):
                warn.append(f"tile_rules.{k} 不合法，沿用目前值")
        if len(clean) == 2:
            out["tile_rules"] = clean
    elif rules is not None:
        warn.append("tile_rules 格式錯誤，沿用目前值")
    return out, warn


# --------------------------------------------------------------------------
# 單一序列完整分析
# --------------------------------------------------------------------------
def analyze_sequence(folder, pattern="*.bmp", thresholds=None,
                     diff_mode="peak_to_peak", block_size=2,
                     paths=None, name=None, y_mode=DEFAULT_Y_MODE,
                     progress=None):
    """
    回傳 {
      "n_frames", "shape",
      "pixel":  {"maps":..., "report":...},
      "block":  {"maps":..., "report":...},
      "stack":  原始序列
    }
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    # 進度分成 N(讀圖) + 3（pixel metric、block metric、評分）步
    n_steps = (len(paths) if paths else 0) + 3

    def _p(done, total, text):           # load_frames 的逐幀進度 -> 整體進度
        if progress:
            progress(done, n_steps, text)

    if paths is None:
        stack, paths = load_sequence(folder, pattern)
    else:
        stack = load_frames(paths, _p if progress else None)
    n_f = len(paths)

    if progress:
        progress(n_f, n_steps, "pixel metric")
    pixel_maps = compute_metric_maps(stack, block=1, diff_mode=diff_mode, y_mode=y_mode)
    if progress:
        progress(n_f + 1, n_steps, f"block {block_size}x{block_size} metric")
    block_maps = compute_metric_maps(stack, block=block_size, diff_mode=diff_mode,
                                     y_mode=y_mode)
    if progress:
        progress(n_f + 2, n_steps, "評分")

    return {
        "folder": folder,
        "name": name or os.path.basename(os.path.normpath(folder)),
        "paths": paths,
        "block_size": block_size,
        "y_mode": y_mode,
        "n_frames": stack.shape[0],
        "shape": stack.shape[1:3],
        "stack": stack,
        "pixel": {"maps": pixel_maps,
                  "report": evaluate_kpi(pixel_maps, thresholds["pixel"])},
        "block": {"maps": block_maps,
                  "report": evaluate_kpi(block_maps, thresholds["block"])},
    }


def format_report(result, level="pixel"):
    """輸出對齊的文字報表"""
    rep = result[level]["report"]
    title = "Pixel-based" if level == "pixel" else "Block-based (2x2)"
    lines = [f"--- {title} ---",
             f"{'metric':<15}{'thresh':>9}{'fail%':>10}{'fail_cnt':>11}"
             f"{'max':>11}{'p99.9':>11}{'mean':>10}{'max/thr':>9}  verdict"]
    for name in METRIC_NAMES:
        s = rep[name]
        verdict = "PASS" if s["fail_count"] == 0 else "FAIL"
        lines.append(
            f"{name:<15}{s['threshold']:>9.1f}{s['fail_ratio']*100:>9.4f}%"
            f"{s['fail_count']:>11d}{s['max']:>11.1f}{s['p999']:>11.1f}"
            f"{s['mean']:>10.2f}{s['margin']:>9.2f}  {verdict}"
        )
    return "\n".join(lines)


def format_full_report(result):
    head = (f"序列: {result['name']}  ({result['folder']})   Y: {result.get('y_mode')}\n"
            f"frames: {result['n_frames']}   解析度: "
            f"{result['shape'][1]}x{result['shape'][0]} (WxH)")
    return "\n".join([head, "", format_report(result, "pixel"),
                      "", format_report(result, "block")])

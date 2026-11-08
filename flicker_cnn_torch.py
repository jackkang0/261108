"""
Flicker 偵測 — 獨立的 PyTorch 訓練 / 推論腳本
=============================================
輸入只需要：影像序列資料夾 + 人工框的 CSV。程式自動切成 tile x tile 的 block，
訓練一個「這個 block 會不會閃」的分類器；推論時把判定會閃的 block 框在圖上輸出。

資料夾規格（與 flicker_kpi GUI 相同）
    <資料夾>/<prefix>0001.bmp ... <prefix>0006.bmp     一組序列 = 6 張連續幀
    <資料夾>/flicker_rois.csv                          人工框（可用 --rois 指定別的路徑）
        欄位: sequence,x0,y0,x1,y1    （x1/y1 不含端點；sequence 為不含編號的 prefix）
    CSV 裡沒有出現的序列 -> 視為整組都不閃（全部 block 為負樣本）
    加 --flicker-only 則改為「沒出現在 CSV 的序列直接略過」

用法
    # 0) 先檢查資料有沒有被正確讀到（不需要 PyTorch）
    python flicker_cnn_torch.py check  --data DIR [DIR ...] --tile 120 --dump-dir chk

    # 1) 訓練（GPU）
    python flicker_cnn_torch.py train  --data TRAIN_DIR [MORE_DIRS ...] --tile 120 \
           --epochs 60 --arch tiny2d --out model.pt --device cuda

    # 2) 推論：在圖上框出會閃的 block
    python flicker_cnn_torch.py predict --data TEST_DIR --model model.pt --out pred_out

    # 3) 把 tiny2d 權重轉成 GUI 能直接用的 .npz（推論端不需要 PyTorch）
    python flicker_cnn_torch.py export --model model.pt --out model_for_gui.npz

網路
    tiny2d  與 GUI 內建的 NumPy 版同構：6 幀殘差攤平成通道的 2D CNN，
            可以 export 成 .npz 給 GUI 用（推論不需 PyTorch）
    cnn3d   時間軸也做卷積的 3D CNN，容量較大，只能用 PyTorch 推論

輸入通道
    殘差      r_t = frame_t - mean_t(frame)，再除以整張圖殘差的 MAD
    --content 另外加 5 個內容通道（平均影像 R/G/B、梯度能量、局部噪聲底），
              用來表達「同樣大小的殘差在暗部/亮部、平坦區/紋理區的意義不同」
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import numpy as np
import cv2

VERSION = 1
FRAME_RE = re.compile(r"^(.*?)(\d{4})$")
ROI_FILE = "flicker_rois.csv"


# ==========================================================================
# 資料
# ==========================================================================
def imread_unicode(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"無法讀取: {path}")
    return img


def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if not ok:
        raise IOError(f"編碼失敗: {path}")
    buf.tofile(path)


def discover_sequences(folder, pattern="*.bmp", n_frames=6):
    """<prefix>0001..000N 分組；回傳 [{"name","paths"}], 缺幀的 prefix 列表"""
    groups = {}
    for p in sorted(glob.glob(os.path.join(folder, pattern))):
        stem = os.path.splitext(os.path.basename(p))[0]
        m = FRAME_RE.match(stem)
        if m:
            groups.setdefault(m.group(1), {})[int(m.group(2))] = p
    need = list(range(1, n_frames + 1))
    seqs, skipped = [], []
    for prefix in sorted(groups):
        fr = groups[prefix]
        missing = [i for i in need if i not in fr]
        if missing:
            skipped.append((prefix, missing))
            continue
        seqs.append({"name": prefix.rstrip("_-. ") or prefix,
                     "prefix": prefix, "paths": [fr[i] for i in need]})
    return seqs, skipped


def load_rois(path):
    """回傳 {sequence: [(x0,y0,x1,y1)]}"""
    out = {}
    if not path or not os.path.isfile(path):
        return out
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                box = tuple(int(float(row[k])) for k in ("x0", "y0", "x1", "y1"))
                name = row["sequence"]
            except (KeyError, TypeError, ValueError):
                continue
            if box[2] > box[0] and box[3] > box[1]:
                out.setdefault(name, []).append(box)
    return out


RES_CLIP = 200.0        # 正規化殘差的上限（float16 上限是 65504）
POS_W_MAX = 50.0        # BCEWithLogitsLoss 的 pos_weight 上限


def sequence_input(paths, content=False, eps=1e-6):
    """
    回傳 (X (C,H,W) float32, scale)
      18 ch 殘差（6 幀 x RGB，除以整張圖殘差的 MAD）
      +5 ch 內容（平均影像 R/G/B、梯度能量、局部噪聲底），各自標準化
    """
    frames = [imread_unicode(p)[:, :, ::-1].astype(np.float32) for p in paths]
    stack = np.stack(frames, axis=0)
    mean_img = stack.mean(axis=0)
    res = stack - mean_img[None]
    # 殘差的 MAD。靜態畫面 + dither 的素材有超過一半的殘差剛好是 0，
    # median 會等於 0；原本只加 eps=1e-6 當保護，除下去會放大到 1e7 量級，
    # 存成 float16（上限 65504）就變成 inf，之後 loss 立刻是 nan。
    # 以半個量化階當下限：小於 0.5 LSB 的殘差本來就沒有意義。
    mad = float(np.median(np.abs(res))) * 1.4826
    scale = max(mad, 0.5)
    # 再加一道保險：正規化後夾在 +-RES_CLIP（遠大於任何真實閃爍，但遠小於
    # float16 上限），單一壞點就無法污染整個 batch
    chans = [np.clip(res / scale, -RES_CLIP, RES_CLIP)
             .transpose(0, 3, 1, 2).reshape(-1, *res.shape[1:3])]
    if content:
        extra = [mean_img.transpose(2, 0, 1) / 128.0 - 1.0]
        g = (0.299 * mean_img[..., 0] + 0.587 * mean_img[..., 1]
             + 0.114 * mean_img[..., 2]).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        extra.append((np.log1p(np.abs(gx) + np.abs(gy)) / 5.0)[None])
        amp = np.abs(res).mean(axis=0).mean(axis=2)
        H, W = amp.shape
        small = cv2.resize(amp, (max(1, W // 16), max(1, H // 16)),
                           interpolation=cv2.INTER_AREA)
        extra.append(np.log1p(cv2.resize(small, (W, H),
                                         interpolation=cv2.INTER_NEAREST) / scale)[None])
        e = np.concatenate(extra, axis=0)
        mu = e.mean(axis=(1, 2), keepdims=True)
        sd = e.std(axis=(1, 2), keepdims=True) + 1e-3
        chans.append(np.clip((e - mu) / sd, -4, 4))
    return np.concatenate(chans, axis=0).astype(np.float32), scale


def split_tiles(X, tile):
    """(C,H,W) -> (n_tile, C, tile, tile), grid=(th,tw)；不足一格的邊緣捨去"""
    C, H, W = X.shape
    th, tw = H // tile, W // tile
    if th == 0 or tw == 0:
        raise ValueError(f"影像 {W}x{H} 比 tile {tile} 還小")
    x = X[:, :th * tile, :tw * tile].reshape(C, th, tile, tw, tile)
    return np.ascontiguousarray(x.transpose(1, 3, 0, 2, 4).reshape(th * tw, C, tile, tile)), (th, tw)


def tile_labels(boxes, grid, tile):
    th, tw = grid
    y = np.zeros(th * tw, np.float32)
    for (x0, y0, x1, y1) in boxes or ():
        for r in range(max(0, y0 // tile), min(th, -(-y1 // tile))):
            for c in range(max(0, x0 // tile), min(tw, -(-x1 // tile))):
                y[r * tw + c] = 1.0
    return y


def collect(dirs, tile, content, pattern="*.bmp", n_frames=6, rois_path=None,
            flicker_only=False, neg_per_pos=0, seed=0, verbose=True):
    """
    掃描資料夾 -> (X float16 (n,C,t,t), y, groups, info)
    neg_per_pos > 0 時對負樣本抽樣（省記憶體），正樣本全留。
    """
    rng = np.random.default_rng(seed)
    Xs, ys, gs, info = [], [], [], []
    for d in dirs:
        rois = load_rois(rois_path or os.path.join(d, ROI_FILE))
        seqs, skipped = discover_sequences(d, pattern, n_frames)
        if verbose:
            print(f"[{d}] 序列 {len(seqs)} 組，CSV 中有框的 {len(rois)} 組"
                  + (f"，缺幀略過 {len(skipped)} 組" if skipped else ""))
        for s in seqs:
            boxes = rois.get(s["name"], [])
            if flicker_only and not boxes:
                continue
            arr, scale = sequence_input(s["paths"], content)
            X, grid = split_tiles(arr, tile)
            y = tile_labels(boxes, grid, tile)
            idx = np.arange(len(y))
            if neg_per_pos > 0:
                pos = np.flatnonzero(y == 1)
                neg = np.flatnonzero(y == 0)
                k = min(len(neg), max(len(pos) * neg_per_pos, 20))
                idx = np.concatenate([pos, rng.choice(neg, k, replace=False)])
            xi = X[idx].astype(np.float16)
            if not np.isfinite(xi.astype(np.float32)).all():
                raise SystemExit(
                    f"{s['name']}: 正規化後的殘差超出 float16 範圍（scale={scale:g}）。"
                    "這通常代表整組序列幾乎沒有幀間變化，請確認輸入不是同一張圖複製 6 份。")
            Xs.append(xi)
            ys.append(y[idx])
            gs += [f"{os.path.basename(os.path.normpath(d))}/{s['name']}"] * len(idx)
            info.append({"dir": d, "name": s["name"], "grid": grid, "n_box": len(boxes),
                         "n_pos": int(y[idx].sum()), "n_tile": len(idx), "scale": scale})
            if verbose:
                print(f"    {s['name']:<20} grid {grid[0]}x{grid[1]}  "
                      f"框 {len(boxes)}  正樣本格 {int(y[idx].sum())}/{len(idx)}")
    if not Xs:
        raise SystemExit("沒有讀到任何序列，請檢查 --data / --pattern")
    return (np.concatenate(Xs), np.concatenate(ys), np.array(gs), info)


# ==========================================================================
# 模型（torch 只在這裡之後才用到）
# ==========================================================================
def build_model(arch, in_ch, n_frames=6):
    import torch
    import torch.nn as nn

    class Tiny2D(nn.Module):
        """與 GUI 內建 NumPy 版同構：conv-relu x3 -> (max, mean) pool -> linear"""

        def __init__(self, c_in, chans=(8, 16, 16)):
            super().__init__()
            layers, prev = [], c_in
            for c in chans:
                layers += [nn.Conv2d(prev, c, 3, stride=2, padding=0), nn.ReLU()]
                prev = c
            self.features = nn.Sequential(*layers)
            self.head = nn.Linear(prev * 2, 1)
            self.chans = tuple(chans)

        def forward(self, x):
            h = self.features(x)
            mx = h.amax(dim=(2, 3))
            mean = h.mean(dim=(2, 3))
            return self.head(torch.cat([mx, mean], dim=1)).squeeze(1)

    class CNN3D(nn.Module):
        """時間軸也做卷積；輸入 (B, C_per_frame, T, H, W)"""

        def __init__(self, c_in, c_extra=0, width=(16, 32, 48)):
            super().__init__()
            a, b, c = width
            self.stem = nn.Sequential(
                nn.Conv3d(c_in, a, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
                nn.BatchNorm3d(a), nn.ReLU(inplace=True),
                nn.Conv3d(a, b, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
                nn.BatchNorm3d(b), nn.ReLU(inplace=True))
            self.temporal = nn.Sequential(                      # 收掉時間軸
                nn.Conv3d(b, b, (3, 1, 1), padding=(1, 0, 0)),
                nn.BatchNorm3d(b), nn.ReLU(inplace=True))
            self.spatial = nn.Sequential(
                nn.Conv2d(b + c_extra, c, 3, stride=2, padding=1),
                nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, stride=2, padding=1),
                nn.BatchNorm2d(c), nn.ReLU(inplace=True))
            self.head = nn.Linear(c * 2, 1)
            self.c_extra = c_extra

        def forward(self, x, extra=None):
            h = self.temporal(self.stem(x))
            h = h.amax(dim=2)                                   # 時間軸取 max
            if self.c_extra and extra is not None:
                e = nn.functional.interpolate(extra, size=h.shape[-2:], mode="nearest")
                h = torch.cat([h, e], dim=1)
            h = self.spatial(h)
            return self.head(torch.cat([h.amax(dim=(2, 3)), h.mean(dim=(2, 3))],
                                       dim=1)).squeeze(1)

    if arch == "tiny2d":
        return Tiny2D(in_ch)
    c_extra = in_ch - 3 * n_frames
    return CNN3D(3, max(0, c_extra))


def to_model_input(xb, arch, n_frames=6):
    """(B,C,H,W) -> tiny2d 直接用；cnn3d 拆成 (B,3,T,H,W) + 內容通道"""
    if arch == "tiny2d":
        return xb, None
    b, c, h, w = xb.shape
    res = xb[:, :3 * n_frames].reshape(b, n_frames, 3, h, w).permute(0, 2, 1, 3, 4)
    extra = xb[:, 3 * n_frames:] if c > 3 * n_frames else None
    return res.contiguous(), extra


# ==========================================================================
# 評估
# ==========================================================================
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


def auc_score(scores, y):
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


def pick_threshold(scores, y, groups, budget):
    """在訓練資料上挑門檻：每組序列的框外誤判格 <= budget，且框召回最大"""
    best_t, best_hit = float("inf"), -1
    for t in np.concatenate([[np.inf], np.sort(np.unique(scores))[::-1]]):
        pred = scores >= t
        if any(int((pred & (y == 0) & (groups == g)).sum()) > budget
               for g in np.unique(groups)):
            continue
        hit = int((pred & (y == 1)).sum())
        if hit > best_hit:
            best_hit, best_t = hit, float(t)
    return best_t


def report(s_tr, y_tr, g_tr, s_va, y_va, g_va, budgets=(0, 1, 3)):
    lines = [f"驗證集 AUC: {auc_score(s_va, y_va):.4f}",
             f"{'每序列容許框外誤判格':<22}{'框召回':>10}{'實際誤判格':>12}"]
    for b in budgets:
        t = pick_threshold(s_tr, y_tr, g_tr, b)
        pred = s_va >= t
        rec = float((pred & (y_va == 1)).sum()) / max(int((y_va == 1).sum()), 1)
        fp = int((pred & (y_va == 0)).sum())
        lines.append(f"{b:<22}{rec*100:>9.0f}%{fp:>12}")
    lines.append("  （分數門檻在訓練序列上決定，再套到沒看過的驗證序列）")
    return "\n".join(lines)


# ==========================================================================
# 訓練
# ==========================================================================
def split_by_group(groups, y, val_ratio, seed=0):
    """以序列為單位切訓練/驗證，盡量讓驗證集含有正樣本序列"""
    rng = np.random.default_rng(seed)
    seqs = np.unique(groups)
    pos_seq = np.unique(groups[y == 1])
    n_val = max(1, int(round(len(seqs) * val_ratio)))
    n_pos_val = max(1, int(round(len(pos_seq) * val_ratio)))
    val = set(rng.choice(pos_seq, min(n_pos_val, len(pos_seq)), replace=False).tolist())
    rest = [s for s in seqs if s not in val]
    if rest and len(val) < n_val:
        val |= set(rng.choice(rest, min(n_val - len(val), len(rest)),
                              replace=False).tolist())
    return ~np.isin(groups, list(val)), np.isin(groups, list(val)), sorted(val)


def augment_batch(xb, rng):
    """翻轉 / 旋轉 90 度 / 時間反轉（閃爍對這些變換不變）"""
    import torch
    if rng.random() < 0.5:
        xb = torch.flip(xb, dims=[-1])
    if rng.random() < 0.5:
        xb = torch.flip(xb, dims=[-2])
    k = int(rng.integers(0, 4))
    if k:
        xb = torch.rot90(xb, k, dims=[-2, -1])
    return xb.contiguous()


def cmd_train(a):
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    dev = torch.device(a.device if a.device != "auto"
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"裝置: {dev}" + (f" ({torch.cuda.get_device_name(0)})"
                            if dev.type == "cuda" else ""))
    X, y, g, _ = collect(a.data, a.tile, a.content, a.pattern, a.frames,
                         a.rois, a.flicker_only, a.neg_per_pos, a.seed)
    print(f"樣本 {X.shape}  正 {int(y.sum())}  負 {int((y==0).sum())}  "
          f"序列 {len(np.unique(g))} 組")
    if y.sum() < 5:
        raise SystemExit("正樣本太少（< 5 格），請先多畫一些框")

    tr, va, val_seq = split_by_group(g, y, a.val_ratio, a.seed)
    print(f"驗證序列（{len(val_seq)} 組）: {', '.join(val_seq)}")

    model = build_model(a.arch, X.shape[1], a.frames).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"架構 {a.arch}，參數 {n_par:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    # 正樣本權重：負/正比值。不設上限時（例如沒用 --neg-per-pos 抽樣）
    # 可能到數百倍，正樣本的梯度跟著放大數百倍，tiny2d 沒有 BatchNorm 會直接發散
    ratio = float((y[tr] == 0).sum()) / max(float((y[tr] == 1).sum()), 1.0)
    if ratio > POS_W_MAX:
        print(f"  正負比 {ratio:.0f}:1 過大，pos_weight 限制在 {POS_W_MAX:g}"
              f"（建議改用 --neg-per-pos 抽樣負樣本）")
    pos_w = torch.tensor([min(ratio, POS_W_MAX)],
                         dtype=torch.float32, device=dev)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)

    ds = TensorDataset(torch.from_numpy(X[tr].astype(np.float32)),
                       torch.from_numpy(y[tr]))
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=False,
                    num_workers=a.workers, pin_memory=(dev.type == "cuda"))
    rng = np.random.default_rng(a.seed)
    Xva = torch.from_numpy(X[va].astype(np.float32))
    best, best_state, t0 = -1.0, None, time.time()

    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            if not a.no_augment:
                xb = augment_batch(xb, rng)
            inp, extra = to_model_input(xb, a.arch, a.frames)
            opt.zero_grad(set_to_none=True)
            out = model(inp) if extra is None else model(inp, extra)
            loss = lossf(out, yb)
            if not torch.isfinite(loss):
                raise SystemExit(
                    f"第 {ep} 輪的 loss 變成 {float(loss)}，訓練中止。\n"
                    f"  輸入是否有限: {bool(torch.isfinite(inp).all())}；"
                    f"輸入絕對值最大 {float(inp.abs().max()):.3g}\n"
                    f"  模型輸出是否有限: {bool(torch.isfinite(out).all())}\n"
                    "  常見原因是學習率過高，請降 --lr（例如 1e-3）後重試。")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss) * len(yb)
        sched.step()
        s_va = infer_scores(model, Xva, a.arch, a.frames, dev, a.batch)
        va_auc = auc_score(s_va, y[va]) if va.any() else float("nan")
        print(f"  epoch {ep:>3}/{a.epochs}  loss {tot/max(tr.sum(),1):.4f}"
              + (f"  驗證 AUC {va_auc:.4f}" if va_auc == va_auc else "")
              + f"  {time.time()-t0:.0f}s")
        if va_auc == va_auc and va_auc > best:
            best = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None and not a.last_epoch:
        model.load_state_dict(best_state)
        print(f"採用驗證 AUC 最佳的那一輪（{best:.4f}）")

    s_tr = infer_scores(model, torch.from_numpy(X[tr].astype(np.float32)),
                        a.arch, a.frames, dev, a.batch)
    s_va = infer_scores(model, Xva, a.arch, a.frames, dev, a.batch)
    txt = report(s_tr, y[tr], g[tr], s_va, y[va], g[va]) if va.any() else "（沒有驗證集）"
    print("\n" + txt)

    meta = {"version": VERSION, "arch": a.arch, "tile": a.tile, "in_ch": int(X.shape[1]),
            "content": bool(a.content), "frames": a.frames, "epochs": a.epochs,
            "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
            "n_seq": int(len(np.unique(g))), "val_seq": val_seq,
            "val_auc": None if best < 0 else round(float(best), 4),
            "chans": list(getattr(model, "chans", ()))}
    torch.save({"meta": meta, "state_dict": model.state_dict()}, a.out)
    print(f"\n模型已存檔: {a.out}")
    if a.arch == "tiny2d":
        print(f"可用 export 轉成 GUI 能直接載入的 .npz：\n"
              f"    python {os.path.basename(__file__)} export --model {a.out} "
              f"--out {os.path.splitext(a.out)[0]}.npz")


def infer_scores(model, X, arch, frames, dev, batch=64):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i + batch].to(dev)
            inp, extra = to_model_input(xb, arch, frames)
            z = model(inp) if extra is None else model(inp, extra)
            out.append(torch.sigmoid(z).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


# ==========================================================================
# 推論：在圖上框出會閃的 block
# ==========================================================================
def draw_result(paths, scores, grid, tile, thr, show_score=True):
    """第 1 幀上畫出超過門檻的 block；回傳 BGR 影像"""
    img = imread_unicode(paths[0]).copy()
    th, tw = grid
    hot = np.argsort(-scores)
    for i in hot:
        s = float(scores[i])
        r, c = divmod(int(i), tw)
        if s < thr:
            break
        x0, y0 = c * tile, r * tile
        cv2.rectangle(img, (x0, y0), (x0 + tile - 1, y0 + tile - 1), (0, 0, 255), 3)
        if show_score:
            cv2.putText(img, f"{s:.2f}", (x0 + 4, y0 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    n = int((scores >= thr).sum())
    cv2.putText(img, f"flicker blocks: {n}/{len(scores)}  thr={thr:g}",
                (10, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255) if n else (0, 200, 0), 2, cv2.LINE_AA)
    return img


def cmd_predict(a):
    import torch
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    dev = torch.device(a.device if a.device != "auto"
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(meta["arch"], meta["in_ch"], meta.get("frames", 6)).to(dev)
    model.load_state_dict(ck["state_dict"])
    tile = a.tile or meta["tile"]
    os.makedirs(a.out, exist_ok=True)
    rows = [["dir", "sequence", "tile_col", "tile_row", "x0", "y0", "x1", "y1", "score",
             "flagged"]]
    for d in a.data:
        seqs, _ = discover_sequences(d, a.pattern, meta.get("frames", 6))
        for s in seqs:
            arr, _ = sequence_input(s["paths"], meta.get("content", False))
            X, grid = split_tiles(arr, tile)
            sc = infer_scores(model, torch.from_numpy(X.astype(np.float32)),
                              meta["arch"], meta.get("frames", 6), dev, a.batch)
            img = draw_result(s["paths"], sc, grid, tile, a.thr, not a.no_score_text)
            name = f"{os.path.basename(os.path.normpath(d))}_{s['name']}"
            imwrite_unicode(os.path.join(a.out, f"{name}_pred.png"), img)
            th, tw = grid
            for i, v in enumerate(sc):
                r, c = divmod(i, tw)
                rows.append([d, s["name"], c, r, c * tile, r * tile,
                             (c + 1) * tile, (r + 1) * tile, f"{v:.4f}",
                             int(v >= a.thr)])
            print(f"{name}: 會閃的 block {int((sc>=a.thr).sum())}/{len(sc)}"
                  f"   最高分 {sc.max():.3f}")
    with open(os.path.join(a.out, "tile_scores.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)
    print(f"\n輸出目錄: {a.out}（標註圖 + tile_scores.csv）")


# ==========================================================================
# export / check
# ==========================================================================
def cmd_export(a):
    """tiny2d -> GUI 的 .npz（conv weight 轉成 NumPy 版的排列）"""
    import torch
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    if meta["arch"] != "tiny2d":
        raise SystemExit("只有 arch=tiny2d 能匯出給 GUI（cnn3d 需要 PyTorch 推論）")
    sd = ck["state_dict"]
    convs = [(k, v) for k, v in sd.items() if k.startswith("features") and k.endswith("weight")]
    convs.sort(key=lambda kv: int(kv[0].split(".")[1]))
    Ws = [v.numpy().astype(np.float32) for _, v in convs]
    bs = [sd[k.replace("weight", "bias")].numpy().astype(np.float32) for k, _ in convs]
    arrs = {f"p{i}": w for i, w in enumerate(Ws)}
    arrs.update({f"p{len(Ws)+i}": b for i, b in enumerate(bs)})
    arrs[f"p{2*len(Ws)}"] = sd["head.weight"].numpy().ravel().astype(np.float32)
    arrs[f"p{2*len(Ws)+1}"] = sd["head.bias"].numpy().ravel().astype(np.float32)
    gui_meta = {"kind": "cnn", "tile": meta["tile"], "in_ch": meta["in_ch"],
                "content": meta.get("content", False), "from": "pytorch",
                "val_auc": meta.get("val_auc")}
    np.savez_compressed(a.out, meta=np.array(json.dumps(gui_meta)), n=len(Ws), **arrs)
    print(f"已匯出: {a.out}（在 GUI 按「載入 AI 模型...」選這個檔案）")


def cmd_check(a):
    """只跑資料流程，不需要 PyTorch"""
    X, y, g, info = collect(a.data, a.tile, a.content, a.pattern, a.frames,
                            a.rois, a.flicker_only, 0, a.seed)
    print(f"\n合計：格子 {len(y)}，正樣本 {int(y.sum())}，序列 {len(np.unique(g))} 組，"
          f"輸入通道 {X.shape[1]}")
    if y.sum() == 0:
        print("!! 沒有任何正樣本：確認 CSV 的 sequence 欄位與序列名稱一致"
              "（序列名稱是檔名去掉 4 位數編號與結尾的底線）")
    if a.dump_dir:
        os.makedirs(a.dump_dir, exist_ok=True)
        for d in a.data:
            rois = load_rois(a.rois or os.path.join(d, ROI_FILE))
            for s in discover_sequences(d, a.pattern, a.frames)[0]:
                boxes = rois.get(s["name"], [])
                if a.flicker_only and not boxes:
                    continue
                img = imread_unicode(s["paths"][0]).copy()
                th, tw = (img.shape[0] // a.tile, img.shape[1] // a.tile)
                for r in range(th):
                    for c in range(tw):
                        cv2.rectangle(img, (c * a.tile, r * a.tile),
                                      ((c + 1) * a.tile - 1, (r + 1) * a.tile - 1),
                                      (90, 90, 90), 1)
                lab = tile_labels(boxes, (th, tw), a.tile)
                for i in np.flatnonzero(lab):
                    r, c = divmod(int(i), tw)
                    cv2.rectangle(img, (c * a.tile + 4, r * a.tile + 4),
                                  ((c + 1) * a.tile - 5, (r + 1) * a.tile - 5),
                                  (0, 255, 0), 3)      # 內縮，避免被 CSV 的框蓋住
                for (x0, y0, x1, y1) in boxes:
                    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 0), 2)
                name = f"{os.path.basename(os.path.normpath(d))}_{s['name']}_check.png"
                imwrite_unicode(os.path.join(a.dump_dir, name), img)
        print(f"已輸出標記檢查圖到 {a.dump_dir}（藍=CSV 的框，綠=被標成正樣本的格）")


# ==========================================================================
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Flicker 偵測：以 block 為單位的 CNN 訓練與推論",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q, need_data=True):
        if need_data:
            q.add_argument("--data", nargs="+", required=True, help="影像序列資料夾")
        q.add_argument("--tile", type=int, default=120, help="block 大小")
        q.add_argument("--pattern", default="*.bmp", help="影像檔名 pattern")
        q.add_argument("--frames", type=int, default=6, help="一組序列幾幀")
        q.add_argument("--rois", default=None,
                       help=f"人工框 CSV（預設用各資料夾裡的 {ROI_FILE}）")
        q.add_argument("--content", action="store_true",
                       help="加入 5 個內容通道（亮度/顏色/梯度/局部噪聲底）")
        q.add_argument("--flicker-only", action="store_true",
                       help="略過 CSV 中沒有框的序列（預設視為整組不閃）")
        q.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("train", help="訓練")
    common(t)
    t.add_argument("--out", default="flicker_model.pt")
    t.add_argument("--arch", choices=["tiny2d", "cnn3d"], default="tiny2d")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--val-ratio", type=float, default=0.25, help="驗證序列比例")
    t.add_argument("--neg-per-pos", type=int, default=10,
                   help="每個正樣本保留幾個負樣本（0 = 全留）")
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--device", default="auto", help="cuda / cpu / auto")
    t.add_argument("--no-augment", action="store_true")
    t.add_argument("--last-epoch", action="store_true",
                   help="用最後一輪而不是驗證最佳的那一輪")
    t.set_defaults(func=cmd_train)

    q = sub.add_parser("predict", help="推論並在圖上框出會閃的 block")
    q.add_argument("--data", nargs="+", required=True)
    q.add_argument("--model", required=True)
    q.add_argument("--out", default="pred_out")
    q.add_argument("--thr", type=float, default=0.5)
    q.add_argument("--tile", type=int, default=None, help="預設用模型記錄的大小")
    q.add_argument("--pattern", default="*.bmp")
    q.add_argument("--batch", type=int, default=64)
    q.add_argument("--device", default="auto")
    q.add_argument("--no-score-text", action="store_true")
    q.set_defaults(func=cmd_predict)

    e = sub.add_parser("export", help="tiny2d 權重 -> GUI 用的 .npz")
    e.add_argument("--model", required=True)
    e.add_argument("--out", default="model_for_gui.npz")
    e.set_defaults(func=cmd_export)

    c = sub.add_parser("check", help="只檢查資料（不需要 PyTorch）")
    common(c)
    c.add_argument("--dump-dir", default=None, help="輸出標記檢查圖的資料夾")
    c.set_defaults(func=cmd_check)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()

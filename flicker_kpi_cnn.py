"""
Flicker KPI — AI 偵測（路線 B：小型 CNN，吃原始 6 幀）
======================================================
輸入（預設 23 個通道）
  18 ch  時間殘差 r_t = frame_t - mean_t(frame)，除以整張圖殘差的 MAD
         （畫面內容被消掉，只留 pipeline 造成的擾動；數值是「相對噪聲底的倍率」）
   3 ch  6 幀平均影像的 R/G/B（內容本身）
   1 ch  平均影像的梯度能量（平坦區 / 邊緣紋理區）
   1 ch  局部噪聲底（16x16 區塊的殘差 MAD，取 log）
  後 5 個通道是必要的：同樣大小的殘差，在暗部與亮部、平坦區與紋理區的可見度不同，
  壓縮與 SR 的行為也跟局部內容有關。只給殘差的話網路無從判斷這些差異。

網路（約 4 千個參數，刻意很小，因為標記資料通常只有幾百個框）
      conv 3x3 stride2 (18->8)  ReLU
      conv 3x3 stride2 (8->16)  ReLU
      conv 3x3 stride2 (16->16) ReLU
      global max-pool + mean-pool -> 32 維 -> linear -> logit
  max-pool 是關鍵：120x120 的格子裡閃爍可能只占幾個 pixel，用平均會被稀釋。

實作全部以 NumPy 完成（含反向傳播與 Adam），訓練和推論都不需要額外套件。
"""

import numpy as np
import cv2

from flicker_kpi_core import imread_unicode

MODEL_VERSION = 1
DEFAULT_TILE = 120
RES_CLIP = 200.0        # 正規化殘差的上限（存成 float16 時上限是 65504）


# --------------------------------------------------------------------------
# 資料前處理
# --------------------------------------------------------------------------
def sequence_input(paths, content=True, eps=1e-6):
    """
    回傳 (X (C,H,W) float32, scale)
      C = 18（純殘差）或 23（殘差 + 內容 + 局部噪聲底）
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
        chans.append((mean_img.transpose(2, 0, 1) / 128.0 - 1.0))      # 內容 R/G/B
        g = (0.299 * mean_img[..., 0] + 0.587 * mean_img[..., 1]
             + 0.114 * mean_img[..., 2]).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        chans.append((np.log1p(np.abs(gx) + np.abs(gy)) / 5.0)[None])  # 梯度能量
        amp = np.abs(res).mean(axis=0).mean(axis=2)                    # 局部噪聲底
        H, W = amp.shape
        small = cv2.resize(amp, (max(1, W // 16), max(1, H // 16)),
                           interpolation=cv2.INTER_AREA)
        local = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
        chans.append((np.log1p(local / scale))[None])
        # 內容通道各自標準化到與殘差相近的尺度，否則初期會被網路忽略
        extra = np.concatenate(chans[1:], axis=0)
        mu = extra.mean(axis=(1, 2), keepdims=True)
        sd = extra.std(axis=(1, 2), keepdims=True) + 1e-3
        chans[1:] = [np.clip((extra - mu) / sd, -4, 4)]
    return np.concatenate(chans, axis=0).astype(np.float32), scale


def sequence_residual(paths, eps=1e-6):
    """
    讀 6 幀 -> (N, H, W, 3) 時間殘差 / 噪聲尺度
    噪聲尺度取整張圖殘差的 MAD（中位數絕對偏差），對少數強閃爍不敏感。
    """
    frames = [imread_unicode(p)[:, :, ::-1].astype(np.float32) for p in paths]
    stack = np.stack(frames, axis=0)
    res = stack - stack.mean(axis=0, keepdims=True)
    scale = float(np.median(np.abs(res))) * 1.4826 + eps
    return (res / scale).astype(np.float32), scale


def tile_patches(res, tile=DEFAULT_TILE):
    """(N,H,W,3) -> (n_tile, N*3, tile, tile)；通道順序為 frame-major（舊介面）"""
    N, H, W, C = res.shape
    th, tw = H // tile, W // tile
    x = res[:, :th * tile, :tw * tile]
    x = x.reshape(N, th, tile, tw, tile, C)
    x = x.transpose(1, 3, 0, 5, 2, 4)
    return np.ascontiguousarray(x.reshape(th * tw, N * C, tile, tile)), (th, tw)


def split_tiles(X, tile=DEFAULT_TILE):
    """(C,H,W) -> (n_tile, C, tile, tile)"""
    C, H, W = X.shape
    th, tw = H // tile, W // tile
    x = X[:, :th * tile, :tw * tile].reshape(C, th, tile, tw, tile)
    x = x.transpose(1, 3, 0, 2, 4)
    return np.ascontiguousarray(x.reshape(th * tw, C, tile, tile)), (th, tw)


def tile_labels(boxes, grid, tile=DEFAULT_TILE):
    th, tw = grid
    y = np.zeros(th * tw, np.int8)
    for (x0, y0, x1, y1) in boxes or ():
        for r in range(max(0, y0 // tile), min(th, -(-y1 // tile))):
            for c in range(max(0, x0 // tile), min(tw, -(-x1 // tile))):
                y[r * tw + c] = 1
    return y


# --------------------------------------------------------------------------
# 基本層（im2col 版卷積 + 反向傳播）
# --------------------------------------------------------------------------
def _im2col(x, k, s):
    n, c, h, w = x.shape
    oh, ow = (h - k) // s + 1, (w - k) // s + 1
    strd = x.strides
    view = np.lib.stride_tricks.as_strided(
        x, shape=(n, c, oh, ow, k, k),
        strides=(strd[0], strd[1], strd[2] * s, strd[3] * s, strd[2], strd[3]))
    return view.reshape(n, c, oh, ow, k * k), (oh, ow)


def conv_forward(x, W, b, s):
    k = W.shape[-1]
    col, (oh, ow) = _im2col(x, k, s)
    n, c = x.shape[:2]
    col = col.transpose(0, 2, 3, 1, 4).reshape(n * oh * ow, c * k * k)
    out = col @ W.reshape(W.shape[0], -1).T + b
    return out.reshape(n, oh, ow, -1).transpose(0, 3, 1, 2), (col, x.shape, (oh, ow))


def conv_backward(dout, cache, W, s):
    col, xshape, (oh, ow) = cache
    n, c, h, w = xshape
    k = W.shape[-1]
    dout_t = dout.transpose(0, 2, 3, 1).reshape(n * oh * ow, -1)
    dW = (dout_t.T @ col).reshape(W.shape)
    db = dout_t.sum(axis=0)
    dcol = (dout_t @ W.reshape(W.shape[0], -1)).reshape(n, oh, ow, c, k * k)
    dx = np.zeros(xshape, np.float32)
    dcol = dcol.transpose(0, 3, 1, 2, 4)                      # (n,c,oh,ow,k*k)
    for i in range(k):                                        # k 很小，直接累加
        for j in range(k):
            dx[:, :, i:i + oh * s:s, j:j + ow * s:s] += dcol[..., i * k + j]
    return dx, dW, db


class TinyCNN:
    """conv-relu x3 -> (max, mean) global pool -> linear"""

    def __init__(self, in_ch, chans=(8, 16, 16), seed=0):
        rng = np.random.default_rng(seed)
        self.s = 2
        self.k = 3
        self.W, self.b = [], []
        prev = in_ch
        for c in chans:
            fan = prev * self.k * self.k
            self.W.append((rng.standard_normal((c, prev, self.k, self.k))
                           * np.sqrt(2.0 / fan)).astype(np.float32))
            self.b.append(np.zeros(c, np.float32))
            prev = c
        self.Wf = (rng.standard_normal(prev * 2) * 0.1).astype(np.float32)
        self.bf = np.float32(0.0)

    # ---- forward / backward ----
    def forward(self, x, cache=None):
        h = x
        caches = []
        for W, b in zip(self.W, self.b):
            h, cc = conv_forward(h, W, b, self.s)
            mask = h > 0
            h = h * mask
            caches.append((cc, mask))
        n, c = h.shape[:2]
        flat = h.reshape(n, c, -1)
        idx = flat.argmax(axis=2)
        mx = np.take_along_axis(flat, idx[:, :, None], axis=2)[:, :, 0]
        mean = flat.mean(axis=2)
        feat = np.concatenate([mx, mean], axis=1)
        z = feat @ self.Wf + self.bf
        if cache is not None:
            cache.update(caches=caches, feat=feat, idx=idx, hshape=h.shape)
        return z

    def backward(self, dz, cache):
        feat, idx, hshape = cache["feat"], cache["idx"], cache["hshape"]
        n, c, oh, ow = hshape
        dWf = feat.T @ dz
        dbf = dz.sum()
        dfeat = dz[:, None] * self.Wf[None, :]
        dmx, dmean = dfeat[:, :c], dfeat[:, c:]
        dflat = np.zeros((n, c, oh * ow), np.float32)
        np.put_along_axis(dflat, idx[:, :, None], dmx[:, :, None], axis=2)
        dflat += (dmean / (oh * ow))[:, :, None]
        dh = dflat.reshape(hshape)
        dW_all, db_all = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            cc, mask = cache["caches"][i]
            dh = dh * mask
            dh, dW, db = conv_backward(dh, cc, self.W[i], self.s)
            dW_all[i], db_all[i] = dW, db
        return dW_all, db_all, dWf, dbf

    # ---- 參數存取 ----
    def params(self):
        return self.W + self.b + [self.Wf, np.array([self.bf], np.float32)]

    def set_params(self, ps):
        n = len(self.W)
        self.W = [np.asarray(p, np.float32) for p in ps[:n]]
        self.b = [np.asarray(p, np.float32) for p in ps[n:2 * n]]
        self.Wf = np.asarray(ps[2 * n], np.float32)
        self.bf = float(np.asarray(ps[2 * n + 1]).ravel()[0])


# --------------------------------------------------------------------------
# 訓練
# --------------------------------------------------------------------------
def augment(x, rng):
    """翻轉、旋轉 90 度、幀順序反轉（閃爍對這些變換不變）"""
    if rng.random() < 0.5:
        x = x[:, :, :, ::-1]
    if rng.random() < 0.5:
        x = x[:, :, ::-1, :]
    r = int(rng.integers(0, 4))
    if r:
        x = np.rot90(x, r, axes=(2, 3))
    return np.ascontiguousarray(x)


def train(X, y, in_ch=None, epochs=40, batch=16, lr=3e-3, wd=1e-4, seed=0,
          chans=(8, 16, 16), val=None, log=None):
    """
    X: (n, C, t, t) float32；y: (n,) 0/1
    回傳 (model, history)
    """
    rng = np.random.default_rng(seed)
    net = TinyCNN(in_ch or X.shape[1], chans, seed)
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    y = np.asarray(y, np.float32)
    pos, neg = max(y.sum(), 1), max((1 - y).sum(), 1)
    w_pos, w_neg = 0.5 * len(y) / pos, 0.5 * len(y) / neg
    hist, step = [], 0
    for ep in range(epochs):
        order = rng.permutation(len(X))
        tot = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            xb = augment(np.asarray(X[idx], np.float32), rng)
            yb = y[idx]
            cache = {}
            z = net.forward(xb, cache)
            wb = np.where(yb > 0, w_pos, w_neg).astype(np.float32)
            loss = float((wb * (np.logaddexp(0, z) - yb * z)).mean())
            dz = (wb * (1 / (1 + np.exp(-z)) - yb) / len(idx)).astype(np.float32)
            dW, db, dWf, dbf = net.backward(dz, cache)
            grads = dW + db + [dWf, np.array([dbf], np.float32)]
            ps = net.params()
            step += 1
            for j, (p, g) in enumerate(zip(ps, grads)):
                g = g + wd * p
                m[j] = 0.9 * m[j] + 0.1 * g
                v[j] = 0.999 * v[j] + 0.001 * g * g
                mh = m[j] / (1 - 0.9 ** step)
                vh = v[j] / (1 - 0.999 ** step)
                ps[j] = p - lr * mh / (np.sqrt(vh) + 1e-8)
            net.set_params(ps)
            tot += loss * len(idx)
        rec = {"epoch": ep + 1, "loss": tot / len(X)}
        if val is not None:
            rec["val_auc"] = auc(predict(net, val[0]), val[1])
        hist.append(rec)
        if log:
            log(rec)
    return net, hist


def predict(net, X, batch=64):
    out = []
    for i in range(0, len(X), batch):
        out.append(net.forward(np.ascontiguousarray(X[i:i + batch], np.float32)))
    z = np.concatenate(out) if out else np.zeros(0)
    return 1.0 / (1.0 + np.exp(-z))


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


# --------------------------------------------------------------------------
# 存檔 / 載入
# --------------------------------------------------------------------------
def save(path, net, meta):
    arrs = {f"p{i}": p for i, p in enumerate(net.params())}
    import json
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), n=len(net.W), **arrs)
    return path


def load(path):
    import json
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    n = int(d["n"])
    ps = [d[f"p{i}"] for i in range(2 * n + 2)]
    net = TinyCNN(ps[0].shape[1], tuple(p.shape[0] for p in ps[:n]))
    net.set_params(ps)
    return net, meta


def score_tiles(net, paths, tile=DEFAULT_TILE, content=True):
    """回傳 (分數圖 (th,tw), grid)"""
    arr, _ = sequence_input(paths, content)
    X, grid = split_tiles(arr, tile)
    return predict(net, X).reshape(grid), grid

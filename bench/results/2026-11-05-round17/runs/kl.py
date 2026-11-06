# KL(ref || test) of the tower's logits per step: the main token's head (CS + 1) and the 9 code heads (CS each)
import sys
import numpy as np
CS, NH = 4096, 9
W = CS + 1 + NH * CS
a = np.fromfile(sys.argv[1], dtype=np.float32)
b = np.fromfile(sys.argv[2], dtype=np.float32)
n = min(a.size, b.size) // W
a = a[: n * W].reshape(n, W).astype(np.float64)
b = b[: n * W].reshape(n, W).astype(np.float64)
def kl(p, q):
    p = p - p.max(1, keepdims=True); q = q - q.max(1, keepdims=True)
    lp = p - np.log(np.exp(p).sum(1, keepdims=True)); lq = q - np.log(np.exp(q).sum(1, keepdims=True))
    return (np.exp(lp) * (lp - lq)).sum(1), (p.argmax(1) == q.argmax(1)).mean()
out = []
k, t = kl(a[:, : CS + 1], b[:, : CS + 1])
out.append(f"main {k.mean():.4f} top1 {t*100:.1f}%")
ks, ts = [], []
for h in range(NH):
    s = CS + 1 + h * CS
    k, t = kl(a[:, s : s + CS], b[:, s : s + CS])
    ks.append(k.mean()); ts.append(t)
out.append("codes " + " ".join(f"{x:.4f}" for x in ks) + f" | mean {np.mean(ks):.4f} top1 {np.mean(ts)*100:.1f}%")
print(f"{sys.argv[2].split('/')[-1]}: {n} steps; " + "; ".join(out))

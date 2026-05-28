"""
独立验证 torch.sparse CSR 的 index_select + matmul 在当前环境是否可用。
在服务器上先跑这个, 通过了再跑 eval_zero_shot.py。
  python test_csr.py
"""
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__}, device={device}")

# 3 个 candidate 的稀疏激活 (cand1 的 feat5 故意重复, 测 coalesce 累加)
sparse_list = [
    (torch.tensor([1.0, 2.0], device=device), torch.tensor([0, 5], device=device)),
    (torch.tensor([3.0, 1.0, 1.0], device=device), torch.tensor([5, 5, 2], device=device)),
    (torch.tensor([1.0], device=device), torch.tensor([2], device=device)),
]
N = len(sparse_list)

rows, cols, vals = [], [], []
for i, (acts, inds) in enumerate(sparse_list):
    rows.append(torch.full((inds.shape[0],), i, dtype=torch.long, device=device))
    cols.append(inds.long())
    vals.append(acts.float())
rows = torch.cat(rows); cols = torch.cat(cols); vals = torch.cat(vals)
V_total = int(cols.max().item()) + 1
print("V_total =", V_total)

# COO + coalesce
coo = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (N, V_total)).coalesce()
print("coalesced values (cand1 feat5 应=4):", coo.values().tolist())

# 行 L2 归一化
ci = coo.indices(); cv = coo.values()
row_idx = ci[0]
row_sq = torch.zeros(N, device=device).scatter_add_(0, row_idx, cv * cv)
row_norm = row_sq.sqrt() + 1e-8
cv_normed = cv / row_norm[row_idx]

# 转 CSR
try:
    csr = torch.sparse_coo_tensor(ci, cv_normed, (N, V_total)).coalesce().to_sparse_csr()
    print("[OK] to_sparse_csr")
except Exception as e:
    print("[FAIL] to_sparse_csr:", e); raise

# index_select 不支持 CSR! 改成全池 matmul 再取子集
q = torch.zeros(V_total, device=device); q[5] = 1.0; q = q / (q.norm() + 1e-8)
try:
    all_sims = (csr @ q.unsqueeze(1)).squeeze(1)   # [N] 全池
    print("[OK] CSR @ dense (full pool), all_sims:", all_sims.tolist())
except Exception as e:
    print("[FAIL] CSR @ dense:", e); raise

cand_idx = torch.tensor([0, 2], device=device)
sims = all_sims[cand_idx]      # 稠密索引取子集
print("[OK] dense index subset, sims (cand0, cand2):", sims.tolist())
print("  期望 ~[0.894, 0.0] (cand0 feat5 归一化分量, cand2 无重叠)")

print("\n全部通过! CSR 方案可用 (全池 matmul + 稠密取子集)。")
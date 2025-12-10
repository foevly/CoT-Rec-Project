import numpy as np

emb_path = "/Users/foevly/Desktop/DL/CoT-Rec-main/CoT-Rec/GPT/output_64/user/Grocery_and_Gourmet_Food_user.npy"

emb = np.load(emb_path)  # 读入矩阵

print("形状:", emb.shape)  # (num_users, dim)，例如 (6849, 64)

# 看前两行的前 10 个维度
print("第 0 个用户的前 10 维:", emb[0, :10])
print("第 1 个用户的前 10 维:", emb[1, :10])

# 检查一下 L2 范数（之前我们做了 normalize_l2）
norms = np.linalg.norm(emb, axis=1)
print("范数范围:", norms.min(), "→", norms.max())
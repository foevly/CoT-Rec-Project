# make_processed_mindsmall.py
import os, re, json
import pandas as pd

DATASET = "MINDsmall"
ROOT = "data"
os.makedirs("../CoT-Rec/llmehance/datasets/processed", exist_ok=True)
out_csv = f"llmehance/datasets/processed/{DATASET}_org.csv"

# 1) 建立 item 的 “id -> index” 映射（必须与嵌入行顺序一致）
# 我们在做嵌入时用的是：feature/caption 的 JSON 键（纯数字字符串）按数字升序
feat_json = f"GPT/input/feature/{DATASET}_feature.json"
cap_json  = f"GPT/input/caption/{DATASET}_caption.json"
src = feat_json if os.path.exists(feat_json) else cap_json
m = json.load(open(src, "r", encoding="utf-8"))
def nat_key(s):
    m = re.search(r"(\d+)$", s); return int(m.group(1)) if m else 0
item_ids = sorted(m.keys(), key=nat_key)
id2idx = {k:i for i, k in enumerate(item_ids)}
num_items = len(item_ids)
print(f"[OK] items={num_items}  (id->index built from {os.path.basename(src)})")

# 2) 读取 behaviors 并按时间构造每个用户的序列
cols = ["impid","uid","time","history","impressions"]
def read_beh(part):
    p = f"{ROOT}/{DATASET}_{part}/behaviors.tsv"
    return pd.read_csv(p, sep="\t", header=None, names=cols, dtype=str, na_filter=False, engine="python")
beh = pd.concat([read_beh("train"), read_beh("dev")], ignore_index=True)

# 每个用户的完整点击序列：把 history 拆开并按 time 排序（简化：history 已是时间序）
def split_hist(s):
    return [x for x in (s or "").split() if x.startswith("N")]

user_hist = {}
for _, row in beh.iterrows():
    uid = row["uid"]
    seq = split_hist(row["history"])
    user_hist.setdefault(uid, [])
    user_hist[uid].extend(seq)

# 映射到 item 索引（丢弃不在映射里的新闻）
def to_num(s):
    m = re.search(r"(\d+)$", s); return m.group(1) if m else None

proc = []
for uid, seq in user_hist.items():
    idx_seq = [id2idx[x] for x in (to_num(i) for i in seq) if x in id2idx]
    # 只保留长度>=3（train>=1，valid=1，test=1）
    if len(idx_seq) < 3:
        continue
    train_seq = idx_seq[:-2]
    valid = idx_seq[-2]
    test  = idx_seq[-1]
    # 用空格拼接成字符串，数据集类会再按 max_length 切窗
    proc.append({
        "user": uid,
        "seq": " ".join(map(str, train_seq)),
        "valid": valid,
        "test": test,
        "num_items": num_items
    })

df = pd.DataFrame(proc)
df.to_csv(out_csv, index=False)
print(f"[OK] saved -> {out_csv}  rows={len(df)}")


# # convert_seq_to_pairs.py
import re, pandas as pd

in_csv  = "llmehance/datasets/processed/MINDsmall_org.csv"
out_csv = in_csv  # 直接覆盖；若想保留，改成别的文件名

df = pd.read_csv(in_csv)
assert {"user","seq","valid","test","num_items"}.issubset(df.columns), "CSV列名不匹配"

num_items = int(df["num_items"].iloc[0])

def uid_to_int(u):
    m = re.search(r"(\d+)$", str(u))
    return int(m.group(1)) if m else 0

rows = []
for _, r in df.iterrows():
    uid = uid_to_int(r["user"])
    # 序列+valid+test 合并为完整点击序列
    seq = [int(x) for x in str(r["seq"]).split() if str(x).strip().isdigit()]
    seq += [int(r["valid"]), int(r["test"])]
    for t, it in enumerate(seq):
        rows.append((uid, it, t))

pdf = pd.DataFrame(rows, columns=["user_id","item_id","time"])
# 自检：item 不应超界
mx = int(pdf["item_id"].max())
assert mx < num_items, f"item_id({mx}) 超出 num_items({num_items-1})，请检查索引一致性"

pdf.to_csv(out_csv, index=False)
print(f"[OK] saved {out_csv}  rows={len(pdf)}  users={pdf['user_id'].nunique()}  items_seen={pdf['item_id'].nunique()}  num_items={num_items}")

import pandas as pd

IN  = "llmehance/datasets/processed/MINDsmall_org.csv"
OUT = "llmehance/datasets/processed/MINDsmall.csv"

df = pd.read_csv(IN, encoding="utf-8-sig", dtype=str)
df.rename(columns=lambda c: c.strip(), inplace=True)

# 统计并保留交互数 >=5 的用户（不改原始 id）
cnt = df.groupby("user_id")["item_id"].size()
keep_users = set(cnt[cnt >= 5].index)
out = df[df["user_id"].isin(keep_users)][["user_id", "item_id"]].copy()

# ------- 这里把 user_id 连续化：0..N-1 -------
uids = out["user_id"].astype(str).unique()                 # 保留顺序
uid2new = pd.Series(range(len(uids)), index=uids)          # old->new
out["user_id"] = out["user_id"].map(uid2new).astype("int64")

# item_id 不要改动（它已经是 0..num_items-1，与 .npy 对齐）
# 可做个断言以防万一：
min_item = pd.to_numeric(out["item_id"], errors="coerce").min()
assert int(min_item) == 0, f"item_id 最小值不是 0，请检查上游 id2idx 构建"

# 强制类型并保存
out["item_id"] = pd.to_numeric(out["item_id"], errors="coerce").astype("int64")
out.to_csv(OUT, index=False)

print(f"[OK] saved -> {OUT}")
print(f"users kept: {out['user_id'].nunique()}  rows: {len(out)}")

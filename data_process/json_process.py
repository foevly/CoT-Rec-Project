import os, re, json, numpy as np, pandas as pd

DATASET = "MINDsmall"
ROOT = "data"
PARTS_ALL  = [f"{DATASET}_train", f"{DATASET}_dev"]
PARTS_USER = [f"{DATASET}_train", f"{DATASET}_dev"]

os.makedirs("../CoT-Rec/GPT/input/caption", exist_ok=True)
os.makedirs("../CoT-Rec/GPT/input/feature", exist_ok=True)
os.makedirs("../CoT-Rec/GPT/input/user", exist_ok=True)

news_cols = ["nid","category","subcategory","title","abstract","url","title_entities","abstract_entities"]

def clean_str(x):
    if x is None: return ""
    if isinstance(x, float) and np.isnan(x): return ""
    s = str(x).strip()
    return "" if s.lower() in {"<unset>", "<null>", "nan", "none"} else s

def to_num_id(s: str) -> int | None:
    """把 'N12345' / 'U678' 这样的 ID 转成纯数字 12345 / 678；不匹配则返回 None。"""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    m = re.search(r'(\d+)$', s.strip())
    return int(m.group(1)) if m else None

# ---------- 1) news.tsv -> caption / feature ----------
news_frames = []
for p in PARTS_ALL:
    df = pd.read_csv(
        os.path.join(ROOT, p, "news.tsv"),
        sep="\t", header=None, names=news_cols,
        dtype=str, na_filter=False, engine="python", quoting=3, on_bad_lines="skip"
    )
    df = df[["nid","title","abstract"]].drop_duplicates("nid").copy()
    df["nid_num"]   = df["nid"].map(to_num_id)
    df["title"]     = df["title"].map(clean_str)
    df["abstract"]  = df["abstract"].map(clean_str)
    df = df[df["nid_num"].notna()]                # 只保留能转成数字的
    news_frames.append(df)

news = pd.concat(news_frames, ignore_index=True).drop_duplicates("nid_num")

# caption：用 title（键用数字字符串）
caption_map = {str(int(r.nid_num)): r.title
               for r in news.itertuples(index=False) if r.title}

# feature：用 abstract（为空时回退 title）
feature_map = {str(int(r.nid_num)): (r.abstract if r.abstract else r.title)
               for r in news.itertuples(index=False) if (r.abstract or r.title)}

with open(f"CoT-Rec/GPT/input/caption/{DATASET}_caption.json", "w", encoding="utf-8") as f:
    json.dump(caption_map, f, ensure_ascii=False)
with open(f"CoT-Rec/GPT/input/feature/{DATASET}_feature.json", "w", encoding="utf-8") as f:
    json.dump(feature_map, f, ensure_ascii=False)

print(f"[OK] caption: {len(caption_map)}  -> GPT/input/caption/{DATASET}_caption.json")
print(f"[OK] feature: {len(feature_map)}  -> GPT/input/feature/{DATASET}_feature.json")

# ---------- 2) behaviors.tsv -> user 偏好（uid 也转纯数字） ----------
b_cols = ["impid","uid","time","history","impressions"]

def hist_to_pref(hist: str, max_items=20, sep=" ; ") -> str:
    hist = clean_str(hist)
    if not hist: return ""
    titles = []
    for raw in hist.split():
        nid_num = to_num_id(raw)
        if nid_num is None:
            continue
        key = str(nid_num)
        t = caption_map.get(key) or feature_map.get(key) or ""
        if t:
            titles.append(t)
            if len(titles) >= max_items:
                break
    return sep.join(titles)

user_pref: dict[str, str] = {}
for p in PARTS_USER:
    b = pd.read_csv(
        os.path.join(ROOT, p, "behaviors.tsv"),
        sep="\t", header=None, names=b_cols,
        dtype=str, na_filter=False, engine="python", on_bad_lines="skip"
    )
    # 同一用户多行合并
    g = b.groupby("uid")["history"].apply(lambda s: " ".join(clean_str(x) for x in s)).reset_index()
    for _, row in g.iterrows():
        uid_num = to_num_id(row["uid"])
        if uid_num is None:
            continue
        uid_key = str(uid_num)  # JSON 键仍写成字符串，但内容是纯数字
        pref = hist_to_pref(row["history"])
        if not pref:
            continue
        if uid_key in user_pref and user_pref[uid_key]:
            user_pref[uid_key] = (user_pref[uid_key] + " ; " + pref)[:2000]
        else:
            user_pref[uid_key] = pref[:2000]

with open(f"CoT-Rec/GPT/input/user/{DATASET}_user.json", "w", encoding="utf-8") as f:
    json.dump(user_pref, f, ensure_ascii=False)

print(f"[OK] user: {len(user_pref)}  -> GPT/input/user/{DATASET}_user.json")

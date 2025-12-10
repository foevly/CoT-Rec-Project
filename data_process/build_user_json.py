import os
import json
import pickle
import argparse


def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_user_json(dataset_name: str,
                    valid_pkl: str,
                    test_pkl: str,
                    out_dir: str = "input/user"):
    """
    从 LLM_Reply_pickle 里的 valid / test pkl 中抽取 user_preferences，
    生成 input/user/{dataset_name}_user.json
    JSON 结构：{ user_id(int): "user_preferences 文本", ... }
    """
    # 1. 读取 pkl
    valid_data = load_pkl(valid_pkl)
    test_data = load_pkl(test_pkl)

    # 2. 合并两个 dict，按 user_id 去重
    user_texts = {}

    def collect(src):
        for uid, info in src.items():
            # uid 可能是 str 或 int，这里统一转成 int 作为 key
            int_uid = int(uid)
            text = (info.get("user_preferences") or "").strip()
            if not text:
                # 没有文本就跳过
                continue

            # 如果同一个用户在两个 pkl 里都有，就只保留第一次的，
            # 如果你想检查是否不一致，可以在这里加 warning
            if int_uid not in user_texts:
                user_texts[int_uid] = text

    collect(valid_data)
    collect(test_data)

    # 3. 按 user_id 排序，保证后面 npy 行号和 id 有序
    sorted_ids = sorted(user_texts.keys())
    ordered_dict = {uid: user_texts[uid] for uid in sorted_ids}

    # 4. 写出 json
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}_user.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ordered_dict, f, ensure_ascii=False, indent=2)

    print(f"✅ 已生成用户 json：{out_path}")
    print(f"   共 {len(ordered_dict)} 个用户")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="例如：Grocery_and_Gourmet_Food")
    parser.add_argument("--valid_pkl", type=str, required=True,
                        help="例如：LLM_Reply_pickle/Grocery_and_Gourmet_Food_valid.pkl")
    parser.add_argument("--test_pkl", type=str, required=True,
                        help="例如：LLM_Reply_pickle/Grocery_and_Gourmet_Food_test.pkl")
    args = parser.parse_args()

    build_user_json(
        dataset_name=args.dataset_name,
        valid_pkl=args.valid_pkl,
        test_pkl=args.test_pkl,
    )

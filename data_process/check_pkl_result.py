import pickle

with open("/Users/foevly/Desktop/DL/CoT-Rec-main/LLM_Reply_pickle/Grocery_and_Gourmet_Food_test.pkl", "rb") as f:
    data = pickle.load(f)

print(len(data))            # 看有多少条
first_key = next(iter(data.keys()))
print(first_key)
print(data[first_key].keys())
print(data[first_key]["user_preferences"])
print(list(data[first_key]["candidate_perception"].items())[:3])

with open("/Users/foevly/Desktop/DL/CoT-Rec-main/LLM_Reply_pickle/Grocery_and_Gourmet_Food_valid.pkl", "rb") as f:
    data = pickle.load(f)

print(len(data))            # 看有多少条
first_key = next(iter(data.keys()))
print(first_key)
print(data[first_key].keys())
print(data[first_key]["user_preferences"])
print(list(data[first_key]["candidate_perception"].items())[:3])
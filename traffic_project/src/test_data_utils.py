from data_utils import load_and_preprocess

X, y, scaler = load_and_preprocess("../data/milano_traffic_nid.csv")

print("X shape:", X.shape)
print("y shape:", y.shape)

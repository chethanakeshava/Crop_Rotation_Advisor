import pickle

with open("crop_rotation_model.pkl", "rb") as f:
    model = pickle.load(f)

print("✅ PKL loaded successfully")

result = model.predict_rotation(
    N=50,
    P=10,
    K=60,
    pH=5.4,
    rainfall=1111,
    temperature=28,
    input_season="Rabi"
)

print("\n📊 RESULT:")
print(result)

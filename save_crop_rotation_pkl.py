import pickle
import os
from crop_rotation_model import CropRotationModel

print("🚀 Creating CropRotationModel pickle...")

# Initialize model
rotation_model = CropRotationModel()

# File name
file_name = "crop_rotation_model.pkl"

# Save to pickle
with open(file_name, "wb") as f:
    pickle.dump(rotation_model, f)

print("✅ CropRotationModel saved successfully")
print("📍 Saved at:", os.path.abspath(file_name))

# CropRotationAdvisor

A machine learning-powered crop rotation recommendation system that analyzes agricultural data to suggest optimal crop sequences for sustainable farming practices. Includes data processing, model training, and prediction tools.

## Features

- Analyzes agricultural data with crop family classifications
- Trains a crop rotation model using machine learning
- Provides crop rotation recommendations
- Saves and tests trained models

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/chethanakeshava/Crop_Rotation_Advisor.git
   cd Crop_Rotation_Advisor
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the main script:
   ```
   python rotaion.py
   ```

2. Train the model:
   ```
   python save_crop_rotation_pkl.py
   ```

3. Test the model:
   ```
   python test_crop_rotation_pkl.py
   ```

4. Use the Jupyter notebook for recommendations:
   ```
   jupyter notebook croprecommend.ipynb
   ```

## Files

- `agricultural_data_with_crop_family.csv`: Dataset with agricultural data
- `crop_rotation_model.py`: Core model implementation
- `rotaion.py`: Main script for running recommendations
- `save_crop_rotation_pkl.py`: Script to train and save the model
- `test_crop_rotation_pkl.py`: Script to test the saved model
- `croprecommend.ipynb`: Jupyter notebook for interactive recommendations

## 👩‍💻 Author

**Chethana Keshava Shettigar**

AI & Data Science Student
Passionate about **Machine Learning, Data Science, and Web Development**


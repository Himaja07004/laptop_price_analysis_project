# Laptop Price Analysis and Prediction

This project leverages machine learning and data analysis techniques to predict laptop prices based on detailed specifications. Built with Python, it demonstrates data cleaning, feature engineering, exploratory analysis, regression modeling, and interactive prediction via a terminal application.

## Features

- Cleans and preprocesses real-world laptop datasets
- Extracts features like RAM, storage, brand, GPU, operating system, and spec rating
- Performs exploratory data analysis with visualizations
- Trains a random forest regression model for price prediction
- Accepts new laptop specifications as user input and predicts price
- Displays tabular output and models' error metrics

## Setup

1.**Install dependencies**
pip install numpy pandas matplotlib seaborn 
pip install scikit-learn plotly tabulate

2. **Run the main script**
python laptop_price_analysis.py

Follow prompts to input laptop features for prediction.

## Files

- `laptop_price_analysis.py` — Main Python script (edit as needed for feature selection).
- `laptop.csv` — Your laptop specification dataset (columns: brand, name, price, RAM, ROM, GPU, OS, etc.).
- `requirements.txt` — List of all Python package dependencies.
- `README.md` — Project overview and instructions.

## Model Notes

- Random Forest Regressor is the default model; you can easily swap for others in `laptop_price_analysis.py`.
- Include features like processor, GPU, OS, and spec_rating for better accuracy.
- Analyze and tune using feature importances and real row comparisons.

# test_app.py - Quick Test Script
import os
import pandas as pd

print("=== Testing File Structure ===")

# Check if dataset exists in data folder
if os.path.exists("data/Project 1 - Weather Dataset.csv"):
    print("✅ Dataset found in data/ folder")
    df = pd.read_csv("data/Project 1 - Weather Dataset.csv")
    print(f"✅ Dataset loaded: {len(df)} rows")
    print(f"✅ Columns: {list(df.columns)}")
else:
    print("❌ Dataset NOT found in data/ folder")
    print("Current directory:", os.getcwd())
    print("Files in current directory:", os.listdir())

# Check if weather_prediction_app.py exists
if os.path.exists("weather_prediction_app.py"):
    print("✅ Streamlit app file found")
else:
    print("❌ Streamlit app file NOT found")

print("\n=== Test Complete ===")
import pandas as pd
import os

file_path = "data/raw_data"
def load_data(file_path):
    if file_path:
        try:
            data = pd.read_csv(file_path)
            return data
        
        except FileNotFoundError:
            print("File not found. Please check the file path and try again.")
            
def main():
    # file_path = input("Enter the file path: ")
    data = load_data(file_path)
    print("Data loaded successfully!")
    print(data.head())
    print(f"Data shape: {data.shape}")
    data.info()
    print(data.describe())
    print(data.isnull().sum())
    
    os.makedirs("data", exist_ok=True)
    data.to_csv("data/ingested_data.csv", index=False)
    
    
if __name__ == "__main__":
    main()
        
        
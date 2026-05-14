import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def main():
    # 1. Load the Data
    print("Loading data...")
    url = "https://raw.githubusercontent.com/jonygeta/Data605FinalProject/master/train.csv"
    try:
        df = pd.read_csv(url)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Select Features and Target
    # Features:
    # GrLivArea: Above grade (ground) living area square feet (Size)
    # BedroomAbvGr: Bedrooms above grade (does NOT include basement bedrooms)
    # Neighborhood: Physical locations within Ames city limits
    features = ['GrLivArea', 'BedroomAbvGr', 'Neighborhood']
    target = 'SalePrice'
    
    # Drop rows where target or features are missing
    df = df.dropna(subset=features + [target])

    X = df[features]
    y = df[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Preprocessing
    print("Preprocessing data...")
    # Define numerical and categorical columns
    numeric_features = ['GrLivArea', 'BedroomAbvGr']
    categorical_features = ['Neighborhood']

    # Create transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Combine transformers into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 4. Model Training
    print("Training Gradient Boosting Regression model...")
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # 5. Evaluation
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print("-" * 30)
    print("Model Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print("-" * 30)

    # 6. Visualization
    print("Generating visualization...")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolor='w')
    
    # Plot a reference line for perfect predictions
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')

    plt.title('Actual vs. Predicted House Prices')
    plt.xlabel('Actual Sale Price ($)')
    plt.ylabel('Predicted Sale Price ($)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Save the plot
    output_image = 'actual_vs_predicted.png'
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Visualization saved as '{output_image}'")

if __name__ == "__main__":
    main()

House Price Prediction Walkthrough
I have successfully implemented the House Price Prediction task using the Ames Housing Dataset.

Overview
Dataset Loading: Retrieved the train.csv file from a publicly available GitHub repository.
Preprocessing:
Selected GrLivArea (Square Footage), BedroomAbvGr (Bedrooms), and Neighborhood (Location) as the primary features.
Handled categorical data (Neighborhood) using OneHotEncoder.
Scaled numerical features (GrLivArea, BedroomAbvGr) using StandardScaler.
Model Training: Trained a GradientBoostingRegressor to predict SalePrice.
Evaluation: Computed the following metrics on a 20% test split:
Mean Absolute Error (MAE): $24,794.78
Root Mean Squared Error (RMSE): $36,849.66
Visualization: Plotted the predicted vs. actual house prices to visualize the model's accuracy.
Results
Actual vs. Predicted Plot
Review
Actual vs. Predicted Plot

NOTE

The visualization shows how well the model predicts prices. Points closer to the red dashed line (the "perfect prediction" line) indicate a more accurate prediction.

Files Created
requirements.txt
 - The dependencies needed for the project.
train.py
 - The main script that performs loading, preprocessing, training, and visualization.
actual_vs_predicted.png
 - The generated scatter plot of actual vs predicted house prices.

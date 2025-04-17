import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

# Load the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def train_test_split_by_year(df, test_years=[2022, 2023]):
    """Split the data into training and test sets based on specific years"""
    train_df = df[~df['Year'].isin(test_years)]
    test_df = df[df['Year'].isin(test_years)]
    return train_df, test_df

def build_model(train_data, selected_features, target='CO2 Emissions (Tons/Capita)'):
    """Build a linear regression model with selected features"""
    X = train_data[selected_features]
    y = train_data[target]
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Get feature coefficients
    coefficients = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': model.coef_
    })
    
    return model, coefficients

def evaluate_model(model, test_data, selected_features, target='CO2 Emissions (Tons/Capita)'):
    """Evaluate the model on test data"""
    X_test = test_data[selected_features]
    y_test = test_data[target]
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Create results dataframe for plotting
    results_df = test_data.copy()
    results_df['Predicted CO2'] = y_pred
    results_df['Error'] = results_df['Predicted CO2'] - results_df[target]
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'results': results_df
    }

def plot_predictions(results_df, country=None, target='CO2 Emissions (Tons/Capita)'):
    """Plot actual vs predicted values with error bars"""
    if country:
        results_df = results_df[results_df['Country'] == country]
    
    plt.figure(figsize=(12, 6))
    
    # Plot actual and predicted values
    plt.subplot(1, 2, 1)
    plt.bar(results_df['Year'].astype(str), results_df[target], color='blue', alpha=0.6, label='Actual')
    plt.bar(results_df['Year'].astype(str), results_df['Predicted CO2'], color='red', alpha=0.6, label='Predicted')
    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions (Tons/Capita)')
    plt.title(f'CO2 Emissions: Actual vs Predicted ({country if country else "All Countries"})')
    plt.legend()
    
    # Plot errors
    plt.subplot(1, 2, 2)
    plt.bar(results_df['Year'].astype(str), results_df['Error'], color='orange')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Year')
    plt.ylabel('Error (Predicted - Actual)')
    plt.title('Prediction Error')
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(coefficients):
    """Plot the feature importance based on coefficients"""
    plt.figure(figsize=(10, 6))
    coefficients = coefficients.sort_values('Coefficient', key=abs, ascending=False)
    
    sns.barplot(x='Coefficient', y='Feature', data=coefficients)
    plt.title('Feature Importance (Coefficient Magnitude)')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.show()

def run_co2_prediction(file_path, selected_features=None, country_filter=None):
    """Main function to run the CO2 prediction model"""
    
    # Load data
    df = load_data(file_path)
    
    # Print the first few rows to verify data loading
    print("Data sample:")
    print(df.head())
    
    # List unique countries
    countries = df['Country'].unique()
    print(f"\nAvailable countries: {', '.join(countries)}")
    
    # If country filter is provided, filter the data
    if country_filter:
        if country_filter in countries:
            df = df[df['Country'] == country_filter]
            print(f"\nFiltered data for {country_filter}")
        else:
            print(f"Country '{country_filter}' not found. Using all data.")
    
    # If no features are specified, use all numeric columns except the target and Year
    if not selected_features:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = [col for col in numeric_columns if col not in ['CO2 Emissions (Tons/Capita)', 'Year']]
    
    print(f"\nSelected features: {', '.join(selected_features)}")
    
    # Split data
    train_df, test_df = train_test_split_by_year(df)
    print(f"\nTraining data: {train_df.shape[0]} rows")
    print(f"Test data: {test_df.shape[0]} rows")
    
    # Check if there's enough data
    if train_df.shape[0] < len(selected_features) + 1:
        print("Warning: Not enough training samples compared to the number of features.")
        print("Consider reducing the number of features or using more training data.")
    
    # Build model
    model, coefficients = build_model(train_df, selected_features)
    
    # Print coefficients
    print("\nModel Coefficients:")
    for feature, coef in zip(selected_features, model.coef_):
        print(f"{feature}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    
    # Evaluate model
    evaluation = evaluate_model(model, test_df, selected_features)
    
    # Print evaluation metrics
    print("\nModel Evaluation:")
    print(f"Mean Squared Error (MSE): {evaluation['mse']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {evaluation['rmse']:.4f}")
    print(f"Mean Absolute Error (MAE): {evaluation['mae']:.4f}")
    print(f"R² Score: {evaluation['r2']:.4f}")
    
    # Plot results
    plot_predictions(evaluation['results'], country=country_filter)
    
    # Plot feature importance
    plot_feature_importance(coefficients)
    
    return model, evaluation

# Example usage
if __name__ == "__main__":
    # You can modify these parameters
    csv_file = "climate_data.csv"  # Change to your CSV file path
    
    # Choose which columns to use for prediction (features)
    selected_features = [
        'Year',
        'Avg Temperature (°C)', 
        'Sea Level Rise (mm)', 
        'Rainfall (mm)',
        'Renewable Energy (%)', 
        'Extreme Weather Events'
    ]
    
    # Optionally filter for a specific country
    country_filter = "Argentina"  # Set to None to use all countries
    
    # Run the model
    model, evaluation = run_co2_prediction(csv_file, selected_features, country_filter)
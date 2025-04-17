import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler  # Added for normalization
import os
import json
from datetime import datetime


def create_output_directory(dir_name="xgboost_co2_results"):
    """Create the output directory for results."""
    os.makedirs(dir_name, exist_ok=True)
    print(f"Created output directory: {dir_name}")
    return dir_name


def save_figure(fig, filename, output_dir):
    """Save a matplotlib figure to the output directory."""
    file_path = os.path.join(output_dir, filename)
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {file_path}")
    return file_path


def load_and_explore_data(file_path):
    """Load and perform initial data exploration."""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded: {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit(1)

    # Print column names to verify structure
    print("Column names in dataset:")
    print(df.columns.tolist())

    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values)
    
    return df, missing_values


def save_data_summaries(df, missing_values, output_dir):
    """Save data summaries to output directory."""
    # Save a copy of the loaded data for reference
    data_summary_path = os.path.join(output_dir, "data_summary.csv")
    df.describe().to_csv(data_summary_path)
    print(f"Saved data summary to {data_summary_path}")

    # Save missing values report
    missing_values_path = os.path.join(output_dir, "missing_values.csv")
    pd.DataFrame(missing_values, columns=["Count"]).to_csv(missing_values_path)
    print(f"Saved missing values report to {missing_values_path}")

    # Save full dataset info
    dataset_info_path = os.path.join(output_dir, "full_dataset_info.txt")
    with open(dataset_info_path, 'w') as f:
        f.write(f"Dataset Shape: {df.shape}\n\n")
        f.write("Data Types:\n")
        f.write(str(df.dtypes))
        f.write("\n\nSample Data:\n")
        f.write(str(df.head(10)))
        f.write("\n\nDescriptive Statistics:\n")
        f.write(str(df.describe()))
    print(f"Saved full dataset info to {dataset_info_path}")


def handle_missing_values(df, missing_values):
    """Handle missing values in the dataframe."""
    columns_with_missing = missing_values[missing_values > 0].index.tolist()
    if columns_with_missing:
        print(f"\nColumns with missing values: {columns_with_missing}")
        print("Filling missing values with column means")
        for col in columns_with_missing:
            # Check if column is numeric before applying mean imputation
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                # For non-numeric columns, use most frequent value
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    else:
        print("No missing values found in the dataset")
    return df


def split_train_test(df):
    """Split data into training and testing sets based on years."""
    print("\nSplitting data into training and testing sets...")

    # Check if Year column exists
    if 'Year' not in df.columns:
        print("Warning: 'Year' column not found. Using last 15% of data for testing.")
        # Fallback to simple split if Year column doesn't exist
        split_idx = int(len(df) * 0.85)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        return train_df, test_df

    # Check if we have data for years 2022 and 2023
    year_counts = df['Year'].value_counts().sort_index()
    print("\nRows per year:")
    print(year_counts)

    # Use years 2022 and 2023 as test set
    test_years = [2022, 2023]
    train_df = df[~df['Year'].isin(test_years)]
    test_df = df[df['Year'].isin(test_years)]

    # Check if test set is too small, use only 2023 if needed
    if len(test_df) < 5:
        print("Warning: Test set is too small. Using only 2023 as test set.")
        test_years = [2023]
        train_df = df[~df['Year'].isin(test_years)]
        test_df = df[df['Year'].isin(test_years)]

    # If still too small, use last 15% of data (sorted by year)
    if len(test_df) < 5:
        print("Warning: Test set is still too small. Using last 15% of data (by year) for testing.")
        df = df.sort_values('Year')
        split_idx = int(len(df) * 0.85)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

    print(f"\nTraining set size: {train_df.shape[0]} rows")
    print(f"Testing set size: {test_df.shape[0]} rows")
    
    return train_df, test_df


def save_split_info(train_df, test_df, df, output_dir):
    """Save information about the train-test split."""
    split_info_path = os.path.join(output_dir, "train_test_split_info.txt")
    with open(split_info_path, 'w') as f:
        f.write(f"Training set size: {train_df.shape[0]} rows\n")
        f.write(f"Testing set size: {test_df.shape[0]} rows\n\n")
        
        if 'Year' in train_df.columns:
            f.write("Training set years: \n")
            f.write(str(sorted(train_df['Year'].unique())) + "\n\n")
            
            f.write("Testing set years: \n")
            f.write(str(sorted(test_df['Year'].unique())) + "\n\n")
        
        if 'Country' in df.columns:
            f.write("Countries in dataset: \n")
            f.write(str(sorted(df['Country'].unique())))
    print(f"Saved train-test split info to {split_info_path}")


def prepare_features(train_df, test_df):
    """Prepare features and target variables."""
    # Check for required columns before dropping
    target_col = 'CO2 Emissions (Tons/Capita)'
    drop_cols = [target_col]
    if 'Country' in train_df.columns:
        drop_cols.append('Country')
    
    # Verify target column exists
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
        
    X_train = train_df.drop(drop_cols, axis=1)
    y_train = train_df[target_col]
    X_test = test_df.drop(drop_cols, axis=1)
    y_test = test_df[target_col]
    
    return X_train, y_train, X_test, y_test


def normalize_features(X_train, X_test, output_dir):
    """Normalize features using StandardScaler."""
    print("\nNormalizing features...")
    
    # Create a scaler - using StandardScaler for z-score normalization
    scaler = StandardScaler()
    
    # Fit the scaler on the training data only
    scaler.fit(X_train)
    
    # Transform both training and test data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames to preserve feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Save scaling info
    scaling_info_path = os.path.join(output_dir, "feature_scaling_info.csv")
    scaling_info = pd.DataFrame({
        'Feature': X_train.columns,
        'Mean': scaler.mean_,
        'Standard Deviation': np.sqrt(scaler.var_)
    })
    scaling_info.to_csv(scaling_info_path, index=False)
    print(f"Saved feature scaling information to {scaling_info_path}")
    
    # Compare before and after normalization
    before_after_path = os.path.join(output_dir, "normalization_effect.txt")
    with open(before_after_path, 'w') as f:
        f.write("=== Feature Statistics Before Normalization ===\n")
        f.write(str(X_train.describe()) + "\n\n")
        f.write("=== Feature Statistics After Normalization ===\n")
        f.write(str(X_train_scaled.describe()) + "\n")
    print(f"Saved normalization effect analysis to {before_after_path}")
    
    return X_train_scaled, X_test_scaled, scaler


def save_feature_list(X_train, output_dir):
    """Save the list of features used in the model."""
    features_list_path = os.path.join(output_dir, "model_features.txt")
    with open(features_list_path, 'w') as f:
        f.write("Features used in model:\n")
        for feature in X_train.columns:
            f.write(f"- {feature}\n")
    print(f"Saved feature list to {features_list_path}")


def get_xgboost_params():
    """Define and return XGBoost hyperparameters."""
    return {
        'n_estimators': 500,            # More trees (dataset is small)
        'max_depth': 3,                 # Shallower trees prevent overfitting
        'learning_rate': 0.05,          # Lower learning rate for better convergence
        'subsample': 0.7,               # Slightly lower to add randomness
        'colsample_bytree': 0.7,        # Randomness in feature selection
        'gamma': 1,                     # Some regularization on splits
        'reg_alpha': 0.5,               # L1 regularization (sparsity)
        'reg_lambda': 1,                # L2 regularization
        'min_child_weight': 5,          # Minimum leaf node weight to prevent overfitting
        'objective': 'reg:squarederror',# Regression task
        'booster': 'gbtree',            # Classic boosted trees
        'tree_method': 'hist',          # Faster on small datasets
        'random_state': 42              # Reproducibility
    }


def save_hyperparameters(xgb_params, output_dir):
    """Save model hyperparameters to JSON file."""
    hyperparams_path = os.path.join(output_dir, "hyperparameters.json")
    with open(hyperparams_path, 'w') as f:
        json.dump(xgb_params, f, indent=4)
    print(f"Saved hyperparameters to {hyperparams_path}")


def train_model(X_train, y_train, xgb_params):
    """Create and train XGBoost model."""
    print("\nTraining XGBoost model...")
    model = xgb.XGBRegressor(**xgb_params)
    try:
        model.fit(X_train, y_train)
        print("Training completed!")
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        print("Trying with simpler parameters...")
        # Use simpler parameters if training fails
        simpler_params = {
            'n_estimators': 50,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42
        }
        model = xgb.XGBRegressor(**simpler_params)
        model.fit(X_train, y_train)
        print("Training completed with simpler parameters!")
    
    return model


def save_model(model, output_dir):
    """Save the trained model."""
    model_path = os.path.join(output_dir, "xgboost_co2_model.json")
    model.save_model(model_path)
    print(f"Saved model to {model_path}")


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate the model and return predictions and metrics."""
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate evaluation metrics for training data
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)

    # Calculate evaluation metrics for testing data
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)

    # Create metrics dictionary
    metrics = {
        "training": {
            "MAE": float(train_mae),
            "MSE": float(train_mse),
            "RMSE": float(train_rmse),
            "R²": float(train_r2)
        },
        "testing": {
            "MAE": float(test_mae),
            "MSE": float(test_mse),
            "RMSE": float(test_rmse),
            "R²": float(test_r2)
        }
    }
    
    return y_train_pred, y_test_pred, metrics


def save_metrics(metrics, output_dir):
    """Save model performance metrics."""
    metrics_path = os.path.join(output_dir, "model_performance_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_path}")


def print_metrics(metrics):
    """Print model evaluation metrics."""
    print("\n=== Model Performance ===")
    print(f"Training MAE: {metrics['training']['MAE']:.4f}")
    print(f"Training MSE: {metrics['training']['MSE']:.4f}")
    print(f"Training RMSE: {metrics['training']['RMSE']:.4f}")
    print(f"Training R²: {metrics['training']['R²']:.4f}")
    print("-" * 30)
    print(f"Testing MAE: {metrics['testing']['MAE']:.4f}")
    print(f"Testing MSE: {metrics['testing']['MSE']:.4f}")
    print(f"Testing RMSE: {metrics['testing']['RMSE']:.4f}")
    print(f"Testing R²: {metrics['testing']['R²']:.4f}")


def analyze_feature_importance(model, X_train, output_dir):
    """Analyze and save feature importance."""
    feature_importance = model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df = importance_df.sort_values('Importance', ascending=False)

    # Save feature importance
    importance_path = os.path.join(output_dir, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved feature importance to {importance_path}")

    # Print feature importance
    print("\n=== Feature Importance ===")
    for i, row in importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")
        
    return importance_df


def prepare_test_results(test_df, y_test, y_test_pred, output_dir):
    """Prepare and save test set results."""
    # Create a comprehensive results DataFrame for the test set
    test_results = pd.DataFrame()
    
    # Check if Country column exists
    if 'Country' in test_df.columns:
        test_results['Country'] = test_df['Country']
    
    # Check if Year column exists
    if 'Year' in test_df.columns:
        test_results['Year'] = test_df['Year']
    
    # Add actual and predicted values
    test_results['Actual CO2'] = y_test
    test_results['Predicted CO2'] = y_test_pred
    test_results['Error'] = test_results['Actual CO2'] - test_results['Predicted CO2']
    test_results['% Error'] = (test_results['Error'] / test_results['Actual CO2']) * 100

    # Save test predictions
    predictions_path = os.path.join(output_dir, "test_predictions.csv")
    test_results.to_csv(predictions_path, index=False)
    print(f"Saved test predictions to {predictions_path}")

    print("\n=== Test Set Predictions ===")
    print(test_results.head(10))
    
    return test_results


def create_visualizations(y_train, y_train_pred, y_test, y_test_pred, test_results, importance_df, df, X_train, output_dir):
    """Create and save visualizations."""
    print("\nCreating visualizations...")
    fig_size = (12, 8)

    try:
        # 1. Actual vs Predicted plot
        plt.figure(figsize=fig_size)
        plt.scatter(y_train, y_train_pred, alpha=0.5, color='blue', label='Training')
        plt.scatter(y_test, y_test_pred, alpha=0.5, color='red', label='Testing')

        # Plot the perfect prediction line
        min_val = min(min(y_train), min(y_test))
        max_val = max(max(y_train), max(y_test))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')

        plt.xlabel('Actual CO2 Emissions (Tons/Capita)')
        plt.ylabel('Predicted CO2 Emissions (Tons/Capita)')
        plt.title('Actual vs Predicted CO2 Emissions')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_figure(plt.gcf(), 'actual_vs_predicted.png', output_dir)
        plt.close()
        print("Created actual vs predicted plot")
    except Exception as e:
        print(f"Error creating Actual vs Predicted plot: {str(e)}")

    try:
        # 2. Feature Importance Visualization
        plt.figure(figsize=fig_size)
        # Get top N features (or all if fewer than 10)
        top_n = min(10, len(importance_df))
        top_features = importance_df.head(top_n)
        
        # Use matplotlib's bar plot
        features = top_features['Feature'].values
        importances = top_features['Importance'].values
        y_pos = np.arange(len(features))
        
        plt.barh(y_pos, importances, align='center')
        plt.yticks(y_pos, features)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()

        save_figure(plt.gcf(), 'feature_importance.png', output_dir)
        plt.close()
        print("Created feature importance plot")
    except Exception as e:
        print(f"Error creating Feature Importance plot: {str(e)}")

    try:
        # 3. Test set predictions by year and country
        if 'Year' in test_results.columns and 'Country' in test_results.columns:
            plt.figure(figsize=fig_size)
            
            # Only plot up to 5 countries to avoid overcrowding
            countries = test_results['Country'].unique()
            countries_to_plot = countries[:min(5, len(countries))]
            
            for country in countries_to_plot:
                country_data = test_results[test_results['Country'] == country]
                plt.plot(country_data['Year'], country_data['Actual CO2'], 'o-', label=f'{country} Actual')
                plt.plot(country_data['Year'], country_data['Predicted CO2'], 's--', label=f'{country} Predicted')

            plt.xlabel('Year')
            plt.ylabel('CO2 Emissions (Tons/Capita)')
            plt.title('Test Set: Actual vs Predicted CO2 Emissions by Country and Year')
            plt.legend(loc='best', fontsize='small')
            plt.grid(True)
            plt.tight_layout()

            save_figure(plt.gcf(), 'test_predictions_by_year.png', output_dir)
            plt.close()
            print("Created predictions by year plot")
        else:
            print("Skipping predictions by year plot - missing Year or Country column")
    except Exception as e:
        print(f"Error creating test predictions by year plot: {str(e)}")

    try:
        # 4. Error Distribution
        plt.figure(figsize=fig_size)
        plt.hist(test_results['Error'], bins=10, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Prediction Error (Tons/Capita)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors on Test Set')
        plt.grid(True)
        plt.tight_layout()

        save_figure(plt.gcf(), 'error_distribution.png', output_dir)
        plt.close()
        print("Created error distribution plot")
    except Exception as e:
        print(f"Error creating error distribution plot: {str(e)}")

    try:
        # 5. Residual plot
        plt.figure(figsize=fig_size)
        plt.scatter(y_test_pred, test_results['Error'])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted CO2 Emissions (Tons/Capita)')
        plt.ylabel('Residual (Actual - Predicted)')
        plt.title('Residual Plot for Test Set')
        plt.grid(True)
        plt.tight_layout()

        save_figure(plt.gcf(), 'residual_plot.png', output_dir)
        plt.close()
        print("Created residual plot")
    except Exception as e:
        print(f"Error creating residual plot: {str(e)}")

    try:
        # 6. Correlation heatmap
        # Check if we have too many features for a readable heatmap
        if len(X_train.columns) > 20:
            # If too many features, only use top 20 most important features
            top_features = importance_df['Feature'].head(20).tolist()
            corr_features = top_features + ['CO2 Emissions (Tons/Capita)']
            print(f"Too many features for heatmap. Using only top 20 important features.")
        else:
            # Include all features and target
            corr_features = X_train.columns.tolist() + ['CO2 Emissions (Tons/Capita)']
        
        # Create correlation matrix if all required columns exist
        if all(col in df.columns for col in corr_features):
            plt.figure(figsize=fig_size)
            correlation = df[corr_features].corr()
            
            # Create heatmap using imshow for better compatibility
            plt.imshow(correlation, cmap='coolwarm', interpolation='none', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(correlation)), correlation.columns, rotation=90)
            plt.yticks(range(len(correlation)), correlation.columns)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()

            save_figure(plt.gcf(), 'correlation_heatmap.png', output_dir)
            plt.close()
            print("Created correlation heatmap")
        else:
            print("Skipping correlation heatmap - some required columns missing")
    except Exception as e:
        print(f"Error creating correlation heatmap: {str(e)}")
        
    # 7. New plot: Before and After Normalization Distribution
    try:
        plt.figure(figsize=(14, 10))
        
        # Get top 4 most important features
        top_features = importance_df['Feature'].head(4)['Feature'].tolist()
        
        for i, feature in enumerate(top_features, 1):
            plt.subplot(2, 2, i)
            
            # Plot original distribution
            sns.histplot(X_train[feature], kde=True, color='blue', alpha=0.5, label='Original')
            
            # Scale to get back normalized values (just for visualization)
            # This is just to show the normalized distribution on the same plot
            normalized_feature = (X_train[feature] - X_train[feature].mean()) / X_train[feature].std()
            
            # Plot normalized distribution
            sns.histplot(normalized_feature, kde=True, color='red', alpha=0.5, label='Normalized')
            
            plt.title(f'Distribution of {feature} Before/After Normalization')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_figure(plt.gcf(), 'normalization_distribution.png', output_dir)
        plt.close()
        print("Created normalization distribution plot")
    except Exception as e:
        print(f"Error creating normalization distribution plot: {str(e)}")


def create_summary_report(df, train_df, test_df, X_train, model, metrics, importance_df, output_dir, scaler=None):
    """Create and save a summary report."""
    summary_report_path = os.path.join(output_dir, "summary_report.txt")
    with open(summary_report_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("CO2 EMISSIONS PREDICTION MODEL SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Date generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATA SUMMARY:\n")
        f.write("-"*50 + "\n")
        f.write(f"Total rows: {df.shape[0]}\n")
        f.write(f"Training rows: {train_df.shape[0]}\n")
        f.write(f"Testing rows: {test_df.shape[0]}\n")
        f.write(f"Features used: {len(X_train.columns)}\n\n")
        
        f.write("FEATURE PREPROCESSING:\n")
        f.write("-"*50 + "\n")
        f.write("Feature normalization: Applied (StandardScaler)\n")
        if scaler is not None:
            f.write("Mean values used for normalization (first 5):\n")
            for i, (feat, mean_val) in enumerate(zip(X_train.columns[:5], scaler.mean_[:5])):
                f.write(f"- {feat}: {mean_val:.4f}\n")
            f.write("Standard deviations used for normalization (first 5):\n")
            for i, (feat, std_val) in enumerate(zip(X_train.columns[:5], np.sqrt(scaler.var_)[:5])):
                f.write(f"- {feat}: {std_val:.4f}\n")
        f.write("\n")
        
        f.write("MODEL HYPERPARAMETERS:\n")
        f.write("-"*50 + "\n")
        for param, value in model.get_params().items():
            f.write(f"{param}: {value}\n")
        f.write("\n")
        
        f.write("MODEL PERFORMANCE:\n")
        f.write("-"*50 + "\n")
        f.write("Training Metrics:\n")
        f.write(f"MAE: {metrics['training']['MAE']:.4f}\n")
        f.write(f"MSE: {metrics['training']['MSE']:.4f}\n")
        f.write(f"RMSE: {metrics['training']['RMSE']:.4f}\n")
        f.write(f"R²: {metrics['training']['R²']:.4f}\n\n")
        
        f.write("Testing Metrics:\n")
        f.write(f"MAE: {metrics['testing']['MAE']:.4f}\n")
        f.write(f"MSE: {metrics['testing']['MSE']:.4f}\n")
        f.write(f"RMSE: {metrics['testing']['RMSE']:.4f}\n")
        f.write(f"R²: {metrics['testing']['R²']:.4f}\n\n")
        
        f.write("TOP 5 MOST IMPORTANT FEATURES:\n")
        f.write("-"*50 + "\n")
        for i, row in importance_df.head(min(5, len(importance_df))).iterrows():
            f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
        f.write("\n")
        
        f.write("GENERATED ARTIFACTS:\n")
        f.write("-"*50 + "\n")
        f.write("1. Data summary (data_summary.csv)\n")
        f.write("2. Missing values report (missing_values.csv)\n")
        f.write("3. Feature scaling information (feature_scaling_info.csv)\n")
        f.write("4. Feature importance ranking (feature_importance.csv)\n")
        f.write("5. Test predictions (test_predictions.csv)\n")
        f.write("6. Trained model (xgboost_co2_model.json)\n")
        f.write("7. Various visualizations (PNG files)\n")
        
    print(f"Saved summary report to {summary_report_path}")


def main():
    """Main function to run the entire analysis pipeline."""
    # Setup
    output_dir = create_output_directory()
    file_path = "processed_climate_change_dataset.csv"  # Complete the file path
    
    # Data loading and exploration
    df, missing_values = load_and_explore_data(file_path)
    save_data_summaries(df, missing_values, output_dir)
    
    # Data preprocessing
    df = handle_missing_values(df, missing_values)
    train_df, test_df = split_train_test(df)
    save_split_info(train_df, test_df, df, output_dir)
    
    # Feature preparation
    X_train, y_train, X_test, y_test = prepare_features(train_df, test_df)
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test, output_dir)
    save_feature_list(X_train, output_dir)
    
    # Model training
    xgb_params = get_xgboost_params()
    save_hyperparameters(xgb_params, output_dir)
    model = train_model(X_train_scaled, y_train, xgb_params)
    save_model(model, output_dir)
    
    # Model evaluation
    y_train_pred, y_test_pred, metrics = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    save_metrics(metrics, output_dir)
    print_metrics(metrics)
    
    # Analysis and visualization
    importance_df = analyze_feature_importance(model, X_train, output_dir)
    test_results = prepare_test_results(test_df, y_test, y_test_pred, output_dir)
    create_visualizations(y_train, y_train_pred, y_test, y_test_pred, test_results, importance_df, df, X_train, output_dir)
    
    # Summary report
    create_summary_report(df, train_df, test_df, X_train, model, metrics, importance_df, output_dir, scaler)
    
    print("\nAnalysis pipeline completed successfully!")
    print(f"All results saved to directory: {output_dir}")


if __name__ == "__main__":
    main()  
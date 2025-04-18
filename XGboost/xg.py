import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
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


def encode_country_column(train_df, test_df):
    """Encode the country column to integer values using LabelEncoder."""
    print("\nEncoding country column...")
    
    # Verify Country column exists
    if 'Country' not in train_df.columns:
        print("Warning: Country column not found. Skipping encoding.")
        return train_df, test_df, None
    
    # Create label encoder
    label_encoder = LabelEncoder()
    
    # Fit on all countries from both train and test sets to ensure consistency
    all_countries = pd.concat([train_df['Country'], test_df['Country']]).unique()
    label_encoder.fit(all_countries)
    
    # Create new encoded column
    train_df['Country_Code'] = label_encoder.transform(train_df['Country'])
    test_df['Country_Code'] = label_encoder.transform(test_df['Country'])
    
    # Save mapping for reference
    country_mapping = {country: code for country, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
    
    print(f"Encoded {len(country_mapping)} unique countries")
    print("Sample mappings:")
    # Print first 5 country mappings
    for i, (country, code) in enumerate(country_mapping.items()):
        if i >= 5:
            break
        print(f"  {country} -> {code}")
    
    return train_df, test_df, country_mapping


def save_country_mapping(country_mapping, output_dir):
    """Save country to integer mapping."""
    if country_mapping is None:
        return
    
    mapping_path = os.path.join(output_dir, "country_code_mapping.csv")
    mapping_df = pd.DataFrame([
        {'Country': country, 'Code': code} 
        for country, code in country_mapping.items()
    ])
    mapping_df.sort_values('Code').to_csv(mapping_path, index=False)
    print(f"Saved country code mapping to {mapping_path}")


def prepare_features(train_df, test_df, use_country_encoding=False):
    """Prepare features and target variables."""
    # Check for required columns before dropping
    target_col = 'CO2 Emissions (Tons/Capita)'
    drop_cols = [target_col,'Sea Level Rise (mm)']
    
    # Always drop Country string column
    if 'Country' in train_df.columns:
        drop_cols.append('Country')
    
    # If country encoding is disabled, drop the Country_Code column if it exists
    if not use_country_encoding and 'Country_Code' in train_df.columns:
        drop_cols.append('Country_Code')
        print("Not using country encoding - dropping Country_Code column")
    elif use_country_encoding and 'Country_Code' in train_df.columns:
        print("Using country encoding - keeping Country_Code column as a feature")
    
    # Verify target column exists
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Create feature sets
    X_train = train_df.drop(drop_cols, axis=1)
    y_train = train_df[target_col]
    X_test = test_df.drop(drop_cols, axis=1)
    y_test = test_df[target_col]
    
    print(f"\nFeatures used in model: {X_train.columns.tolist()}")
    
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
    return  {
        'n_estimators': 300,            # More trees (dataset is small)
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


def save_model(model, output_dir, suffix=""):
    """Save the trained model."""
    model_path = os.path.join(output_dir, f"xgboost_co2_model{suffix}.json")
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


def save_metrics(metrics, output_dir, suffix=""):
    """Save model performance metrics."""
    metrics_path = os.path.join(output_dir, f"model_performance_metrics{suffix}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_path}")


def print_metrics(metrics, title="Model Performance"):
    """Print model evaluation metrics."""
    print(f"\n=== {title} ===")
    print(f"Training MAE: {metrics['training']['MAE']:.4f}")
    print(f"Training MSE: {metrics['training']['MSE']:.4f}")
    print(f"Training RMSE: {metrics['training']['RMSE']:.4f}")
    print(f"Training R²: {metrics['training']['R²']:.4f}")
    print("-" * 30)
    print(f"Testing MAE: {metrics['testing']['MAE']:.4f}")
    print(f"Testing MSE: {metrics['testing']['MSE']:.4f}")
    print(f"Testing RMSE: {metrics['testing']['RMSE']:.4f}")
    print(f"Testing R²: {metrics['testing']['R²']:.4f}")


def analyze_feature_importance(model, X_train, output_dir, suffix=""):
    """Analyze and save feature importance."""
    feature_importance = model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df = importance_df.sort_values('Importance', ascending=False)

    # Save feature importance
    importance_path = os.path.join(output_dir, f"feature_importance{suffix}.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved feature importance to {importance_path}")

    # Print feature importance
    print(f"\n=== Feature Importance {suffix} ===")
    for i, row in importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")
        
    return importance_df


def prepare_test_results(test_df, y_test, y_test_pred, output_dir, suffix=""):
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
    predictions_path = os.path.join(output_dir, f"test_predictions{suffix}.csv")
    test_results.to_csv(predictions_path, index=False)
    print(f"Saved test predictions to {predictions_path}")

    print(f"\n=== Test Set Predictions {suffix} ===")
    print(test_results.head(10))
    
    return test_results


def create_visualizations(y_train, y_train_pred, y_test, y_test_pred, test_results, importance_df, df, X_train, output_dir, suffix=""):
    """Create and save visualizations."""
    print(f"\nCreating visualizations{suffix}...")
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

        save_figure(plt.gcf(), f'actual_vs_predicted{suffix}.png', output_dir)
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

        save_figure(plt.gcf(), f'feature_importance{suffix}.png', output_dir)
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

            save_figure(plt.gcf(), f'test_predictions_by_year{suffix}.png', output_dir)
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

        save_figure(plt.gcf(), f'error_distribution{suffix}.png', output_dir)
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

        save_figure(plt.gcf(), f'residual_plot{suffix}.png', output_dir)
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

            save_figure(plt.gcf(), f'correlation_heatmap{suffix}.png', output_dir)
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
        save_figure(plt.gcf(), f'normalization_distribution{suffix}.png', output_dir)
        plt.close()
        print("Created normalization distribution plot")
    except Exception as e:
        print(f"Error creating normalization distribution plot: {str(e)}")


def create_summary_report(df, train_df, test_df, X_train, model, metrics, importance_df, output_dir, scaler=None, suffix="", use_country_encoding=False):
    """Create and save a summary report."""
    summary_report_path = os.path.join(output_dir, f"summary_report{suffix}.txt")
    with open(summary_report_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("CO2 EMISSIONS PREDICTION MODEL SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Date generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Country encoding used: {'Yes' if use_country_encoding else 'No'}\n\n")
        
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
        f.write(f"1. Data summary (data_summary.csv)\n")
        f.write(f"2. Missing values report (missing_values.csv)\n")
        f.write(f"3. Feature scaling information (feature_scaling_info.csv)\n")
        f.write(f"4. Feature importance ranking (feature_importance{suffix}.csv)\n")
        f.write(f"5. Test predictions (test_predictions{suffix}.csv)\n")
        f.write(f"6. Trained model (xgboost_co2_model{suffix}.json)\n")
        f.write(f"7. Various visualizations (PNG files)\n")
        
    print(f"Saved summary report to {summary_report_path}")


def run_model_with_country_encoding(df, output_dir):
    """Run the entire model pipeline with country encoding."""
    print("\n\n=== RUNNING MODEL WITH COUNTRY ENCODING ===")
    suffix = "_with_country"
    
    # Data preprocessing
    df_clean = handle_missing_values(df.copy(), df.isnull().sum())
    train_df, test_df = split_train_test(df_clean)
    train_df, test_df, country_mapping = encode_country_column(train_df, test_df)
    save_country_mapping(country_mapping, output_dir)
    
    # Use the country encoding for this version
    use_country_encoding = True
    X_train, y_train, X_test, y_test = prepare_features(train_df, test_df, use_country_encoding)
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test, output_dir)
    
    # Model training
    xgb_params = get_xgboost_params()
    model = train_model(X_train_scaled, y_train, xgb_params)
    save_model(model, output_dir, suffix)
    
    # Model evaluation
    y_train_pred, y_test_pred, metrics = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    save_metrics(metrics, output_dir, suffix)
    print_metrics(metrics, "Model Performance (With Country Encoding)")
    
    # Analysis and visualization
    importance_df = analyze_feature_importance(model, X_train, output_dir, suffix)
    test_results = prepare_test_results(test_df, y_test, y_test_pred, output_dir, suffix)
    create_visualizations(y_train, y_train_pred, y_test, y_test_pred, test_results, importance_df, df_clean, X_train, output_dir, suffix)
    
    # Summary report
    create_summary_report(df_clean, train_df, test_df, X_train, model, metrics, importance_df, output_dir, scaler, suffix, use_country_encoding)
    
    return metrics


def perform_hyperparameter_search(X_train, y_train, X_test, y_test, output_dir, search_type='random', n_iter=20, cv=5):
    """
    Perform hyperparameter search for XGBoost model.
    
    Parameters:
    -----------
    X_train, y_train : training data
    X_test, y_test : testing data for final evaluation
    output_dir : directory to save results
    search_type : 'random' for RandomizedSearchCV or 'grid' for GridSearchCV
    n_iter : number of iterations for random search
    cv : number of cross-validation folds
    
    Returns:
    --------
    best_model : trained XGBoost model with best hyperparameters
    best_params : dictionary of best hyperparameters
    search_results : DataFrame with search results
    """
    import xgboost as xgb
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import json
    import os
    from datetime import datetime
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    import time
    
    print(f"\nPerforming {search_type} hyperparameter search...")
    start_time = time.time()
    
    # Define parameter grid for search
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [2, 3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'gamma': [0, 0.5, 1, 2],
        'min_child_weight': [1, 3, 5, 7],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.1, 0.5, 1.0, 2.0]
    }
    
    # Base XGBoost model
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        booster='gbtree',
        tree_method='hist',
        random_state=42
    )
    
    # Create appropriate search CV
    if search_type.lower() == 'random':
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='neg_root_mean_squared_error',
            cv=cv,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        search_name = f"RandomizedSearchCV (n_iter={n_iter}, cv={cv})"
    else:
        # For grid search, reduce parameter space to avoid combinatorial explosion
        reduced_param_grid = {
            'n_estimators': [100, 300],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.7, 0.9],
            'colsample_bytree': [0.7, 0.9],
            'min_child_weight': [1, 5]
        }
        search = GridSearchCV(
            estimator=base_model,
            param_grid=reduced_param_grid,
            scoring='neg_root_mean_squared_error',
            cv=cv,
            verbose=1,
            n_jobs=-1
        )
        search_name = f"GridSearchCV (cv={cv})"
    
    # Fit the search
    try:
        search.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during hyperparameter search: {str(e)}")
        print("Trying with reduced parameter space...")
        
        # Fallback with minimal parameter space
        minimal_param_grid = {
            'n_estimators': [100, 300],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
        
        if search_type.lower() == 'random':
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=minimal_param_grid,
                n_iter=min(n_iter, 6),  # Reduce iterations for smaller space
                scoring='neg_root_mean_squared_error',
                cv=cv,
                verbose=1,
                random_state=42,
                n_jobs=-1
            )
        else:
            search = GridSearchCV(
                estimator=base_model,
                param_grid=minimal_param_grid,
                scoring='neg_root_mean_squared_error',
                cv=cv,
                verbose=1,
                n_jobs=-1
            )
        
        search.fit(X_train, y_train)
    
    # Record execution time
    execution_time = time.time() - start_time
    
    # Get best parameters and model
    best_params = search.best_params_
    best_model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
    best_model.fit(X_train, y_train)
    
    # Extract and format search results
    cv_results = search.cv_results_
    search_results = pd.DataFrame()
    
    # Add relevant columns
    for key in cv_results.keys():
        if key.startswith('param_') or key in ['mean_test_score', 'std_test_score', 'rank_test_score', 'mean_fit_time']:
            if key.startswith('param_'):
                param_name = key.replace('param_', '')
                search_results[param_name] = cv_results[key]
            else:
                search_results[key] = cv_results[key]
    
    # Convert negative RMSE back to positive for clarity
    search_results['mean_test_score'] = -search_results['mean_test_score']
    search_results.rename(columns={'mean_test_score': 'mean_test_rmse'}, inplace=True)
    
    # Sort by performance
    search_results = search_results.sort_values('mean_test_rmse')
    
    # Save search results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"hyperparameter_search_results_{timestamp}.csv")
    search_results.to_csv(results_path, index=False)
    print(f"Saved search results to {results_path}")
    
    # Save best parameters
    best_params_path = os.path.join(output_dir, f"best_hyperparameters_{timestamp}.json")
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"Saved best parameters to {best_params_path}")
    
    # Create visualization of parameter importance
    try:
        create_parameter_importance_plot(search_results, output_dir, timestamp)
    except Exception as e:
        print(f"Error creating parameter importance plot: {str(e)}")
    
    # Create summary report
    summary_path = os.path.join(output_dir, f"hyperparameter_search_summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("HYPERPARAMETER SEARCH SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Method: {search_name}\n")
        f.write(f"Execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)\n\n")
        
        f.write("BEST PARAMETERS:\n")
        f.write("-"*50 + "\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
        
        f.write("\nTOP 5 PARAMETER COMBINATIONS:\n")
        f.write("-"*50 + "\n")
        top_5 = search_results.head(5)
        for i, row in top_5.iterrows():
            f.write(f"Rank {int(row['rank_test_score'])}: RMSE = {row['mean_test_rmse']:.4f}\n")
            for param in [col for col in top_5.columns if not col.startswith(('mean_', 'std_', 'rank_'))]:
                f.write(f"  - {param}: {row[param]}\n")
            f.write("\n")
    
    print(f"Saved search summary to {summary_path}")
    print(f"Best RMSE during CV: {-search.best_score_:.4f}")
    print(f"Best parameters: {best_params}")
    
    return best_model, best_params, search_results


def create_parameter_importance_plot(search_results, output_dir, timestamp):
    """
    Create and save visualization showing parameter importance based on search results.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    print("Creating parameter importance visualization...")
    
    # Identify parameter columns
    param_cols = [col for col in search_results.columns if not col.startswith(('mean_', 'std_', 'rank_'))]
    
    # Prepare figure
    plt.figure(figsize=(12, 8))
    
    # Calculate parameter importance for each parameter
    param_importance = {}
    
    for param in param_cols:
        # Skip parameters with non-numeric values
        try:
            # Check if all values can be converted to float
            search_results[param].astype(float)
            
            # Calculate correlation with RMSE
            correlation = np.corrcoef(search_results[param].astype(float), 
                                     search_results['mean_test_rmse'])[0, 1]
            
            # Use absolute correlation as importance
            param_importance[param] = abs(correlation)
        except:
            # For categorical parameters, use ANOVA or similar technique
            unique_values = search_results[param].nunique()
            if unique_values > 1 and unique_values < 10:  # Skip if too many categories
                # Calculate variance between groups
                group_means = search_results.groupby(param)['mean_test_rmse'].mean()
                overall_mean = search_results['mean_test_rmse'].mean()
                
                # Calculate weighted variance as a simple importance metric
                between_group_var = sum([(group_mean - overall_mean)**2 * count 
                                        for (_, count), group_mean in zip(
                                            search_results[param].value_counts().items(), 
                                            group_means)])
                
                param_importance[param] = between_group_var / unique_values
            else:
                param_importance[param] = 0
    
    # Sort parameters by importance
    sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Plot importance
    params = [p[0] for p in sorted_params]
    importances = [p[1] for p in sorted_params]
    
    plt.barh(range(len(params)), importances, align='center')
    plt.yticks(range(len(params)), params)
    plt.xlabel('Relative Importance')
    plt.title('Hyperparameter Importance')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, f"parameter_importance_{timestamp}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved parameter importance plot to {fig_path}")


def integrate_hyperparameter_search(df, output_dir):
    """
    Integrate hyperparameter search with the existing model pipeline.
    
    Parameters:
    -----------
    df : original dataframe
    output_dir : directory to save results
    
    Returns:
    --------
    best_model : trained model with optimal hyperparameters
    metrics : performance metrics of best model
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    import pandas as pd
    import os
    import json
    
    print("\n=== RUNNING MODEL WITH HYPERPARAMETER OPTIMIZATION ===")
    search_suffix = "_optimized"
    
    # Data preprocessing (reusing existing functions)
    df_clean = handle_missing_values(df.copy(), df.isnull().sum())
    train_df, test_df = split_train_test(df_clean)
    
    # Decide if you want to use country encoding based on your analysis
    use_country_encoding = True  # Change based on your preference
    
    if use_country_encoding:
        train_df, test_df, country_mapping = encode_country_column(train_df, test_df)
        save_country_mapping(country_mapping, output_dir)
    
    # Feature preparation
    X_train, y_train, X_test, y_test = prepare_features(train_df, test_df, use_country_encoding)
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test, output_dir)
    save_feature_list(X_train, output_dir)
    
    # Perform hyperparameter search
    best_model, best_params, search_results = perform_hyperparameter_search(
        X_train_scaled, y_train, X_test_scaled, y_test, output_dir, 
        search_type='random',  # or 'grid'
        n_iter=20,  # adjust based on computational resources
        cv=5
    )
    
    # Save the optimized model
    save_model(best_model, output_dir, search_suffix)
    
    # Evaluate best model
    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        "training": {
            "MAE": float(mean_absolute_error(y_train, y_train_pred)),
            "MSE": float(mean_squared_error(y_train, y_train_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            "R²": float(r2_score(y_train, y_train_pred))
        },
        "testing": {
            "MAE": float(mean_absolute_error(y_test, y_test_pred)),
            "MSE": float(mean_squared_error(y_test, y_test_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            "R²": float(r2_score(y_test, y_test_pred))
        }
    }
    
    # Save metrics
    save_metrics(metrics, output_dir, search_suffix)
    print_metrics(metrics, "Model Performance (Optimized Hyperparameters)")
    
    # Analysis and visualization of best model
    importance_df = analyze_feature_importance(best_model, X_train, output_dir, search_suffix)
    test_results = prepare_test_results(test_df, y_test, y_test_pred, output_dir, search_suffix)
    create_visualizations(y_train, y_train_pred, y_test, y_test_pred, test_results, 
                         importance_df, df_clean, X_train, output_dir, search_suffix)
    
    # Summary report
    create_summary_report(df_clean, train_df, test_df, X_train, best_model, metrics, 
                         importance_df, output_dir, scaler, search_suffix, use_country_encoding)
    
    return best_model, metrics

def main():
    """Main function to run the entire analysis pipeline."""
    # Setup
    output_dir = create_output_directory()
    file_path = "processed_climate_change_dataset.csv"  # Update this path to your actual file path
    
    # Data loading and exploration
    df, missing_values = load_and_explore_data(file_path)
    save_data_summaries(df, missing_values, output_dir)
    
    # Run standard model without country encoding
    print("\n=== RUNNING MODEL WITHOUT COUNTRY ENCODING ===")
    
    # Data preprocessing
    df_clean = handle_missing_values(df.copy(), missing_values)
    train_df, test_df = split_train_test(df_clean)
    save_split_info(train_df, test_df, df_clean, output_dir)
    
    # Feature preparation - without country encoding
    use_country_encoding = False
    X_train, y_train, X_test, y_test = prepare_features(train_df, test_df, use_country_encoding)
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
    print_metrics(metrics, "Model Performance (Standard)")
    
    # Analysis and visualization
    importance_df = analyze_feature_importance(model, X_train, output_dir)
    test_results = prepare_test_results(test_df, y_test, y_test_pred, output_dir)
    create_visualizations(y_train, y_train_pred, y_test, y_test_pred, test_results, importance_df, df_clean, X_train, output_dir)
    
    # Summary report
    create_summary_report(df_clean, train_df, test_df, X_train, model, metrics, importance_df, output_dir, scaler, "", use_country_encoding)
    
    # Run alternative model with country encoding
    metrics_with_country = run_model_with_country_encoding(df, output_dir)
    
    # Add hyperparameter optimization
    print("\n=== RUNNING MODEL WITH HYPERPARAMETER OPTIMIZATION ===")
    best_model, metrics_optimized = integrate_hyperparameter_search(df, output_dir)
    
    # Compare all models
    print("\n=== MODEL COMPARISON ===")
    print("Standard model (no country encoding):")
    print(f"  Test R²: {metrics['testing']['R²']:.4f}")
    print(f"  Test RMSE: {metrics['testing']['RMSE']:.4f}")
    print("\nModel with country encoding:")
    print(f"  Test R²: {metrics_with_country['testing']['R²']:.4f}")
    print(f"  Test RMSE: {metrics_with_country['testing']['RMSE']:.4f}")
    print("\nOptimized model:")
    print(f"  Test R²: {metrics_optimized['testing']['R²']:.4f}")
    print(f"  Test RMSE: {metrics_optimized['testing']['RMSE']:.4f}")
    
    # Determine best model
    models = {
        "Standard": metrics['testing']['R²'],
        "With Country": metrics_with_country['testing']['R²'],
        "Optimized": metrics_optimized['testing']['R²']
    }
    best_model_name = max(models, key=models.get)
    print(f"\n{best_model_name} model performed best with R² of {models[best_model_name]:.4f}!")
    
    print("\nAnalysis pipeline completed successfully!")
    print(f"All results saved to directory: {output_dir}")
if __name__ == "__main__":
    main()
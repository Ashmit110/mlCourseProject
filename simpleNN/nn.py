import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class NeuralNetRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=16, dropout_rate=0.2):
        """Simple neural network for regression"""
        super(NeuralNetRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        return self.model(x)

def create_results_folder():
    """Create a folder for saving results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"co2_prediction_results_{timestamp}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def load_and_preprocess_data(file_path):
    """Load and preprocess the data"""
    df = pd.read_csv(file_path)
    print("\nSample row before processing:")
    print(df.iloc[0])
    
    # Check data types
    print("\nData types:")
    print(df.dtypes)
    
    # Fill missing values (median for numeric, mode for categorical)
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def normalize_data(df, target='CO2 Emissions (Tons/Capita)', exclude_cols=['Country', 'Year']):
    """Normalize numeric features and target"""
    df_normalized = df.copy()
    
    # Get numeric columns to normalize (excluding specified columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols and col != target]
    
    # Initialize scalers
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    # Fit and transform features
    if cols_to_normalize:
        df_normalized[cols_to_normalize] = feature_scaler.fit_transform(df[cols_to_normalize])
    
    # Fit and transform target
    df_normalized[target] = target_scaler.fit_transform(df[[target]])
    
    return df_normalized, feature_scaler, target_scaler

def encode_country(df):
    """One-hot encode the country column"""
    df_encoded = df.copy()
    countries = df['Country'].unique()
    
    # Create one-hot encoded columns
    country_cols = []
    for country in countries:
        col_name = f'Country_{country}'
        df_encoded[col_name] = (df['Country'] == country).astype(int)
        country_cols.append(col_name)
    
    return df_encoded, country_cols

def train_test_split_by_year(df, test_years=[2022, 2023]):
    """Split data into training and test sets based on years"""
    train_df = df[~df['Year'].isin(test_years)]
    test_df = df[df['Year'].isin(test_years)]
    return train_df, test_df

def analyze_correlations(df, target='CO2 Emissions (Tons/Capita)', results_folder=None):
    """Analyze correlations between features and target"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr = df[numeric_cols].corr()[target].sort_values(ascending=False)
    print("\nCorrelations with CO2 Emissions:")
    print(corr)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if results_folder:
        plt.savefig(os.path.join(results_folder, "correlation_matrix.png"))
    
    return corr

def build_and_evaluate_nn_model(train_data, test_data, features, target='CO2 Emissions (Tons/Capita)', 
                              target_scaler=None, epochs=100, batch_size=16, learning_rate=0.001):
    """Build and evaluate a neural network model with PyTorch"""
    # Prepare data for PyTorch
    X_train = train_data[features].values.astype(np.float32)
    y_train = train_data[target].values.astype(np.float32).reshape(-1, 1)
    
    X_test = test_data[features].values.astype(np.float32)
    y_test = test_data[target].values.astype(np.float32).reshape(-1, 1)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)
    
    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    input_size = X_train.shape[1]
    model = NeuralNetRegressor(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\nTraining neural network model...")
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Track loss every few epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
    
    y_pred_norm = y_pred_tensor.numpy().flatten()
    
    # Calculate metrics
    mse_norm = mean_squared_error(y_test, y_pred_norm)
    rmse_norm = np.sqrt(mse_norm)
    mae_norm = mean_absolute_error(y_test.flatten(), y_pred_norm)
    r2 = r2_score(y_test.flatten(), y_pred_norm)
    
    # Create results dataframe
    results_df = test_data.copy()
    results_df['Predicted CO2 (Normalized)'] = y_pred_norm
    
    # Convert to original scale if scaler provided
    if target_scaler:
        y_pred_orig = target_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
        y_test_orig = target_scaler.inverse_transform(y_test).flatten()
        
        results_df['Predicted CO2 (Original)'] = y_pred_orig
        results_df['Actual CO2 (Original)'] = y_test_orig
        
        mse_orig = mean_squared_error(y_test_orig, y_pred_orig)
        rmse_orig = np.sqrt(mse_orig)
        mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
        
        metrics = {
            'mse_norm': mse_norm, 'rmse_norm': rmse_norm, 'mae_norm': mae_norm,
            'mse_orig': mse_orig, 'rmse_orig': rmse_orig, 'mae_orig': mae_orig,
            'r2': r2
        }
    else:
        metrics = {
            'mse': mse_norm, 'rmse': rmse_norm, 'mae': mae_norm, 'r2': r2
        }
    
    # Analyze feature importance using gradient-based sensitivity
    feature_importance = analyze_nn_feature_importance(model, X_train_tensor, features)
    
    return model, feature_importance, results_df, metrics, losses

def analyze_nn_feature_importance(model, X_tensor, feature_names):
    """
    Analyze feature importance using gradient-based sensitivity analysis
    This is a simple approach to estimate feature importance in neural networks
    """
    model.eval()
    X_tensor.requires_grad_(True)
    
    # Forward pass
    output = model(X_tensor)
    
    # Calculate gradients
    output.sum().backward()
    
    # Get gradient magnitudes
    gradients = X_tensor.grad.abs().mean(0).detach().numpy()
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': gradients
    }).sort_values('Importance', ascending=False)
    
    return feature_importance

def predict_with_nn_model(model, full_data, features, target='CO2 Emissions (Tons/Capita)', target_scaler=None):
    """Make predictions for all data using the neural network model"""
    X_all = full_data[features].values.astype(np.float32)
    X_all_tensor = torch.tensor(X_all)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_all_tensor)
    
    y_pred_norm = y_pred_tensor.numpy().flatten()
    
    all_results = full_data.copy()
    all_results['Predicted CO2 (Normalized)'] = y_pred_norm
    
    if target_scaler:
        y_pred_orig = target_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
        all_results['Predicted CO2 (Original)'] = y_pred_orig
        
        # Add actual in original scale too
        y_all_orig = target_scaler.inverse_transform(full_data[target].values.reshape(-1, 1)).flatten()
        all_results['Actual CO2 (Original)'] = y_all_orig
    
    return all_results

def plot_features_importance(feature_importance, results_folder=None):
    """Plot feature importance"""
    plt.figure(figsize=(10, 8))
    # Show top 15 features or all if less than 15
    top_n = min(15, len(feature_importance))
    top_features = feature_importance.head(top_n)
    
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top Feature Importance')
    
    if results_folder:
        plt.savefig(os.path.join(results_folder, "feature_importance.png"))

def plot_predictions(results_df, by_country=True, use_original_scale=True, results_folder=None):
    """Plot actual vs predicted values"""
    if use_original_scale and 'Predicted CO2 (Original)' in results_df.columns:
        pred_col = 'Predicted CO2 (Original)'
        actual_col = 'Actual CO2 (Original)'
        y_label = 'CO2 Emissions (Tons/Capita)'
    else:
        pred_col = 'Predicted CO2 (Normalized)'
        actual_col = 'CO2 Emissions (Tons/Capita)'
        y_label = 'CO2 Emissions (Normalized)'
    
    if by_country:
        countries = results_df['Country'].unique()
        for country in countries:
            country_data = results_df[results_df['Country'] == country].sort_values('Year')
            if len(country_data) < 2:
                continue
                
            plt.figure(figsize=(10, 6))
            plt.plot(country_data['Year'], country_data[actual_col], 'o-', label='Actual')
            plt.plot(country_data['Year'], country_data[pred_col], 'o--', label='Predicted')
            plt.title(f'CO2 Emissions for {country}')
            plt.xlabel('Year')
            plt.ylabel(y_label)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if results_folder:
                plt.savefig(os.path.join(results_folder, f"prediction_{country.replace(' ', '_')}.png"))
    else:
        # Plot aggregated results
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df[actual_col], results_df[pred_col])
        plt.plot([results_df[actual_col].min(), results_df[actual_col].max()], 
                 [results_df[actual_col].min(), results_df[actual_col].max()], 'r--')
        plt.xlabel('Actual CO2')
        plt.ylabel('Predicted CO2')
        plt.title('Actual vs Predicted CO2 Emissions')
        
        if results_folder:
            plt.savefig(os.path.join(results_folder, "overall_predictions.png"))

def plot_training_curve(losses, results_folder=None):
    """Plot training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, 'o-')
    plt.title('Training Loss')
    plt.xlabel('Epoch (x10)')
    plt.ylabel('Loss (MSE)')
    plt.grid(True, alpha=0.3)
    
    if results_folder:
        plt.savefig(os.path.join(results_folder, "training_loss.png"))

def run_co2_prediction_nn(file_path, use_country=True, selected_features=None, test_years=[2022, 2023],
                         epochs=100, batch_size=16, learning_rate=0.001):
    """Main function for neural network prediction model"""
    print("="*50)
    print("CO2 EMISSIONS PREDICTION NEURAL NETWORK MODEL")
    print("="*50)
    
    # Set device (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Create results folder
    results_folder = create_results_folder()
    
    # Load and preprocess data
    df = load_and_preprocess_data(file_path)
    print(f"\nLoaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Analyze correlations
    analyze_correlations(df, results_folder=results_folder)
    
    # List potential features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = 'CO2 Emissions (Tons/Capita)'
    available_features = [col for col in numeric_cols if col != target]
    
    print("\nAvailable numeric features:")
    for i, feature in enumerate(available_features):
        print(f"{i+1}. {feature}")
    
    # If no features selected, use all available
    if selected_features is None:
        # Allow user to select features
        print("\nEnter numbers of features to use (comma-separated), or 'all' for all features:")
        user_input = input()
        
        if user_input.lower() == 'all':
            selected_features = available_features
        else:
            try:
                indices = [int(idx.strip()) - 1 for idx in user_input.split(',')]
                selected_features = [available_features[i] for i in indices]
            except (ValueError, IndexError):
                print("Invalid input. Using all features.")
                selected_features = available_features
    
    print(f"\nSelected features ({len(selected_features)}):")
    for feature in selected_features:
        print(f"- {feature}")
    
    # Normalize data
    df_normalized, feature_scaler, target_scaler = normalize_data(df, target=target, exclude_cols=['Country', 'Year'])
    
    all_features = selected_features.copy()
    
    # Add country encoding if requested
    if use_country:
        df_encoded, country_cols = encode_country(df_normalized)
        print(f"\nAdded one-hot encoding for {len(country_cols)} countries")
        all_features.extend(country_cols)
    else:
        df_encoded = df_normalized
        print("\nSkipping country encoding")
    
    # Split data
    train_df, test_df = train_test_split_by_year(df_encoded, test_years=test_years)
    print(f"\nTraining data: {train_df.shape[0]} rows")
    print(f"Test data: {test_df.shape[0]} rows")
    
    # Build and evaluate model
    model, feature_importance, test_results, metrics, losses = build_and_evaluate_nn_model(
        train_df, test_df, all_features, target=target, target_scaler=target_scaler,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )
    
    # Print top features
    print("\nTop 10 Feature Importance:")
    print(feature_importance.head(10))
    
    # Print evaluation metrics
    print("\nModel Evaluation (Test Data):")
    if 'rmse_orig' in metrics:
        print(f"RMSE (Original): {metrics['rmse_orig']:.4f}")
        print(f"MAE (Original): {metrics['mae_orig']:.4f}")
    
    print(f"RMSE (Normalized): {metrics['rmse_norm']:.4f}")
    print(f"R² Score: {metrics['r2']:.4f}")
    
    # Make predictions for all data
    all_results = predict_with_nn_model(model, df_encoded, all_features, target=target, target_scaler=target_scaler)
    
    # Plot predictions
    print("\nPlotting predictions...")
    plot_predictions(all_results, by_country=(use_country or 'Country' in df.columns), 
                     use_original_scale=True, results_folder=results_folder)
    
    # Plot feature importance
    plot_features_importance(feature_importance, results_folder=results_folder)
    
    # Plot training curve
    plot_training_curve(losses, results_folder=results_folder)
    
    # Save results
    all_results.to_csv(os.path.join(results_folder, "predictions.csv"), index=False)
    feature_importance.to_csv(os.path.join(results_folder, "feature_importance.csv"), index=False)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(results_folder, "nn_model.pth"))
    
    print(f"\nAll results saved to {results_folder}")
    
    return model, all_results, feature_importance

def perform_feature_selection_nn(file_path, use_country=True, test_years=[2022, 2023], 
                               max_features=None, selection_method='forward', 
                               metric='r2', results_folder=None,
                               epochs=100, batch_size=16, learning_rate=0.001):
    """
    Performs feature selection for the neural network model and returns the best model
    
    Parameters:
    -----------
    file_path : str
        Path to the dataset CSV file
    use_country : bool
        Whether to use country one-hot encoding
    test_years : list
        Years to use for testing
    max_features : int
        Maximum number of features to use (None for no limit)
    selection_method : str
        'forward', 'backward', or 'exhaustive'
    metric : str
        Metric to optimize ('r2', 'rmse_orig', 'mae_orig')
    results_folder : str
        Folder to save results (if None, creates a new one)
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    learning_rate : float
        Learning rate for optimization
        
    Returns:
    --------
    tuple
        (best_model, best_results, best_feature_importance, best_features, best_metrics)
    """
    print("="*50)
    print("CO2 EMISSIONS PREDICTION - NEURAL NETWORK FEATURE SELECTION")
    print("="*50)
    
    # Set device (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Create results folder if not provided
    if results_folder is None:
        results_folder = create_results_folder()
    
    # Load and preprocess data
    df = load_and_preprocess_data(file_path)
    print(f"\nLoaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Analyze correlations
    corr_results = analyze_correlations(df, results_folder=results_folder)
    
    # Get potential features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = 'CO2 Emissions (Tons/Capita)'
    available_features = [col for col in numeric_cols if col != target and col != 'Year']
    
    # Normalize data
    df_normalized, feature_scaler, target_scaler = normalize_data(df, target=target, exclude_cols=['Country', 'Year'])
    
    # Add country encoding if requested
    country_cols = []
    if use_country:
        df_encoded, country_cols = encode_country(df_normalized)
        print(f"\nAdded one-hot encoding for {len(country_cols)} countries")
    else:
        df_encoded = df_normalized
        print("\nSkipping country encoding")
    
    # Split data
    train_df, test_df = train_test_split_by_year(df_encoded, test_years=test_years)
    print(f"\nTraining data: {train_df.shape[0]} rows")
    print(f"Test data: {test_df.shape[0]} rows")
    
    # Always include Year as a feature
    base_features = ['Year']
    
    # Limit features if max_features is specified
    if max_features and len(available_features) > max_features:
        # Sort by correlation with target
        sorted_features = corr_results.index.tolist()
        # Filter out target and non-features
        sorted_features = [f for f in sorted_features if f in available_features]
        # Take top features
        available_features = sorted_features[:max_features]
    
    # Feature selection
    best_model = None
    best_features = None
    best_results = None
    best_feature_importance = None
    best_metric_value = -float('inf') if metric == 'r2' else float('inf')
    best_metrics = None
    best_losses = None
    
    # Store all results for comparison
    all_selection_results = []
    
    # Forward selection
    if selection_method == 'forward':
        print("\nPerforming forward feature selection...")
        current_features = base_features.copy()
        remaining_features = available_features.copy()
        improving = True
        
        while remaining_features and improving:
            best_new_metric = -float('inf') if metric == 'r2' else float('inf')
            best_new_feature = None
            
            for feature in remaining_features:
                # Try adding this feature
                test_features = current_features + [feature]
                if use_country:
                    test_features += country_cols
                
                # Build and evaluate model
                model, feat_imp, results, metrics, losses = build_and_evaluate_nn_model(
                    train_df, test_df, test_features, target=target, target_scaler=target_scaler,
                    epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
                )
                
                metric_value = metrics[metric]
                
                # For RMSE and MAE, lower is better; for R², higher is better
                is_better = (metric_value > best_new_metric) if metric == 'r2' else (metric_value < best_new_metric)
                
                if is_better:
                    best_new_metric = metric_value
                    best_new_feature = feature
                    temp_model = model
                    temp_feat_imp = feat_imp
                    temp_results = results
                    temp_metrics = metrics
                    temp_losses = losses
                
                # Store result
                all_selection_results.append({
                    'features': test_features.copy(),
                    'metric_name': metric,
                    'metric_value': metric_value,
                    'num_features': len(test_features)
                })
            
            # Check if adding the best feature improves the model
            is_improvement = ((metric == 'r2' and best_new_metric > best_metric_value) or 
                             (metric != 'r2' and best_new_metric < best_metric_value))
            
            if best_new_feature and is_improvement:
                current_features.append(best_new_feature)
                remaining_features.remove(best_new_feature)
                best_metric_value = best_new_metric
                best_model = temp_model
                best_features = current_features.copy()
                best_results = temp_results
                best_feature_importance = temp_feat_imp
                best_metrics = temp_metrics
                best_losses = temp_losses
                
                print(f"Added feature: {best_new_feature}, {metric}: {best_metric_value:.4f}")
            else:
                improving = False
    
    # Backward elimination
    elif selection_method == 'backward':
        print("\nPerforming backward feature elimination...")
        current_features = base_features + available_features.copy()
        improving = True
        
        # Start with all features
        if use_country:
            all_features = current_features + country_cols
        else:
            all_features = current_features
            
        # Initial evaluation with all features
        model, feat_imp, results, metrics, losses = build_and_evaluate_nn_model(
            train_df, test_df, all_features, target=target, target_scaler=target_scaler,
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
        )
        
        best_metric_value = metrics[metric]
        best_model = model
        best_features = current_features.copy()
        best_results = results
        best_feature_importance = feat_imp
        best_metrics = metrics
        best_losses = losses
        
        print(f"Starting with all {len(all_features)} features, {metric}: {best_metric_value:.4f}")
        
        while len(current_features) > 1 and improving:  # Always keep at least base features
            best_new_metric = -float('inf') if metric == 'r2' else float('inf')
            worst_feature = None
            
            for feature in [f for f in current_features if f not in base_features]:
                # Try removing this feature
                test_features = [f for f in current_features if f != feature]
                if use_country:
                    test_features += country_cols
                
                # Build and evaluate model
                model, feat_imp, results, metrics, losses = build_and_evaluate_nn_model(
                    train_df, test_df, test_features, target=target, target_scaler=target_scaler,
                    epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
                )
                
                metric_value = metrics[metric]
                
                # For RMSE and MAE, lower is better; for R², higher is better
                is_better = (metric_value > best_new_metric) if metric == 'r2' else (metric_value < best_new_metric)
                
                if is_better:
                    best_new_metric = metric_value
                    worst_feature = feature
                    temp_model = model
                    temp_feat_imp = feat_imp
                    temp_results = results
                    temp_metrics = metrics
                    temp_losses = losses
                
                # Store result
                all_selection_results.append({
                    'features': test_features.copy(),
                    'metric_name': metric,
                    'metric_value': metric_value,
                    'num_features': len(test_features)
                })
            
            # Check if removing the worst feature improves the model
            is_improvement = ((metric == 'r2' and best_new_metric > best_metric_value) or 
                             (metric != 'r2' and best_new_metric < best_metric_value))
            
            if worst_feature and is_improvement:
                current_features.remove(worst_feature)
                best_metric_value = best_new_metric
                best_model = temp_model
                best_features = current_features.copy()
                best_results = temp_results
                best_feature_importance = temp_feat_imp
                best_metrics = temp_metrics
                best_losses = temp_losses
                
                print(f"Removed feature: {worst_feature}, {metric}: {best_metric_value:.4f}")
            else:
                improving = False
    
    # Exhaustive search - try combinations of features, useful for small feature sets
    elif selection_method == 'exhaustive':
        import itertools
        
        print("\nPerforming exhaustive feature search...")
        # Limit the number of features to avoid combinatorial explosion
        max_combi = min(max_features if max_features else 10, len(available_features))
        
        for k in range(1, max_combi + 1):
            print(f"Testing combinations of {k} features...")
            for features_combo in itertools.combinations(available_features, k):
                test_features = base_features + list(features_combo)
                if use_country:
                    test_features += country_cols
                
                # Build and evaluate model
                # Build and evaluate model
                model, feat_imp, results, metrics, losses = build_and_evaluate_nn_model(
                    train_df, test_df, test_features, target=target, target_scaler=target_scaler,
                    epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
                )
                
                metric_value = metrics[metric]
                
                # For RMSE and MAE, lower is better; for R², higher is better
                is_better = ((metric == 'r2' and metric_value > best_metric_value) or 
                             (metric != 'r2' and metric_value < best_metric_value))
                
                if is_better:
                    best_metric_value = metric_value
                    best_model = model
                    best_features = test_features.copy()
                    best_results = results
                    best_feature_importance = feat_imp
                    best_metrics = metrics
                    best_losses = losses
                
                # Store result
                all_selection_results.append({
                    'features': test_features.copy(),
                    'metric_name': metric,
                    'metric_value': metric_value,
                    'num_features': len(test_features)
                })
                
                # Print progress occasionally
                if len(all_selection_results) % 10 == 0:
                    print(f"Tested {len(all_selection_results)} combinations so far...")
    
    # Print final results
    country_feature_count = len(country_cols) if use_country else 0
    print("\n" + "="*50)
    print(f"FEATURE SELECTION RESULTS ({selection_method.upper()} SELECTION)")
    print("="*50)
    print(f"Best {metric}: {best_metric_value:.4f}")
    print(f"Number of features selected: {len(best_features) + country_feature_count}")
    print("\nSelected features:")
    for feature in best_features:
        print(f"- {feature}")
    
    if use_country:
        print(f"- Plus {country_feature_count} country features")
    
    # Plot feature importance for the best model
    print("\nPlotting feature importance for the best model...")
    plot_features_importance(best_feature_importance, results_folder=results_folder)
    
    # Make predictions for all data using the best model
    all_features = best_features.copy()
    if use_country:
        all_features.extend(country_cols)
        
    all_results = predict_with_nn_model(best_model, df_encoded, all_features, target=target, target_scaler=target_scaler)
    
    # Plot predictions
    print("\nPlotting predictions for the best model...")
    plot_predictions(all_results, by_country=(use_country or 'Country' in df.columns), 
                     use_original_scale=True, results_folder=results_folder)
    
    # Plot training curve
    if best_losses:
        plot_training_curve(best_losses, results_folder=results_folder)
    
    # Save results
    best_results.to_csv(os.path.join(results_folder, "best_predictions.csv"), index=False)
    best_feature_importance.to_csv(os.path.join(results_folder, "best_feature_importance.csv"), index=False)
    
    # Save feature selection results
    selection_df = pd.DataFrame(all_selection_results)
    selection_df.to_csv(os.path.join(results_folder, "feature_selection_results.csv"), index=False)
    
    # Save best feature set
    with open(os.path.join(results_folder, "best_features.txt"), 'w') as f:
        f.write("Best features:\n")
        for feature in best_features:
            f.write(f"- {feature}\n")
            
    # Save model
    torch.save(best_model.state_dict(), os.path.join(results_folder, "best_nn_model.pth"))
    
    # Plot feature selection progress
    if len(all_selection_results) > 1:
        plt.figure(figsize=(10, 6))
        selection_df = selection_df.sort_values('num_features')
        plt.plot(selection_df['num_features'], selection_df['metric_value'], 'o-')
        plt.title(f'Feature Selection Progress ({metric})')
        plt.xlabel('Number of Features')
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_folder, "feature_selection_progress.png"))
    
    print(f"\nAll results saved to {results_folder}")
    return best_model, all_results, best_feature_importance, best_features, best_metrics

# Example usage:
if __name__ == "__main__":
    csv_file = "processed_climate_change_dataset.csv"
    
    # Option 1: Interactive mode - will prompt for features and country encoding
    # model, results, feature_importance = run_co2_prediction_nn(csv_file)
    
    # Option 2: Specific features, with country encoding
    selected_features = ["Avg Temperature (°C)", "Population", "Renewable Energy (%)","Forest Area (%)","Rainfall (mm)"]
    model, results, feature_importance = run_co2_prediction_nn(
        csv_file, 
        use_country=False, 
        selected_features=selected_features,
        epochs=500,
        batch_size=16
    )
    
    # Option 3: Feature selection with forward selection
    # model, results, feature_importance, features, metrics = perform_feature_selection_nn(
    #     csv_file, 
    #     use_country=True,
    #     selection_method='forward',
    #     metric='r2',
    #     epochs=100,
    #     batch_size=16,
    #     learning_rate=0.001
    # )
    
    # Alternative: Backward elimination
    # model, results, feature_importance, features, metrics = perform_feature_selection_nn(
    #     csv_file, 
    #     use_country=True,
    #     selection_method='backward',
    #     metric='rmse_orig',
    #     epochs=100
    # )
    
    # Alternative: Exhaustive search (for small feature sets)
    # model, results, feature_importance, features, metrics = perform_feature_selection_nn(
    #     csv_file, 
    #     use_country=True,
    #     selection_method='exhaustive',
    #     max_features=8,  # Limit to avoid combinatorial explosion
    #     metric='r2',
    #     epochs=100
    # )
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator

def load_data(file_path):
    """Load climate data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def simple_moving_average(data, window_size):
    """Calculate simple moving average for a series"""
    return data.rolling(window=window_size).mean()

def predict_co2_emissions(df, window_sizes=[3,5]):
    """Predict CO2 emissions for 2022 and 2023 using simple moving averages"""
    # Get unique countries
    countries = df['Country'].unique()
    
    # Create a results directory if it doesn't exist
    results_dir = "baselineMA\sma_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Dictionary to store predictions
    predictions = {}
    
    # DataFrame to store all predictions and actual values for summary graph
    summary_data = []
    
    # Process each country
    for country in countries:
        country_data = df[df['Country'] == country].sort_values('Year')
        
        # Check if we have data for 2022 and 2023
        has_2022 = 2022 in country_data['Year'].values
        has_2023 = 2023 in country_data['Year'].values
        
        if not (has_2022 or has_2023):
            print(f"No data for 2022 or 2023 for {country}, skipping...")
            continue
        
        # Store actual values if available
        actual_values = {}
        if has_2022:
            actual_values[2022] = country_data[country_data['Year'] == 2022]['CO2 Emissions (Tons/Capita)'].values[0]
        if has_2023:
            actual_values[2023] = country_data[country_data['Year'] == 2023]['CO2 Emissions (Tons/Capita)'].values[0]
        
        # Dictionary to store predictions for this country
        country_predictions = {'Actual': actual_values}
        
        # For each window size
        for window_size in window_sizes:
            predictions_ws = {}
            
            # Filter data up to 2021 for predicting 2022
            if has_2022:
                data_for_2022 = country_data[country_data['Year'] < 2022].copy()
                if len(data_for_2022) >= window_size:
                    # Calculate moving average for CO2 emissions
                    sma = simple_moving_average(data_for_2022['CO2 Emissions (Tons/Capita)'], window_size)
                    # Predict 2022 using the last available moving average
                    predictions_ws[2022] = sma.iloc[-1]
                    # Add to summary data
                    summary_data.append({
                        'Country': country,
                        'Year': 2022,
                        'Actual': actual_values[2022],
                        'Predicted': sma.iloc[-1],
                        'Method': f'SMA-{window_size}'
                    })
            
            # Filter data up to 2022 for predicting 2023
            if has_2023:
                data_for_2023 = country_data[country_data['Year'] < 2023].copy()
                if len(data_for_2023) >= window_size:
                    # Calculate moving average for CO2 emissions
                    sma = simple_moving_average(data_for_2023['CO2 Emissions (Tons/Capita)'], window_size)
                    # Predict 2023 using the last available moving average
                    predictions_ws[2023] = sma.iloc[-1]
                    # Add to summary data
                    summary_data.append({
                        'Country': country,
                        'Year': 2023,
                        'Actual': actual_values[2023],
                        'Predicted': sma.iloc[-1],
                        'Method': f'SMA-{window_size}'
                    })
            
            # Store predictions for this window size
            country_predictions[f'SMA-{window_size}'] = predictions_ws
        
        predictions[country] = country_predictions
        
        # Plot predictions vs actual values with improved year axis
        plot_predictions(country, country_data, country_predictions, results_dir)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Create summary graph
    if not summary_df.empty:
        create_summary_graph(summary_df, results_dir)
    
    return predictions

def plot_predictions(country, country_data, predictions, results_dir):
    """Plot predictions against actual values with improved year axis"""
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    historical_data = country_data[country_data['Year'] < 2022]
    plt.plot(historical_data['Year'], historical_data['CO2 Emissions (Tons/Capita)'], 
             marker='o', linestyle='-', label='Historical Data')
    
    # Plot actual values for 2022 and 2023 if available
    actual_values = predictions['Actual']
    years = list(actual_values.keys())
    values = list(actual_values.values())
    if years:
        plt.plot(years, values, marker='*', linestyle='', markersize=12, 
                 color='black', label='Actual Values')
    
    # Plot predictions
    for method, pred_dict in predictions.items():
        if method == 'Actual':
            continue
        
        pred_years = list(pred_dict.keys())
        pred_values = list(pred_dict.values())
        if pred_years:
            plt.plot(pred_years, pred_values, marker='s', linestyle='--', 
                     label=f'{method} Prediction')
    
    plt.title(f'CO2 Emissions Predictions for {country}')
    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions (Tons/Capita)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Improve year axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer ticks
    
    # Set a reasonable range for x-axis - include a bit of history and predictions
    min_year = min(historical_data['Year'].min(), min(years) if years else float('inf')) - 1
    max_year = max(max(years) if years else float('-inf'), 2023) + 1
    plt.xlim(min_year, max_year)
    
    # Add year labels for all years in range
    plt.xticks(range(int(min_year), int(max_year) + 1))
    
    plt.tight_layout()
    
    # Save the plot
    safe_country_name = country.replace(' ', '_')
    plt.savefig(f"{results_dir}/{safe_country_name}_predictions.png")
    plt.close()

def create_summary_graph(summary_df, results_dir):
    """Create a summary graph comparing all countries' predictions and actual values"""
    # Get unique countries, years and methods
    countries = summary_df['Country'].unique()
    years = sorted(summary_df['Year'].unique())
    methods = summary_df['Method'].unique()
    
    # Choose best method based on average error
    summary_df['Error'] = abs(summary_df['Actual'] - summary_df['Predicted'])
    best_method = summary_df.groupby('Method')['Error'].mean().idxmin()
    
    # Filter for best method
    best_method_df = summary_df[summary_df['Method'] == best_method]
    
    # Create a summary plot - one plot per year with all countries
    for year in years:
        year_data = best_method_df[best_method_df['Year'] == year]
        
        if not year_data.empty:
            # Determine plot dimensions based on number of countries
            n_countries = len(year_data['Country'].unique())
            fig_width = max(10, n_countries * 1.5)  # Scale width with number of countries
            
            plt.figure(figsize=(fig_width, 8))
            
            # Sort by country name for consistent ordering
            year_data = year_data.sort_values('Country')
            
            # Extract x and y data
            countries_list = year_data['Country'].tolist()
            actual_values = year_data['Actual'].tolist()
            predicted_values = year_data['Predicted'].tolist()
            
            # Set up x positions for bars
            x = np.arange(len(countries_list))
            width = 0.35
            
            # Create grouped bar chart
            plt.bar(x - width/2, actual_values, width, label='Actual', color='darkblue')
            plt.bar(x + width/2, predicted_values, width, label=f'Predicted ({best_method})', color='orange')
            
            # Add value labels on top of bars
            for i, (actual, pred) in enumerate(zip(actual_values, predicted_values)):
                plt.text(i - width/2, actual + 0.1, f'{actual:.2f}', ha='center', va='bottom', fontsize=9)
                plt.text(i + width/2, pred + 0.1, f'{pred:.2f}', ha='center', va='bottom', fontsize=9)
            
            # Add error percentages between bars
            for i, (actual, pred) in enumerate(zip(actual_values, predicted_values)):
                error_pct = ((pred - actual) / actual) * 100 if actual != 0 else float('inf')
                color = 'red' if abs(error_pct) > 10 else 'green'
                plt.text(i, min(actual, pred) - 0.8, f'{error_pct:.1f}%', ha='center', va='top', 
                         color=color, fontweight='bold')
            
            plt.xlabel('Country')
            plt.ylabel('CO2 Emissions (Tons/Capita)')
            plt.title(f'Actual vs Predicted CO2 Emissions for {year} ({best_method})')
            plt.xticks(x, countries_list, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, axis='y', alpha=0.3)
            
            # Ensure y-axis starts from 0
            plt.ylim(bottom=0)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(f"{results_dir}/summary_comparison_{year}.png")
            plt.close()
    
    # Create a comprehensive summary plot with all years and countries
    # This will be a scatter plot of Actual vs Predicted values
    plt.figure(figsize=(12, 8))
    
    # Plot the identity line (y=x)
    min_val = min(summary_df['Actual'].min(), summary_df['Predicted'].min()) - 0.5
    max_val = max(summary_df['Actual'].max(), summary_df['Predicted'].max()) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction (y=x)')
    
    # Create scatter plot with different colors for each country
    for country in countries:
        country_data = summary_df[summary_df['Country'] == country]
        plt.scatter(country_data['Actual'], country_data['Predicted'], 
                   label=country, s=100, alpha=0.7)
    
    plt.xlabel('Actual CO2 Emissions (Tons/Capita)')
    plt.ylabel('Predicted CO2 Emissions (Tons/Capita)')
    plt.title('Actual vs Predicted CO2 Emissions - All Countries and Years')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add diagonal bands to indicate 10% error bounds
    plt.fill_between([min_val, max_val], 
                     [min_val*0.9, max_val*0.9], 
                     [min_val*1.1, max_val*1.1], 
                     color='green', alpha=0.1, label='±10% Error Band')
    
    # Adjust legend
    if len(countries) > 8:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/comprehensive_summary.png")
    plt.close()

def calculate_errors(predictions):
    """Calculate prediction errors for all predictions"""
    error_metrics = {}
    
    for country, country_predictions in predictions.items():
        actual_values = country_predictions['Actual']
        country_errors = {}
        
        for method, pred_dict in country_predictions.items():
            if method == 'Actual':
                continue
            
            # Calculate errors for each predicted year
            method_errors = {}
            for year, actual in actual_values.items():
                if year in pred_dict:
                    predicted = pred_dict[year]
                    error = actual - predicted
                    percent_error = (error / actual) * 100 if actual != 0 else float('inf')
                    abs_error = abs(error)
                    abs_percent_error = abs(percent_error)
                    
                    method_errors[year] = {
                        'Actual': actual,
                        'Predicted': predicted,
                        'Error': error,
                        'Absolute Error': abs_error,
                        'Percent Error': percent_error,
                        'Absolute Percent Error': abs_percent_error
                    }
            
            country_errors[method] = method_errors
        
        error_metrics[country] = country_errors
    
    return error_metrics

def print_error_summary(error_metrics):
    """Print a summary of prediction errors"""
    print("\nPrediction Error Summary:")
    print("========================")
    
    # Prepare data for ranking methods
    method_overall_errors = {}
    
    for country, country_errors in error_metrics.items():
        print(f"\nCountry: {country}")
        
        for method, year_errors in country_errors.items():
            print(f"  Method: {method}")
            
            total_abs_error = 0
            total_abs_percent_error = 0
            count = 0
            
            for year, metrics in year_errors.items():
                print(f"    Year {year}: Actual = {metrics['Actual']:.2f}, Predicted = {metrics['Predicted']:.2f}, "
                      f"Error = {metrics['Error']:.2f}, Percent Error = {metrics['Percent Error']:.2f}%")
                
                total_abs_error += metrics['Absolute Error']
                total_abs_percent_error += metrics['Absolute Percent Error']
                count += 1
            
            if count > 0:
                mean_abs_error = total_abs_error / count
                mean_abs_percent_error = total_abs_percent_error / count
                print(f"    Mean Absolute Error: {mean_abs_error:.2f}")
                print(f"    Mean Absolute Percent Error: {mean_abs_percent_error:.2f}%")
                
                # Store for method ranking
                if method not in method_overall_errors:
                    method_overall_errors[method] = {'total_error': 0, 'count': 0}
                
                method_overall_errors[method]['total_error'] += total_abs_percent_error
                method_overall_errors[method]['count'] += count
    
    # Print overall method ranking
    print("\nOverall Method Performance Ranking:")
    print("=================================")
    
    for method, data in method_overall_errors.items():
        if data['count'] > 0:
            overall_mape = data['total_error'] / data['count']
            print(f"{method}: Mean Absolute Percent Error = {overall_mape:.2f}%")

def main():
    """Main function to run the SMA prediction model"""
    print("="*80)
    print("SIMPLE MOVING AVERAGE MODEL FOR CO2 EMISSIONS PREDICTION")
    print("="*80)
    
    # Load data
    file_path = "processed_climate_change_dataset.csv"  # Change this to your file path
    df = load_data(file_path)
    
    if df is None:
        return
    
    # Display basic info about the data
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    print("\nUnique countries in the dataset:")
    print(df['Country'].unique())
    
    print("\nYear range in the dataset:")
    print(f"Min year: {df['Year'].min()}, Max year: {df['Year'].max()}")
    
    # Predict CO2 emissions using SMA with window sizes 3 and 5
    print("\nPredicting CO2 emissions for 2022 and 2023...")
    predictions = predict_co2_emissions(df, window_sizes=[3, 5])
    
    # Calculate and print error metrics
    error_metrics = calculate_errors(predictions)
    print_error_summary(error_metrics)
    
    print("\nPrediction completed. Plots have been saved to the 'sma_results' directory.")

if __name__ == "__main__":
    main()
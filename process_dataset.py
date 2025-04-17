import pandas as pd

# Function to process the CSV file
def process_csv(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Reorder columns to put Country first, Year second, and CO2 Emissions last
    columns = df.columns.tolist()
    country_col = 'Country'
    year_col = 'Year'
    co2_col = 'CO2 Emissions (Tons/Capita)'
    
    # Remove these columns from the list
    columns.remove(country_col)
    columns.remove(year_col)
    columns.remove(co2_col)
    
    # Reorder columns
    new_order = [country_col, year_col] + columns + [co2_col]
    df = df[new_order]
    
    # Group by Country and Year, and calculate the mean for all numeric columns
    df = df.groupby(['Country', 'Year']).mean().reset_index()
    
    # Round all numeric columns to 2 decimal places
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].round(2)
    
    # Sort by Country first, then by Year
    df = df.sort_values(['Country', 'Year'])
    
    # Save to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

# Example usage
if __name__ == "__main__":
    input_file = "climate_change_dataset.csv"  # Replace with your input filename
    output_file = "processed_climate_change_dataset.csv"  # Replace with your desired output filename
    process_csv(input_file, output_file)
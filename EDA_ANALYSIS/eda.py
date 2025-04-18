# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set visualization style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the dataset
df = pd.read_csv('climate_change_dataset-1.csv')

# Basic dataset information
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Unique countries and years
countries = df['Country'].unique()
years = df['Year'].unique()
print(f"\nNumber of unique countries: {len(countries)}")
print(f"Number of unique years: {len(years)}")
print(f"Year range: {min(years)} to {max(years)}")

# Box plot for temperature distribution by country
plt.figure(figsize=(14, 10))
sns.boxplot(x='Country', y='Avg Temperature (°C)', data=df)
plt.xticks(rotation=90)
plt.title('Temperature Distribution by Country')
plt.tight_layout()
plt.show()

# Box plot for CO2 emissions by country
plt.figure(figsize=(14, 10))
sns.boxplot(x='Country', y='CO2 Emissions (Tons/Capita)', data=df)
plt.xticks(rotation=90)
plt.title('CO2 Emissions Distribution by Country')
plt.tight_layout()
plt.show()

# Box plot for renewable energy by country
plt.figure(figsize=(14, 10))
sns.boxplot(x='Country', y='Renewable Energy (%)', data=df)
plt.xticks(rotation=90)
plt.title('Renewable Energy Distribution by Country')
plt.tight_layout()
plt.show()

# Scatter plot of temperature vs CO2 emissions
plt.figure()
sns.scatterplot(x='Avg Temperature (°C)', y='CO2 Emissions (Tons/Capita)', 
                hue='Country', size='Population', sizes=(20, 200), data=df)
plt.title('Relationship Between Temperature and CO2 Emissions')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Scatter plot of renewable energy vs CO2 emissions
plt.figure()
sns.scatterplot(x='Renewable Energy (%)', y='CO2 Emissions (Tons/Capita)', 
                hue='Country', size='Population', sizes=(20, 200), data=df)
plt.title('Relationship Between Renewable Energy and CO2 Emissions')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Scatter plot of forest area vs extreme weather events
plt.figure()
sns.scatterplot(x='Forest Area (%)', y='Extreme Weather Events', 
                hue='Country', size='Year', sizes=(50, 200), data=df)
plt.title('Relationship Between Forest Area and Extreme Weather Events')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Line plot of average global temperature over time
plt.figure()
yearly_avg_temp = df.groupby('Year')['Avg Temperature (°C)'].mean().reset_index()
plt.plot(yearly_avg_temp['Year'], yearly_avg_temp['Avg Temperature (°C)'], marker='o', linestyle='-', linewidth=2)
plt.title('Global Average Temperature Over Time')
plt.xlabel('Year')
plt.ylabel('Average Temperature (°C)')
plt.grid(True)
plt.show()

# Line plot of sea level rise over time
plt.figure()
yearly_avg_sea = df.groupby('Year')['Sea Level Rise (mm)'].mean().reset_index()
plt.plot(yearly_avg_sea['Year'], yearly_avg_sea['Sea Level Rise (mm)'], marker='o', linestyle='-', color='blue', linewidth=2)
plt.title('Global Average Sea Level Rise Over Time')
plt.xlabel('Year')
plt.ylabel('Sea Level Rise (mm)')
plt.grid(True)
plt.show()

# Line plot of temperature trends for selected countries
plt.figure()
for country in ['USA', 'China', 'India', 'Germany', 'Brazil']:
    if country in countries:
        country_data = df[df['Country'] == country].sort_values('Year')
        plt.plot(country_data['Year'], country_data['Avg Temperature (°C)'], marker='o', linestyle='-', label=country)
plt.title('Temperature Trends by Country')
plt.xlabel('Year')
plt.ylabel('Average Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

# Bubble chart for renewable energy progression
fig = px.scatter(df, x='Year', y='Renewable Energy (%)', 
                 size='Population', color='Country',
                 hover_name='Country', size_max=60)
fig.update_layout(
    title='Renewable Energy Adoption Over Time',
    xaxis_title='Year',
    yaxis_title='Renewable Energy (%)',
    legend_title='Country'
)
fig.show()

# Violin plot for temperature distribution by country
plt.figure(figsize=(14, 10))
sns.violinplot(x='Country', y='Avg Temperature (°C)', data=df)
plt.xticks(rotation=90)
plt.title('Detailed Temperature Distribution by Country')
plt.tight_layout()
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Fri May  3 01:39:29 2024

@author: ChelseySSS
"""

# PPHA 30537
# Spring 2024
# Homework 3

# SHIHAN ZHAO

# SHIHAN ZHAO
# sz111111

# Due date: Sunday May 5th before midnight
# Write your answers in the space between the questions, and commit/push only
# this file to your repo. Note that there can be a difference between giving a
# "minimally" right answer, and a really good answer, so it can pay to put
# thought into your work.

##################

#NOTE: All of the plots the questions ask for should be saved and committed to
# your repo under the name "q1_1_plot.png" (for 1.1), "q1_2_plot.png" (for 1.2),
# etc. using fig.savefig. If a question calls for more than one plot, name them
# "q1_1a_plot.png", "q1_1b_plot.png",  etc.

# Question 1.1: With the x and y values below, create a plot using only Matplotlib.
# You should plot y1 as a scatter plot and y2 as a line, using different colors
# and a legend.  You can name the data simply "y1" and "y2".  Make sure the
# axis tick labels are legible.  Add a title that reads "HW3 Q1.1".

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

x = pd.date_range(start='1990/1/1', end='1991/12/1', freq='MS')
y1 = np.random.normal(10, 2, len(x))
y2 = [np.sin(v)+10 for v in range(len(x))]


# Creating the plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y1, color='blue', label='y1')
plt.plot(x, y2, color='red', label='y2', linewidth=2)

# Adding legend
plt.legend()

# Making sure the x-axis labels are legible
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Value')

# Adding title
plt.title('HW3 Q1.1')

# Saving the plot
plt.savefig('q1_1_plot.png')

# Showing the plot
plt.show()



# Question 1.2: Using only Matplotlib, reproduce the figure in this repo named
# question_2_figure.png.

# Define data points for x and y axes
x = range(10, 19)
y_blue = range(10, 19)
y_red = range(18, 9, -1)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y_blue, label='Blue', color='blue')  # Blue line
plt.plot(x, y_red, label='Red', color='red')    # Red line

# Adding title and legend
plt.title('X marks the spot')
plt.legend()

# Set the range for the axes
plt.xlim(10, 18)
plt.ylim(10, 18)

# Saving the plot
plt.savefig('q1_2_plot.png')

# Show the plot
plt.show()



# Question 1.3: Load the mpg.csv file that is in this repo, and create a
# plot that tests the following hypothesis: a car with an engine that has
# a higher displacement (i.e. is bigger) will get worse gas mileage than
# one that has a smaller displacement.  Test the same hypothesis for mpg
# against horsepower and weight.

# Load the mpg.csv file
df = pd.read_csv('C:/Users/ChelseySSS/Desktop/mpg.csv')

# Creating the plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Scatter plot for Displacement vs MPG
axes[0].scatter(df['displacement'], df['mpg'], alpha=0.5, color='blue')
axes[0].set_title('Displacement vs MPG')
axes[0].set_xlabel('Displacement (cubic inches)')
axes[0].set_ylabel('Miles Per Gallon')
# Adding grid for better readability
axes[0].grid(True)  

# Scatter plot for Horsepower vs MPG
axes[1].scatter(df['horsepower'], df['mpg'], alpha=0.5, color='red')
axes[1].set_title('Horsepower vs MPG')
axes[1].set_xlabel('Horsepower')
# Adding grid for better readability
axes[1].grid(True)  

# Scatter plot for Weight vs MPG
axes[2].scatter(df['weight'], df['mpg'], alpha=0.5, color='green')
axes[2].set_title('Weight vs MPG')
axes[2].set_xlabel('Weight (lbs)')
# Adding grid for better readability
axes[2].grid(True)  

# Improve layout and save the plot
plt.tight_layout()
plt.savefig('q1_3_plot.png')

# Show the plot
plt.show()



# Question 1.4: Continuing with the data from question 1.3, create a scatter plot 
# with mpg on the y-axis and cylinders on the x-axis.  Explain what is wrong 
# with this plot with a 1-2 line comment.  Now create a box plot using Seaborn
# that uses cylinders as the groupings on the x-axis, and mpg as the values
# up the y-axis.

# Scatter plot for mpg against cylinders
plt.figure(figsize=(8, 6))
plt.scatter(df['cylinders'], df['mpg'], alpha=0.5)
plt.xlabel('Cylinders')
plt.ylabel('Miles Per Gallon (MPG)')
plt.title('Scatter Plot of MPG vs. Cylinders')
plt.savefig('q1_4a_scatter_plot.png')
plt.show()

# The scatter plot does not effectively represent the relationship between 'cylinders' and 'mpg'. 
# Points are overlaid on top of one another, making it hard to see the true density or variation in data. 

# Box plot for mpg against cylinders using Seaborn
plt.figure(figsize=(8, 6))
sns.boxplot(x='cylinders', y='mpg', data=df)
plt.xlabel('Cylinders')
plt.ylabel('Miles Per Gallon (MPG)')
plt.title('Box Plot of MPG vs. Cylinders')
plt.savefig('q1_4b_box_plot.png')
plt.show()



# Question 1.5: Continuing with the data from question 1.3, create a two-by-two 
# grid of subplots, where each one has mpg on the y-axis and one of 
# displacement, horsepower, weight, and acceleration on the x-axis.  To clean 
# up this plot:
#   - Remove the y-axis tick labels (the values) on the right two subplots - 
#     the scale of the ticks will already be aligned because the mpg values 
#     are the same in all axis.  
#   - Add a title to the figure (not the subplots) that reads "Changes in MPG"
#   - Add a y-label to the figure (not the subplots) that says "mpg"
#   - Add an x-label to each subplot for the x values
# Finally, use the savefig method to save this figure to your repo.  If any
# labels or values overlap other chart elements, go back and adjust spacing.

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# Adjust horizontal and vertical spaces between plots
fig.subplots_adjust(hspace=0.3, wspace=0.2)  

# Displacement vs. MPG
axs[0, 0].scatter(df['displacement'], df['mpg'], alpha=0.5)
axs[0, 0].set_xlabel('Displacement (cu inches)')
axs[0, 0].set_ylabel('MPG')

# Horsepower vs. MPG
axs[0, 1].scatter(df['horsepower'], df['mpg'], alpha=0.5)
axs[0, 1].set_xlabel('Horsepower')
axs[0, 1].set_ylabel('MPG')
# Remove y-axis tick labels
axs[0, 1].set_yticklabels([])  

# Weight vs. MPG
axs[1, 0].scatter(df['weight'], df['mpg'], alpha=0.5)
axs[1, 0].set_xlabel('Weight (lbs)')
axs[1, 0].set_ylabel('MPG')

# Acceleration vs. MPG
axs[1, 1].scatter(df['acceleration'], df['mpg'], alpha=0.5)
axs[1, 1].set_xlabel('Acceleration')
axs[1, 1].set_ylabel('MPG')
# Remove y-axis tick labels
axs[1, 1].set_yticklabels([])  

# Add a figure-wide title and labels
fig.suptitle('Changes in MPG', fontsize=16)
fig.text(0.5, 0.04, 'Vehicle Characteristics', ha='center', va='center', fontsize=12)
fig.text(0.04, 0.5, 'MPG', ha='center', va='center', rotation='vertical', fontsize=12)

# Save the figure
plt.savefig('q1_5_plot.png')

# Show the plot
plt.show()



# Question 1.6: Are cars from the USA, Japan, or Europe the least fuel
# efficient, on average?  Answer this with a plot and a one-line comment.

# Calculate average MPG by origin
average_mpg = df.groupby('origin')['mpg'].mean()

# Create a bar plot of average MPG by origin
plt.figure(figsize=(8, 6))
average_mpg.plot(kind='bar', color=['red', 'blue', 'green'])
plt.title('Average Fuel Efficiency by Car Origin')
plt.xlabel('Origin')
plt.ylabel('Average Miles Per Gallon (MPG)')
# Rotate the x labels to be horizontal
plt.xticks(rotation=0)  

# Save the plot
plt.savefig('q1_6_plot.png')

# Show the plot
plt.show()

# Cars from the USA are the least fuel-efficient on average compared to those from Europe and Japan.



# Question 1.7: Using Seaborn, create a scatter plot of mpg versus displacement,
# while showing dots as different colors depending on the country of origin.
# Explain in a one-line comment what this plot says about the results of 
# question 1.6.

# Create a scatter plot of mpg versus displacement colored by origin
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='displacement', y='mpg', hue='origin', palette='bright')
plt.title('MPG vs Displacement by Car Origin')
plt.xlabel('Displacement (cubic inches)')
plt.ylabel('Miles Per Gallon (MPG)')

# Save the plot
plt.savefig('q1_7_plot.png')

# Show the plot
plt.show()

# This plot reinforces the results of question 1.6 by showing that cars from the USA, 
# which generally have higher displacement, also tend to have lower MPG, indicating less fuel efficiency compared to cars from Japan and Europe.



# Question 2: The file unemp.csv contains the monthly seasonally-adjusted unemployment
# rates for US states from January 2020 to December 2022. Load it as a dataframe, as well
# as the data from the policy_uncertainty.xlsx file from homework 2 (you do not have to make
# any of the changes to this data that were part of HW2, unless you need to in order to 
# answer the following questions).
#    2.1: Merge both dataframes together

import pandas as pd

# Load the unemployment data from a CSV file
unemp_df = pd.read_csv('C:/Users/ChelseySSS/Desktop/unemp.csv')

# Convert 'DATE' in unemp_df to datetime for consistency
unemp_df['DATE'] = pd.to_datetime(unemp_df['DATE'])

# Load the policy uncertainty data from an Excel file
policy_df = pd.read_excel('C:/Users/ChelseySSS/Desktop/policy_uncertainty.xlsx')

# Create a 'DATE' column in policy_df
policy_df['DATE'] = pd.to_datetime(policy_df[['year', 'month']].assign(day=1))

# Standardize state names in policy_df to match those in unemp_df and uppercased
policy_df['state'] = policy_df['state'].str.upper()

# Merge the dataframes together on 'DATE' and 'STATE'
merged_df = pd.merge(unemp_df, policy_df, left_on=['DATE', 'STATE'], right_on=['DATE', 'state'], how='inner')

# Print out of the merged dataframe 
print(merged_df.head())
print("Unemployment data types:", unemp_df.dtypes)
print("Policy data types:", policy_df.dtypes)
print("Unique states in unemployment data:", unemp_df['STATE'].unique())
print("Unique states in policy data:", policy_df['state'].unique())



#    2.2: Calculate the log-first-difference (LFD) of the EPU-C data
#    2.2: Select five states and create one Matplotlib figure that shows the unemployment rate
#         and the LFD of EPU-C over time for each state. Save the figure and commit it with 
#         your code.

# Calculate log-first-difference of EPU_Composite
merged_df['EPU_Composite_log'] = np.log(merged_df['EPU_Composite'])
merged_df['EPU_Composite_LFD'] = merged_df.groupby('STATE')['EPU_Composite_log'].diff()

# Select five states for the analysis
selected_states = ['CALIFORNIA', 'TEXAS', 'NEW YORK', 'FLORIDA', 'ILLINOIS']

# Create a figure and axes with a subplot for each state
fig, axs = plt.subplots(len(selected_states), 1, figsize=(10, 20), sharex=True)

for index, state in enumerate(selected_states):
    state_data = merged_df[merged_df['STATE'] == state]

    # Create subplot for each state
    ax = axs[index]
    ax.plot(state_data['DATE'], state_data['unemp_rate'], label='Unemployment Rate', color='blue')
    ax.plot(state_data['DATE'], state_data['EPU_Composite_LFD'], label='LFD of EPU-C', color='red')
    ax.set_title(state)
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.legend(loc='upper left')

# Add a main title and adjust spacing
fig.suptitle('Unemployment Rate and LFD of EPU-C Over Time', fontsize=16)
plt.tight_layout(pad=3.0)
fig.subplots_adjust(top=0.95)  # Adjust the overall top margin

# Save the figure
plt.savefig('unemployment_epu_lfd_plot.png')

# Show the plot
plt.show()



#    2.3: Using statsmodels, regress the unemployment rate on the LFD of EPU-C and fixed
#         effects for states. Include an intercept.

# Dictionary to map full state names to abbreviations
state_map = {
    'ALASKA': 'AK', 'ALABAMA': 'AL', 'ARKANSAS': 'AR', 'ARIZONA': 'AZ', 'CALIFORNIA': 'CA',
    'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DISTRICT OF COLUMBIA': 'DC', 'DELAWARE': 'DE',
    'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IOWA': 'IA', 'IDAHO': 'ID',
    'ILLINOIS': 'IL', 'INDIANA': 'IN', 'KANSAS': 'KS', 'KENTUCKY': 'KY', 'LOUISIANA': 'LA',
    'MASSACHUSETTS': 'MA', 'MARYLAND': 'MD', 'MAINE': 'ME', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN',
    'MISSOURI': 'MO', 'MISSISSIPPI': 'MS', 'MONTANA': 'MT', 'NORTH CAROLINA': 'NC',
    'NORTH DAKOTA': 'ND', 'NEBRASKA': 'NE', 'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ',
    'NEW MEXICO': 'NM', 'NEVADA': 'NV', 'NEW YORK': 'NY', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
    'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
    'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT', 'VIRGINIA': 'VA',
    'VERMONT': 'VT', 'WASHINGTON': 'WA', 'WISCONSIN': 'WI', 'WEST VIRGINIA': 'WV', 'WYOMING': 'WY'
}

# Apply this map to the 'state' column in policy data
policy_df['state'] = policy_df['state'].map(state_map)

# Merging process
merged_df = pd.merge(unemp_df, policy_df, left_on=['DATE', 'STATE'], right_on=['DATE', 'state'], how='inner')
print("DataFrame after merging:", merged_df.shape)

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Calculate log-first-difference of EPU_Composite
merged_df['EPU_Composite_log'] = np.log(merged_df['EPU_Composite'] + 1)  # Adding 1 to avoid log(0)
merged_df['EPU_Composite_LFD'] = merged_df.groupby('STATE')['EPU_Composite_log'].diff()

# Drop any rows with NaN values that could affect the regression
merged_df.dropna(subset=['unemp_rate', 'EPU_Composite_LFD'], inplace=True)

# Print DataFrame size to confirm data is present
print("DataFrame size after processing:", merged_df.shape)

# Check if DataFrame is not empty
if not merged_df.empty:
    # Define and fit the model with fixed effects
    model = smf.ols('unemp_rate ~ EPU_Composite_LFD + C(STATE)', data=merged_df)
    results = model.fit()
    print(results.summary())



#    2.4: Print the summary of the results, and write a 1-3 line comment explaining the basic
#         interpretation of the results (e.g. coefficient, p-value, r-squared), the way you 
#         might in an abstract.

# The model has been fitted and results stored in 'results'
if 'results' in locals():
    print(results.summary())

# The regression analysis demonstrates that the log-first-difference of EPU_Composite has a statistically significant effect on the unemployment rate across different states. 
# The model accounts for 17.1% of the variability in unemployment rates (R-squared = 0.171), indicating a moderate explanatory power. 
# The F-statistic of 7.010 with a p-value of approximately 1.31e-41 confirms the overall statistical significance of the model, suggesting that changes in policy uncertainty are indeed influential factors affecting state-level unemployment rates.






import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
data = pd.read_csv('/kaggle/input/road-accident-severity-in-india/Road.csv')
data.sample(3)
warnings.filterwarnings("ignore")
data.head()
data.tail()
data.shape
data.info()
data.describe()
data.columns
data.isna().sum().reset_index()     		
pd.set_option('display.max_columns', None)
data.head()
data['Time'] = data['Time'].astype('datetime64[ns]')
data.loc[::, "Time"].reset_index()
data[["Casualty_class", "Sex_of_casualty", "Age_band_of_casualty", "Casualty_severity", "Work_of_casuality","Fitness_of_casuality"]].dropna()
age_band_counts = data['Age_band_of_driver'].value_counts()
color = ['lightblue', 'blue', 'green', 'red', 'yellow']
plt.figure(figsize=(12, 6)) 
plt.bar(age_band_counts.index, age_band_counts.values, color=color) 
plt.title('Number of Accidents by Age Band of Drivers') 
plt.xlabel('Age Band of Drivers') 
plt.ylabel('Number of Accidents') 
plt.xticks(rotation=45) #This line rotates the x-axis labels by 45 degrees for better readability.
plt.show() 
plt.figure(figsize=(12, 6))
sns.barplot(x=age_band_counts.index, y=age_band_counts.values, palette="husl")
plt.title('Number of Accidents by Age Band of Drivers')
plt.xlabel('Age Band of Drivers')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()
day_of_week_counts = data['Day_of_week'].value_counts()
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_of_week_counts = day_of_week_counts.reindex(days_order)
plt.figure(figsize=(10, 6))
sns.barplot(x=day_of_week_counts.index, y=day_of_week_counts, palette='viridis')
plt.title('Accidents by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Accidents')
plt.show()
education_counts = data['Educational_level'].value_counts()
most_common_education = education_counts.idxmax() 
plt.figure(figsize=(10, 6))
sns.countplot(x='Educational_level', data=data, order=data['Educational_level'].value_counts().index)
plt.title('Distribution of Educational Levels among Drivers Involved in Accidents')
plt.xlabel('Educational Level')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
data['Hour'] = data['Time'].dt.hour
data['Day_of_week'] = pd.Categorical(data['Day_of_week'], categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)
plt.figure(figsize=(12, 6))
sns.countplot(x='Hour', data=data, palette='rainbow')
plt.title('Accidents by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Accidents')
plt.show()
plt.figure(figsize=(12, 6))
sns.countplot(x='Day_of_week', data=data, palette='viridis')
plt.title('Accidents by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Accidents')
plt.show()
accident_due_vehicles = data["Type_of_vehicle"].reset_index() 
plt.figure(figsize = (12, 6))
sns.countplot(x='Type_of_vehicle',data=accident_due_vehicles,palette='Set2')
plt.xticks(rotation=45)
plt.show()
data['Weekday'] = data['Day_of_week'].apply(lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday')
weekday_weekend_counts = data.groupby(['Weekday', 'Accident_severity']).size().unstack().reset_index() 
print(weekday_weekend_counts)
find_cor = data[['Number_of_vehicles_involved', 'Number_of_casualties']]
correlation_mat = find_cor.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(data=correlation_mat, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
plt.title('Correlation Plot: Number of Vehicles Involved vs. Number of Casualties')
plt.show()
data['Time'] = pd.to_datetime(data['Time'], errors='coerce')
def categorize_part_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'
data['Part_of_Day'] = data['Time'].dt.hour.apply(categorize_part_of_day)
data["Part_of_Day"].value_counts()
grouped_data  = data.groupby('Driving_experience')['Accident_severity'].value_counts()
most_common_sources = grouped_data.groupby('Accident_severity').idxmax()
most_common_sources

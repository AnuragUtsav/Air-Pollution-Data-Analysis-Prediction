# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# =========================
# DATA LOADING
# =========================

df = pd.read_csv("C:/Users/dell/OneDrive/Documents/INT375/Project/Air Pollution.csv")

print("Initial Data:\n", df.head())

print("\nInfo:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())


# =========================
# DATA CLEANING
# =========================

# Fix date parsing (IMPORTANT FIX)
df['last_update'] = pd.to_datetime(df['last_update'], format='%d-%m-%Y %H:%M:%S', dayfirst=True)

# Drop missing important values
df = df.dropna(subset=['pollutant_avg'])

# Remove duplicates
df = df.drop_duplicates()

# Clean column names
df.columns = df.columns.str.lower()

# Fix names for better visuals
df['state'] = df['state'].str.replace('_', ' ')
df['city'] = df['city'].str.replace('_', ' ')

print("\nAfter Cleaning Shape:", df.shape)


# =========================
# EDA (ANALYSIS)
# =========================

print("\n--- TOP POLLUTED STATES ---")
state_pollution = df.groupby('state')['pollutant_avg'].mean().sort_values(ascending=False)
print(state_pollution.head(10))

print("\n--- TOP POLLUTED CITIES ---")
city_pollution = df.groupby('city')['pollutant_avg'].mean().sort_values(ascending=False)
print(city_pollution.head(10))

print("\n--- POLLUTANT TYPES ---")
pollutants = df['pollutant_id'].value_counts()
print(pollutants)

print("\n--- POLLUTION BY POLLUTANT TYPE ---")
pollutant_trend = df.groupby('pollutant_id')['pollutant_avg'].mean()
print(pollutant_trend)

print("\n--- CORRELATION ---")
print(df[['pollutant_min', 'pollutant_max', 'pollutant_avg']].corr())


# =========================
# VISUALIZATION (IMPROVED)
# =========================

# 1. Top States
plt.figure(figsize=(10,6))
top_states = state_pollution.head(10)
sns.barplot(x=top_states.values, y=top_states.index)
plt.title("Top 10 Most Polluted States")
plt.xlabel("Average Pollution")
plt.ylabel("State")
plt.show()


# 2. Top Cities
plt.figure(figsize=(10,6))
top_cities = city_pollution.head(10)
sns.barplot(x=top_cities.values, y=top_cities.index)
plt.title("Top 10 Most Polluted Cities")
plt.xlabel("Average Pollution")
plt.ylabel("City")
plt.show()


# 3. Pollutant Frequency
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='pollutant_id', order=df['pollutant_id'].value_counts().index)
plt.title("Pollutant Frequency Distribution")
plt.xticks(rotation=45)
plt.show()


# 4. Pollution by Pollutant Type (FIXED TREND)
plt.figure(figsize=(8,5))
pollutant_trend.sort_values(ascending=False).plot(kind='bar')
plt.title("Average Pollution by Pollutant Type")
plt.ylabel("Pollution Level")
plt.show()


# 5. Scatter Plot
plt.figure(figsize=(6,5))
sns.scatterplot(x=df['pollutant_min'], y=df['pollutant_max'])
plt.title("Min vs Max Pollution")
plt.show()


# 6. Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df[['pollutant_min','pollutant_max','pollutant_avg']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()


# =========================
# OUTLIER DETECTION
# =========================

col = 'pollutant_avg'

Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df[col] < lower) | (df[col] > upper)]

print("\nOutliers Count:", len(outliers))

plt.figure(figsize=(6,4))
sns.boxplot(x=df[col])
plt.title("Outlier Detection")
plt.show()


# =========================
# DASHBOARD
# =========================

plt.figure(figsize=(14,10))

plt.subplot(2,2,1)
top_states.plot(kind='bar')
plt.title("Top States")

plt.subplot(2,2,2)
top_cities.plot(kind='bar')
plt.title("Top Cities")

plt.subplot(2,2,3)
pollutant_trend.plot(kind='bar')
plt.title("Pollutant Types")

plt.subplot(2,2,4)
sns.heatmap(df[['pollutant_min','pollutant_max','pollutant_avg']].corr(), annot=True)

plt.tight_layout()
plt.show()


# =========================
# MACHINE LEARNING (SKLEARN)
# =========================

X = df[['pollutant_min', 'pollutant_max']]
y = df['pollutant_avg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score = r2_score(y_test, y_pred)

print("\nModel R2 Score:", score)


# Prediction Plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Pollution")
plt.ylabel("Predicted Pollution")
plt.title("Actual vs Predicted Pollution")
plt.show()

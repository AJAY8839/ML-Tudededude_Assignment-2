import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,LabelEncoder
import seaborn as sns
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestRegressor

# Load the Dataset
df = pd.read_csv('Global_Pollution_Analysis.csv')
# print(df.head())

# Handle Missing Values
# print(df.isnull().sum()) 
# here we can see that there is no missing value so we not have to use this fillna function
# df.fillna(df.mean(numeric_only=True),inplace=True)


# Encode Categorical Variables
le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'] )
le_year = LabelEncoder()
df['Year'] = le_year.fit_transform(df['Year'])
# print(df.head())

# Normalize Numerical Columns
scaler = StandardScaler()
df[['Air_Pollution_Index','Water_Pollution_Index','Soil_Pollution_Index','Population (in millions)','Energy_Consumption_Per_Capita (in MWh)']]  = scaler.fit_transform(df[['Air_Pollution_Index','Water_Pollution_Index','Soil_Pollution_Index','Population (in millions)','Energy_Consumption_Per_Capita (in MWh)']])
# print(df.head())

# Exploratory Data Analysis (EDA)

# Descriptive Statistics
# print(df.describe())
# print(df[['CO2_Emissions (in MT)','Industrial_Waste (in tons)']].describe())

# Correlation Analysis
# plt.figure(figsize=(10,6))
# sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
# plt.title("Correlation Matrix")
# plt.show()

# Visualizations

#1. Line Plot – Pollution Trend Over Years
# plt.figure(figsize=(10,6))
# sns.lineplot(x='Year',y='Air_Pollution_Index',data=df)
# plt.title("Air Pollution Trend Over Years")
# plt.xlabel('Year')
# plt.ylabel('Air Pollution Index')
# plt.grid(True)
# plt.show()


# 2. Box Plot – Country-wise Pollution Comparison
# plt.figure(figsize=(10,6))
# sns.boxenplot(x='Country',y='Air_Pollution_Index',data=df)
# plt.xticks(rotation=45)
# plt.title('Air Pollution Comparison Across Countries')
# plt.show()

# 3 .Bar Plot – Industrial Waste by Country
# plt.figure(figsize=(12,6))
# sns.barplot(x='Country', y='Industrial_Waste (in tons)', data=df)
# plt.xticks(rotation=45)
# plt.title("Industrial Waste by Country")
# plt.show()


# Feature Engineering

# Yearly Average Pollution
yearly_avg = df.groupby('Year')[['Air_Pollution_Index','Water_Pollution_Index','Soil_Pollution_Index','Population (in millions)','Energy_Consumption_Per_Capita (in MWh)']].mean().reset_index()
# print(df.head())

# plt.figure(figsize=(10,6))
# sns.lineplot(x='Year',y='Energy_Recovered (in GWh)',data=yearly_avg)
# plt.title("Year-wise Energy Recovery Trend")
# plt.grid(True)
# plt.show()

#  Energy per Capita
if 'Population (in millions)' in df.columns and 'Energy_Recovered (in GWh)' in df.columns:
    df['Energy_per_Capita'] = df['Energy_Recovered (in GWh)']*1e6/df['Population (in millions)']
    print(df[['Country', 'Year', 'Energy_per_Capita']].head())
else:
    print("Population column missing, skipping Energy per Capita feature.")

#  Predictive Modeling
# Linear Regression Model (for Pollution Prediction)
y = df['Energy_Recovered (in GWh)']
x = df.drop('Energy_Recovered (in GWh)', axis=1)


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)
# model = LinearRegression()
# model.fit(x_train,y_train)
model = RandomForestRegressor(random_state=42)
model.fit(x_train, y_train)


y_pridict = model.predict(x_test)

mse = mean_squared_error(y_test,y_pridict)
mae = mean_absolute_error(y_test,y_pridict)
r2s = r2_score(y_test,y_pridict)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2s)

# importances = model.feature_importances_
# features = x.columns

# plt.figure(figsize=(10,6))
# plt.barh(features, importances)
# plt.title("Feature Importance (Random Forest)")
# plt.xlabel("Importance Score")
# plt.show()

#  Logistic Regression Model (for Categorization of Pollution Levels)


#  Binary classification target
df['Pollution_Level_Binary'] = df['Air_Pollution_Index'].apply(lambda x: 1 if x > 70 else 0)

#  Features and target
X = df[['CO2_Emissions (in MT)', 'Industrial_Waste (in tons)', 'Water_Pollution_Index', 'Soil_Pollution_Index']]
y = df['Pollution_Level_Binary']

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#  Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#  Logistic Regression model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

#  Predictions and evaluation
y_pred = log_model.predict(X_test)

print(" Accuracy:", accuracy_score(y_test, y_pred))
print(" Precision:", precision_score(y_test, y_pred))
print(" Recall:", recall_score(y_test, y_pred))
print(" F1 Score:", f1_score(y_test, y_pred))

#  Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

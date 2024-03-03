import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('PS_20174392719_1491204439457_log.csv')

# Set style for seaborn
sns.set_style("whitegrid")

# Plotting the distribution of transaction types
plt.figure(figsize=(8, 5))
sns.countplot(x='type', data=df, order = df['type'].value_counts().index)
plt.title('Distribution of Transaction Types')
plt.ylabel('Number of Transactions')
plt.xlabel('Transaction Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting the distribution of transaction amounts, with a focus on non-fraudulent transactions
plt.figure(figsize=(10, 6))
sns.histplot(df[df['isFraud'] == 0]['amount'], bins=100, kde=False, color='blue', stat='density')
plt.xscale('log')
plt.title('Distribution of Transaction Amounts for Non-Fraudulent Transactions')
plt.xlabel('Amount (log scale)')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

# Plotting the distribution of transaction amounts for fraudulent transactions
plt.figure(figsize=(10, 6))
sns.histplot(df[df['isFraud'] == 1]['amount'], bins=100, kde=False, color='red', stat='density')
plt.xscale('log')
plt.title('Distribution of Transaction Amounts for Fraudulent Transactions')
plt.xlabel('Amount (log scale)')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

# Plotting the time-step (hour of the day) against the frequency of fraudulent transactions
plt.figure(figsize=(10, 6))
fraud_over_time = df[df['isFraud'] == 1]['step'].value_counts().sort_index()
sns.lineplot(x=fraud_over_time.index, y=fraud_over_time.values, color='red')
plt.title('Frequency of Fraudulent Transactions Over Time')
plt.xlabel('Hourly Time-Step')
plt.ylabel('Frequency of Fraudulent Transactions')
plt.tight_layout()
plt.show()

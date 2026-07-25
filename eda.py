import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")

# Load dataset
df = pd.read_csv("dataset/customer_churn_processed.csv")

print("Dataset Loaded Successfully")
print(df.head())


plt.figure(figsize=(6,5))

sns.countplot(x="Churn", data=df)

plt.title("Customer Churn Distribution")

plt.xlabel("Churn")

plt.ylabel("Number of Customers")

plt.show()

plt.figure(figsize=(14,10))

sns.heatmap(df.corr(),
            annot=False,
            cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.savefig("static/images/churn_distribution.png")
plt.show()

plt.figure(figsize=(8,5))

sns.histplot(df["MonthlyCharges"], bins=30)

plt.title("Monthly Charges Distribution")

plt.savefig("static/images/churn_distribution.png")
plt.show()

plt.figure(figsize=(8,5))

sns.histplot(df["tenure"], bins=30)

plt.title("Customer Tenure Distribution")

plt.show()

plt.figure(figsize=(7,5))

sns.boxplot(y=df["MonthlyCharges"])

plt.title("Monthly Charges Boxplot")

plt.savefig("static/images/churn_distribution.png")
plt.show()

sns.pairplot(df[
    [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn"
    ]
])

plt.savefig("static/images/churn_distribution.png")
plt.show()


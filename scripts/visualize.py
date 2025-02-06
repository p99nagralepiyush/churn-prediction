import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv('../data/telco_churn_cleaned.csv')

# Churn rate by tenure
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Churn Rate by Tenure')
plt.savefig('../outputs/eda_visualizations/churn_by_tenure.png')  # Save the plot
plt.show()

# Churn rate by MonthlyCharges
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Churn Rate by Monthly Charges')
plt.savefig('../outputs/eda_visualizations/churn_by_monthly_charges.png')  # Save the plot
plt.show()
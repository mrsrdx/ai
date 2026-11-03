# Practical 9: Association Rule Mining using Apriori

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Transaction dataset
transactions = [
    ['milk', 'bread', 'nuts'],
    ['milk', 'bread'],
    ['milk', 'eggs', 'nuts'],
    ['milk', 'bread', 'eggs'],
    ['bread', 'nuts']
]

# Convert transactions into a one-hot encoded table
items = sorted(set(item for row in transactions for item in row))
data = pd.DataFrame([
    {item: item in row for item in items}
    for row in transactions
])

# Find frequent itemsets
frequent_itemsets = apriori(data, min_support=0.4, use_colnames=True)
print("Frequent Itemsets:")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(
    frequent_itemsets, metric='confidence', min_threshold=0.6
)

print("Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence']])

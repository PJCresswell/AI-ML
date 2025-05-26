import pandas as pd

raw = pd.read_csv('datasets/survey-data.csv', encoding='ISO-8859-1')
row_count = len(raw)
new_data = raw.drop(columns=['Unnamed: 0'])
attributes = new_data.keys()
#print('Number of instances ' + str(row_count))
#print(new_data.keys())

print('Level 1 rules')
l1_rules = []
for feature in attributes:
    values = new_data[feature].unique()
    for value in values:
        result = new_data[new_data[feature] == value]
        count = len(result)
        if count > 500 :
            print('Rule: (' + str(feature) + ' == ' + str(value) + ') Coverage: ' + str(count))
            l1_rules.append([feature, value, count])

print('Level 2 rules')
l2_rules = []
for i in range (0, len(l1_rules)-1):
    for j in range(i+1, len(l1_rules)):
        rule1_pre = l1_rules[i][0]
        rule1_post = l1_rules[i][1]
        rule2_pre = l1_rules[j][0]
        rule2_post = l1_rules[j][1]
        result = new_data[(new_data[rule1_pre] == rule1_post) & (new_data[rule2_pre] == rule2_post)]
        count = len(result)
        if count > 500 :
            print('Rule: (' + str(rule1_pre) + ' == ' + str(rule1_post) + ') & (' + str(rule2_pre) + ' == ' + str(rule2_post) + ')')
            l2_rules.append([rule1_pre, rule1_post, rule2_pre, rule2_post])

rules_to_graph = []
print('Final association rules')
for i in range (0, len(l2_rules)):
    rule1_pre = l2_rules[i][0]
    rule1_post = l2_rules[i][1]
    rule2_pre = l2_rules[i][2]
    rule2_post = l2_rules[i][3]
    candidate1_LHS = new_data[new_data[rule1_pre] == rule1_post]
    candidate1_LHS_count = len(candidate1_LHS)
    candidate1_support = new_data[(new_data[rule1_pre] == rule1_post) & (new_data[rule2_pre] == rule2_post)]
    candidate1_support_count = len(candidate1_support)
    candidate1_confidence = candidate1_support_count / candidate1_LHS_count
    print('Rule: (' + str(rule1_pre) + ' == ' + str(rule1_post) + ') & (' + str(rule2_pre) + ' == ' + str(rule2_post) + ') Confidence: ' + str(candidate1_confidence))
    candidate2_LHS = new_data[new_data[rule2_pre] == rule2_post]
    candidate2_LHS_count = len(candidate2_LHS)
    candidate2_support = new_data[(new_data[rule2_pre] == rule2_post) & (new_data[rule1_pre] == rule1_post)]
    candidate2_support_count = len(candidate2_support)
    candidate2_confidence = candidate2_support_count / candidate2_LHS_count
    print('Rule: (' + str(rule2_pre) + ' == ' + str(rule2_post) + ') & (' + str(rule1_pre) + ' == ' + str(rule1_post) + ') Confidence: ' + str(candidate2_confidence))
    rules_to_graph.append( ( str(rule1_pre) + '==' + str(rule1_post), str(rule2_pre) + '==' + str(rule2_post) ) )

import networkx as nx
import matplotlib.pyplot as plt

list_of_rules = [('A', 'B'), ('C', 'D'), ('B', 'D')]
plt.figure()
g = nx.Graph()
g.add_edges_from(rules_to_graph)
nx.draw(g, node_size=3000, with_labels=True, node_color='lightgrey')
plt.show()






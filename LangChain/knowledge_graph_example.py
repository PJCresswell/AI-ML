# https://lopezyse.medium.com/knowledge-graphs-from-scratch-with-python-f3c2a05914cc

# https://python.langchain.com/docs/tutorials/graph/

import pandas as pd

# Define the heads, relations, and tails
head = ['drugA', 'drugB', 'drugC', 'drugD', 'drugA', 'drugC', 'drugD', 'drugE', 'gene1', 'gene2','gene3', 'gene4', 'gene50', 'gene2', 'gene3', 'gene4']
relation = ['treats', 'treats', 'treats', 'treats', 'inhibits', 'inhibits', 'inhibits', 'inhibits', 'associated', 'associated', 'associated', 'associated', 'associated', 'interacts', 'interacts', 'interacts']
tail = ['fever', 'hepatitis', 'bleeding', 'pain', 'gene1', 'gene2', 'gene4', 'gene20', 'obesity', 'heart_attack', 'hepatitis', 'bleeding', 'cancer', 'gene1', 'gene20', 'gene50']

# Create a dataframe
df = pd.DataFrame({'head': head, 'relation': relation, 'tail': tail})

import networkx as nx
import matplotlib.pyplot as plt

# Create a knowledge graph
G = nx.Graph()
for _, row in df.iterrows():
    G.add_edge(row['head'], row['tail'], label=row['relation'])

# Visualize the knowledge graph
pos = nx.spring_layout(G, seed=42, k=0.9)
labels = nx.get_edge_attributes(G, 'label')
plt.figure(figsize=(12, 10))
nx.draw(G, pos, with_labels=True, font_size=10, node_size=700, node_color='lightblue', edge_color='gray', alpha=0.6)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8, label_pos=0.3, verticalalignment='baseline')
plt.title('Knowledge Graph')
plt.show()

# Now for some analysis
# The first thing we can do with our KG is to see how many nodes and edges it has and analyze their relationship.
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
print(f'Number of nodes: {num_nodes}')
print(f'Number of edges: {num_edges}')
print(f'Ratio edges to nodes: {round(num_edges / num_nodes, 2)}')

# Node centrality measures the importance or influence of a node within a graph. It helps identify nodes that are central to the structure of the graph. Some of the most common centrality measures are:
# Degree centrality counts the number of edges incident on a node. Nodes with higher degree of centrality are more connected

degree_centrality = nx.degree_centrality(G)
for node, centrality in degree_centrality.items():
    print(f'{node}: Degree Centrality = {centrality:.2f}')

# Betweenness centrality measures how often a node lies on the shortest path between other nodes, or in other words, the influence of a node on the flow of information between other nodes. Nodes with high betweenness centrality can act as bridges between different parts of the graph.
betweenness_centrality = nx.betweenness_centrality(G)
for node, centrality in betweenness_centrality.items():
    print(f'Betweenness Centrality of {node}: {centrality:.2f}')

# Closeness centrality quantifies how quickly a node can reach all other nodes in the graph. Nodes with higher closeness centrality are considered more central because they can communicate with other nodes more efficiently.
closeness_centrality = nx.closeness_centrality(G)
for node, centrality in closeness_centrality.items():
    print(f'Closeness Centrality of {node}: {centrality:.2f}')

# Visualize centrality measures
plt.figure(figsize=(15, 10))

# Calculate centrality measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

# Degree centrality
plt.subplot(131)
nx.draw(G, pos, with_labels=True, font_size=10, node_size=[v * 3000 for v in degree_centrality.values()], node_color=list(degree_centrality.values()), cmap=plt.cm.Blues, edge_color='gray', alpha=0.6)
plt.title('Degree Centrality')

# Betweenness centrality
plt.subplot(132)
nx.draw(G, pos, with_labels=True, font_size=10, node_size=[v * 3000 for v in betweenness_centrality.values()], node_color=list(betweenness_centrality.values()), cmap=plt.cm.Oranges, edge_color='gray', alpha=0.6)
plt.title('Betweenness Centrality')

# Closeness centrality
plt.subplot(133)
nx.draw(G, pos, with_labels=True, font_size=10, node_size=[v * 3000 for v in closeness_centrality.values()], node_color=list(closeness_centrality.values()), cmap=plt.cm.Greens, edge_color='gray', alpha=0.6)
plt.title('Closeness Centrality')

plt.tight_layout()
plt.show()

# Shortest path analysis focuses on finding the shortest path between two nodes in the graph. This can help you understand the connectivity between different entities and the minimum number of relationships required to connect them. For example, let’s say you want to find the shortest path between the nodes ‘gene2’ and ‘cancer’

source_node = 'gene2'
target_node = 'cancer'

# Find the shortest path
shortest_path = nx.shortest_path(G, source=source_node, target=target_node)

# Visualize the shortest path
plt.figure(figsize=(10, 8))
path_edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]
nx.draw(G, pos, with_labels=True, font_size=10, node_size=700, node_color='lightblue', edge_color='gray', alpha=0.6)
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
plt.title(f'Shortest Path from {source_node} to {target_node}')
plt.show()
print('Shortest Path:', shortest_path)

# Create Embeddings
# Graph embeddings are mathematical representations of nodes or edges in a graph in a continuous vector space. These embeddings capture the structural and relational information of the graph, allowing us to perform various analyses, such as node similarity calculation and visualization in lower-dimensional space.
# Next, we’ll use the node2vec algorithm, which learns embeddings by performing random walks on the graph and optimizing to preserve the local neighborhood structure of nodes.
# In this code, the node2vec algorithm is used to learn 64-dimensional embeddings for nodes in your KG. The embeddings are then reduced to 2 dimensions using t-SNE (t-distributed Stochastic Neighbor Embedding) for visualization purposes. Each point on the resulting scatter plot corresponds to a node in the graph. Do you see how the disconnected subgraph is also represented separately in the vectorized space?

from node2vec import Node2Vec

# Generate node embeddings using node2vec
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4) # You can adjust these parameters
model = node2vec.fit(window=10, min_count=1, batch_words=4) # Training the model

# Visualize node embeddings using t-SNE
from sklearn.manifold import TSNE
import numpy as np

# Get embeddings for all nodes
embeddings = np.array([model.wv[node] for node in G.nodes()])

# Reduce dimensionality using t-SNE
tsne = TSNE(n_components=2, perplexity=10, n_iter=400)
embeddings_2d = tsne.fit_transform(embeddings)

# Visualize embeddings in 2D space with node labels
plt.figure(figsize=(12, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', alpha=0.7)

# Add node labels
for i, node in enumerate(G.nodes()):
    plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], node, fontsize=8)
plt.title('Node Embeddings Visualization')
plt.show()

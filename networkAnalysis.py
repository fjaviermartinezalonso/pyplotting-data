import networkx as nx 
import matplotlib.pyplot as plt
from scipy.stats import bernoulli 
import numpy as np 

output_path = "./output/"


# 1) Simple tutorial [toy example]
G = nx.Graph() 

G.add_node(1)
G.add_nodes_from([2,3])
G.add_nodes_from(['u','v'])

G.add_edge(1,2)
G.add_edge('u','v')
# If u add an edge between nodes that do not exist yet, Python creates them
G.add_edges_from([(1,3),(1,4),(1,5),(1,6)]) 
G.add_edge('u','w')
G.edges()
G.nodes()

G.remove_node(2)
G.remove_nodes_from([4,5])
G.nodes()

G.remove_edge(1,3)
G.remove_edges_from([(1,2),('u','v')])
G.nodes()
G.edges()
G.number_of_nodes()
G.number_of_edges()



# 2) Get the karate club data with members (nodes) and friendships (edges)
G = nx.karate_club_graph()
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
plt.savefig(output_path + 'network_karateGraph.png')

G.degree() # Key: node number, Value: degree
G.degree()[33] # Returns a DegreeView object and then checks the given index
G.degree(33) # Returns just the result



# 3) Generate an ER graph
# Generate a flip coin with Bernoulli (scipy)
def er_graph(N,p):
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 < node2 and bernoulli.rvs(p=p):  # If it is 1, create an edge
                G.add_edge(node1,node2)
    return G

G = er_graph(50,0.08)
nx.draw(G, node_size=40, node_color='gray')
plt.savefig(output_path + 'network_random_graph.png')
print(G.number_of_nodes())
print(G.number_of_edges())

def plot_degree_distribution(G):
    degree_sequence = [d for n, d in G.degree()]
    plt.hist(degree_sequence, histtype="step")
    plt.xlabel("Degree $k$")
    plt.ylabel("P(k)")
    plt.title("Degree distribution")
    
plt.figure(figsize=(10,10))
plot_degree_distribution(er_graph(500,0.08))
plot_degree_distribution(er_graph(500,0.08))
plot_degree_distribution(er_graph(500,0.08))
plt.savefig(output_path + "network_degree_distribution.png")



# 4) Descriptive statistics of empirical social network
A1 = np.loadtxt("PythonScripts/usingPythonForResearch/data/adj_allVillageRelationships_vilno_1.csv", delimiter=",")
A2 = np.loadtxt("PythonScripts/usingPythonForResearch/data/adj_allVillageRelationships_vilno_2.csv", delimiter=",")
G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)

def basic_net_stats(G):
    print("Number of nodes: %d" % G.number_of_nodes())
    print("Number of edges: %d" %G.number_of_edges())
    degree_sequence = [d for n,d in G.degree()]
    print("Avg degree: %.2f" % np.mean(degree_sequence))

basic_net_stats(G1)
basic_net_stats(G2)

plt.figure(figsize=(10,10))
plot_degree_distribution(G1)
plot_degree_distribution(G2)
plt.savefig(output_path + "network_degree_distribution_villages.png")

# The result shows that the histograms are not symmetric, meaning that
# ER graphs are not good for reallistic social networks. Then, there are another graphs better for this



# 5) Find the largest connected component
gen = nx.connected_component_subgraphs(G1) # genertor object
g = gen.__next__() # generators create objects with the __next__() function
type(g) # networkx.classes.graph.Graph
# each time that __next__() is called, the next component of the generator is get, until gen runs out of components
g.number_of_nodes() # the same as len(gen.__next__())
g.number_of_edges()
# each component has some number of nodes, and those components are sorted arbitrary

# This is easier with max function
G1_LLC = max(nx.connected_component_subgraphs(G1), key=len)
G2_LLC = max(nx.connected_component_subgraphs(G2), key=len)
G1_LLC.number_of_nodes() / G1.number_of_nodes()
G2_LLC.number_of_nodes() / G2.number_of_nodes()

plt.figure(figsize=(10,10))
nx.draw(G1_LLC, node_color="red", edge_color="gray", node_size=20)
plt.savefig(output_path + "network_village1.png")

plt.figure(figsize=(10,10))
nx.draw(G2_LLC, node_color="green", edge_color="gray", node_size=20)
plt.savefig(output_path + "network_village2.png")

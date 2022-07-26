import torch
import matplotlib.pyplot as plt

torch.manual_seed(1)
m1=torch.randint(0, 3, (5, 15))
m2=torch.randint(0, 3, (15, 10))
m3=torch.randint(0, 3, (10, 5))
m4=torch.randint(0, 3, (5, 1))
mask=[]
mask.append(m1)
mask.append(m2)
mask.append(m3)
mask.append(m4)
print(m4)
nodelevel=0
odd=0


"""
# Defining a Class
class GraphVisualization:

    def __init__(self):
        # visual is a list which stores all
        # the set of edges that constitutes a
        # graph
        self.visual = []

    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)

    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list
    # nx.draw_networkx(G) - plots the graph
    # plt.show() - displays the graph
    def visualize(self):
        G = nx.Graph()
        G.add_edges_from(self.visual)
        nx.draw_networkx(G)
        plt.show()
temp="l"

G=nx.Graph()
visual=[]
for k in range(len(mask)):
    if (odd % 2 == 0):
        G.add_nodes_from(range(nodelevel,nodelevel+mask[k].shape[0]), layer=k)
        nodelevel +=mask[k].shape[0]
        G.add_nodes_from(range(nodelevel,nodelevel+mask[k].shape[1]), layer=k+1 )
        nodelevel += mask[k].shape[1]
    else:
        G.add_nodes_from(range(nodelevel, nodelevel + mask[k].shape[1]), layer=k+1)
    odd+=1
    for i in range(mask[k].shape[0]):
        for j in range(mask[k].shape[1]):
            if not torch.equal(mask[k][i][j],torch.tensor(0)):
                temp1=0
                temp2=0
                if k>0:
                    for z in range(k):
                        temp1+=mask[z].shape[0]
                    for z in range(k):
                        temp2+=mask[z].shape[0]
                temp2+=mask[k].shape[0]
                temp1+=i
                temp2+=j
                if(torch.equal(mask[k][i][j],torch.tensor(1))):
                    color1="g"
                else:
                    color1 ="r"
                G.add_edge(temp1,temp2,color=color1,weight=1)

pos = nx.multipartite_layout(G, subset_key="layer")
plt.figure(figsize=(8, 8))
colors = nx.get_edge_attributes(G,'color').values()
weights = nx.get_edge_attributes(G,'weight').values()

nx.draw_networkx(G,pos,edge_color=colors,
        width=list(weights), with_labels=False)
#plt.show()
"""
plt.imshow(m3, cmap='hot', interpolation='nearest')
plt.show()

"""
import itertools
import matplotlib.pyplot as plt
import networkx as nx

subset_sizes = [5, 5, 4, 3, 2, 4, 4, 3]
subset_color = [
    "gold",
    "violet",
    "violet",
    "violet",
    "violet",
    "limegreen",
    "limegreen",
    "darkorange",
]


def multilayered_graph(*subset_sizes):
    extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
    layers = [range(start, end) for start, end in extents]
    G = nx.Graph()
    for (i, layer) in enumerate(layers):
        print(layer,i)
        G.add_nodes_from(layer, layer=i)
    for layer1, layer2 in nx.utils.pairwise(layers):
        print(layer1)
        G.add_edges_from(itertools.product(layer1, layer2))
    return G


G = multilayered_graph(*subset_sizes)
color = [subset_color[data["layer"]] for v, data in G.nodes(data=True)]
pos = nx.multipartite_layout(G, subset_key="layer")
plt.figure(figsize=(8, 8))
nx.draw(G, pos, node_color=color, with_labels=False)
plt.axis("equal")
plt.show()
"""
import matplotlib.pyplot as plt
import networkx as nx

def draw_tree(tree, parent_name, pos=None, level=0, width=2., vert_gap=0.4, xcenter=0.5):
    if pos is None:
        pos = {parent_name: (xcenter, 1 - level * vert_gap)}
        width *= 0.5
        nextx = xcenter - width / 2
        for child in tree.get(parent_name, []):
            nextx += width
            pos.update(draw_tree(tree, child, pos=pos, level=level + 1, width=width, xcenter=nextx))
    return pos

def plot_tree(tree):
    G = nx.DiGraph()
    for parent, children in tree.items():
        for child in children:
            G.add_edge(parent, child)
    
    pos = draw_tree(tree, 'root')
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
    plt.show()

# Example binary tree
tree = {
    'root': ['A', 'B'],
    'A': ['C', 'D'],
    'B': ['E', 'F'],
    'C': [],
    'D': [],
    'E': [],
    'F': []
}

plot_tree(tree)

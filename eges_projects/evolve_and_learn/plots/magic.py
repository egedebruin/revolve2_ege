import matplotlib.pyplot as plt
import networkx as nx

def manhattan_dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Grid specification (top to bottom, left to right)
grid = [
    ['B', 'J', 'J', 'B', 'J', 'B'],
    ['_', '_', '_', 'J', '_', '_'],
    ['_', '_', '_', 'B', '_', '_']
]

rows = len(grid)
cols = len(grid[0])

# Map from (x, y) to index for joints only
joint_positions = []
joint_indices = {}

index = 0
for y in range(rows):
    for x in range(cols):
        if grid[y][x] == 'J':
            pos = (x, rows - 1 - y)  # flip y for plot orientation
            joint_positions.append(pos)
            joint_indices[pos] = index
            index += 1

# List of all module positions for Manhattan distance (J and B)
valid_positions = set()
for y in range(rows):
    for x in range(cols):
        if grid[y][x] in ['J', 'B']:
            pos = (x, rows - 1 - y)
            valid_positions.add(pos)

# Build graph
G = nx.DiGraph()
edge_styles = {}

for i, pos_i in enumerate(joint_positions):
    xi = f'x{i}'
    yi = f'y{i}'

    # Nodes
    G.add_node(xi, pos=(pos_i[0] + 0.1, pos_i[1] + 0.1), color='skyblue')
    G.add_node(yi, pos=(pos_i[0] - 0.1, pos_i[1] - 0.1), color='lightgreen')

    # Internal y -> x
    G.add_edge(yi, xi)
    edge_styles[(yi, xi)] = ('gray', 'solid')

    # Neighbor x_j -> x_i connections
    for j, pos_j in enumerate(joint_positions):
        if i != j:
            # Only allow paths that go through valid module cells (J or B)
            dist = manhattan_dist(pos_i, pos_j)
            if dist == 1 or dist == 2:
                # Check that the path doesn't go through '_'
                # Simplified assumption: if both ends are in valid_positions and dist is 1 or 2, it's allowed
                if pos_j in valid_positions:
                    xj = f'x{j}'
                    color = 'blue' if dist == 1 else 'red'
                    style = 'solid' if dist == 1 else 'dashed'
                    G.add_edge(xj, xi)
                    edge_styles[(xj, xi)] = (color, style)

# Draw
pos = nx.get_node_attributes(G, 'pos')
colors = [data['color'] for _, data in G.nodes(data=True)]

plt.figure(figsize=(6, 5))

nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=800)
nx.draw_networkx_labels(G, pos, font_size=16)

for (u, v), (color, style) in edge_styles.items():
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(u, v)],
        edge_color=color,
        style=style,
        connectionstyle='arc3,rad=0.2',
        arrows=True
    )

plt.axis('off')
plt.tight_layout()
plt.show()

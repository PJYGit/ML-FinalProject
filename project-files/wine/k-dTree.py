from math import sqrt
import networkx
import matplotlib.pyplot as plt

dataset = [(9235, 2700), (1800, 1315), (1250, 1050), (7000, 2200), (3800, 1800), (4000, 1900), (800, 960)]

# create the k-d tree
def createTree(dataset, level=0, split_value=2):
    temp_dataset = dataset[:]

    split_value = level % split_value
    temp_dataset.sort(key=lambda x: x[split_value])

    level += 1
    length = len(dataset)
    if length == 0:
        return None
    elif length == 1:
        return {'Node': dataset[0], 'Level': level, 'Median': split_value, 'Left': None, 'Right': None}
    elif length != 1:
        midian = length // 2
        left_tree = temp_dataset[:midian]
        right_tree = temp_dataset[midian + 1:]
        return {'Node': temp_dataset[midian], 'Level': level, 'Median': split_value,
                'Left': createTree(left_tree, level),
                'Right': createTree(right_tree, level)}


# find the initial node
def descend_tree(kdTree, q, init_descend_tree=[]):
    temp_tree = init_descend_tree[:]
    if not kdTree:
        return temp_tree
    elif kdTree['Left'] is None:
        temp_tree.append(kdTree['Node'])
        return temp_tree
    elif kdTree['Left']:
        node = kdTree['Node']
        median = kdTree['Median']
        temp_tree.append(node)
        if q[median] <= node[median]:
            return descend_tree(kdTree['Left'], q, temp_tree)
        elif q[median] > node[median]:
            return descend_tree(kdTree['Right'], q, temp_tree)


def find_nearest(descended_tree, q, searched=[], best_distance=float('inf'), best_node=None):
    temp_tree = descended_tree[:]
    temp_searched = searched[:]

    if best_node is None:
        best_node = temp_tree[-1]

    if len(temp_tree) == 1:
        return best_node
    else:
        node = findNode(kdTree, temp_tree[-1])

        if get_distance(node['Node'], q) < best_distance:
            best_distance = get_distance(node['Node'], q)
            best_node = node['Node']

        parent = findNode(kdTree, temp_tree[-2])
        p_node = parent['Node']
        p_median = parent['Median']

        if get_distance(parent['Node'], q) < best_distance:
            best_distance = get_distance(parent['Node'], q)
            best_node = parent['Node']

        if node == parent['Left']:
            sibling = parent['Right']
        elif node == parent['Right']:
            sibling = parent['Left']

        if (sibling is None or sibling['Node'] in temp_searched or
                abs(p_node[p_median] - q[p_median]) > best_distance):
            searched_node = temp_tree.pop()
            temp_searched.append(searched_node)
            return find_nearest(temp_tree, q, temp_searched, best_distance, best_node)
        else:
            searched_node = temp_tree.pop()
            temp_searched.append(searched_node)
            new_tree = descend_tree(sibling, q)
            temp_tree.extend(new_tree)
            return find_nearest(temp_tree, q, temp_searched, best_distance, best_node)


# get the distance
def get_distance(node1, node2):
    d1 = node1[0] - node2[0]
    d2 = node1[1] - node2[1]

    distance = sqrt(d1 ** 2 + d2 ** 2)

    return distance


def findNode(Tree, node_value):
    if Tree is not None and Tree['Node'] == node_value:
        return Tree
    else:
        if Tree['Left'] is not None:
            return findNode(Tree['Left'], node_value) or findNode(Tree['Right'], node_value)


def create_graph(graph, node, pos={}, x=0, y=0, layer=1):
    value = dataset.index(node['Node']) + 1
    pos[value] = (x, y)
    if node['Left']:
        graph.add_edge(value, dataset.index(node['Left']['Node']) + 1)
        l_x, l_y = x - 1 / 2 ** layer, y - 1
        l_layer = layer + 1
        create_graph(graph, node['Left'], x=l_x, y=l_y, pos=pos, layer=l_layer)
    if node['Right']:
        graph.add_edge(value, dataset.index(node['Right']['Node']) + 1)
        r_x, r_y = x + 1 / 2 ** layer, y - 1
        r_layer = layer + 1
        create_graph(graph, node['Right'], x=r_x, y=r_y, pos=pos, layer=r_layer)
    return graph, pos


def draw(node):
    graph = networkx.DiGraph()
    graph, pos = create_graph(graph, node)
    fig, ax = plt.subplots(figsize=(8, 10))
    networkx.draw_networkx(graph, pos, ax=ax, node_size=500)
    plt.show()


kdTree = createTree(dataset)
draw(kdTree)
q = eval(input('Input the query node in format of RENT,SIZE: '))
nearest = find_nearest(descend_tree(kdTree, q), q)
ID = dataset.index(nearest) + 1
print('The ID of the nearest neighbor is:', ID, '---', nearest)


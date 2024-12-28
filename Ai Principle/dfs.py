import networkx as nx 
import matplotlib.pyplot as plt
from collections import deque

  """
def uninformed_path_finder(cities, roads, start_city, goal_city, strategy):
  
    Parameters:
    - cities: List of city names.
    - roads: Dictionary with city connections as {city: [(connected_city, distance)]}.
    - start_city: The city to start the journey.
    - goal_city: The destination city (for specific tasks).
    - strategy: The uninformed search strategy to use ('bfs' or 'dfs').
    
    Returns:
    - path: List of cities representing the path from start_city to goal_city.
    - cost: Total cost (number of steps or distance) of the path.

    
    # BFS implementation
    def bfs(goal_city):
        queue = deque([start_city])
        visited = {start_city: None}
        
        while queue:
            current = queue.popleft()
            if current == goal_city:
                break
            
            for neighbor, distance in roads[current]:
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)
        
        # Reconstruct the path
        path = []
        while goal_city is not None:
            path.append(goal_city)
            goal_city = visited[goal_city]
        path.reverse()
        
        return path, len(path) - 1  # Cost is number of edges

    # DFS implementation
    def dfs(current, visited):
        if current == goal_city:
            return [current], 0  # Found the goal
        
        visited.add(current)
        for neighbor, distance in roads[current]:
            if neighbor not in visited:
                path, cost = dfs(neighbor, visited)
                if path:
                    return [current] + path, cost + 1  # Cost is number of edges
        
        visited.remove(current)
        return None, float('inf')  # No path found

    if strategy == 'bfs':
        return bfs(goal_city)
    elif strategy == 'dfs':
        return dfs(start_city, set())
    else:
        raise ValueError("Unknown strategy: Choose 'bfs' or 'dfs'.")

# Example usage
cities = ['Addis Ababa', 'Bahir Dar', 'Gondar', 'Hawassa', 'Mekelle']
roads = {
    'Addis Ababa': [('Bahir Dar', 510), ('Hawassa', 275)],
    'Bahir Dar': [('Addis Ababa', 510), ('Gondar', 180)],
    'Gondar': [('Bahir Dar', 180), ('Mekelle', 300)],
    'Hawassa': [('Addis Ababa', 275)],
    'Mekelle': [('Gondar', 300)]
}

# Test the function with BFS
path, cost = uninformed_path_finder(cities, roads, 'Addis Ababa', 'Gondar', 'bfs')
print("BFS Path:", path, "Cost:", cost)

# Visualize the path
def visualize_path(cities, roads, path):
    G = nx.Graph()
    for city, connections in roads.items():
        for connected_city, distance in connections:
            G.add_edge(city, connected_city, weight=distance)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    # Highlight the path found by BFS
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

    plt.title("Ethiopian Road Network with BFS Path Highlighted")
    plt.show()

# Visualize the found path
if path:
    visualize_path(cities, roads, path)
else:
    print("No path found.")
 """

roads = {
    'Addis Ababa': [('Bahir Dar', 510), ('Hawassa', 275)],
    'Bahir Dar': [('Addis Ababa', 510), ('Gondar', 180)],
    'Gondar': [('Bahir Dar', 180), ('Mekelle', 300)],
    'Hawassa': [('Addis Ababa', 275)],
    'Mekelle': [('Gondar', 300)]
}
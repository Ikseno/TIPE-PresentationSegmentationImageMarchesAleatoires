"""Graphe non orienté pondéré stocké sous forme de liste d'adjacence."""


class Graph:
    def __init__(self):
        # Chaque clé est un nœud, la valeur associée une liste de tuples (voisin, poids)
        self.adjacency_list = {}

    def add_edge(self, node1, node2, weight):
        """Ajoute une arête non orientée entre node1 et node2."""
        self.adjacency_list.setdefault(node1, []).append((node2, weight))
        self.adjacency_list.setdefault(node2, []).append((node1, weight))

    def get_neighbors(self, node):
        """Retourne la liste des voisins (voisin, poids) du nœud."""
        return self.adjacency_list.get(node, [])

    def __str__(self):
        return str(self.adjacency_list)

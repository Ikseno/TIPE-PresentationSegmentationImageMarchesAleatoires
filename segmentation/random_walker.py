"""Segmentation d'image semi-automatique par marches aléatoires.

Principe : l'image est modélisée par un graphe de pixels (4-connexité) dont les arêtes
sont pondérées par la similarité de couleur. Pour chaque pixel non marqué, on lance une
marche aléatoire qui s'arrête dès qu'elle atteint un pixel déjà étiqueté (seed) ; le
pixel de départ prend alors l'étiquette atteinte et devient lui-même une seed.
"""

import random
from math import exp, sqrt

import numpy as np
from PIL import Image

from .graph import Graph


def hex_to_rgb(hex_color):
    """'#RRGGBB' -> (R, G, B)"""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[k:k + 2], 16) for k in (0, 2, 4))


class RandomWalkerSegmentation:
    """
    Paramètres
    ----------
    seeds : dict {(y, x): '#RRGGBB'}
        Pixels étiquetés à la main (ou par le CNN).
    image : str | PIL.Image.Image
        Chemin vers l'image ou image PIL déjà chargée.
    height, width : int
        Taille à laquelle l'image est redimensionnée.
    max_steps : int
        Nombre maximal de pas d'une marche (compromis précision / temps).
    sigma : float
        Écart-type de la gaussienne de pondération (≈ 7-8 pour 'rgb', ≈ 2-3 pour 'patch').
    weight_method : 'rgb' | 'patch'
        'rgb'   : distance euclidienne entre les couleurs des deux pixels (méthode 1)
        'patch' : distance entre les couleurs moyennes d'un voisinage (méthode 2,
                  plus robuste aux textures)
    patch_radius : int
        Rayon du voisinage utilisé par la méthode 'patch'.
    """

    def __init__(self, seeds, image, height=256, width=256, max_steps=40000,
                 sigma=7, weight_method="rgb", patch_radius=6):
        img = Image.open(image) if isinstance(image, str) else image
        self.height = height
        self.width = width
        self.img = img.convert("RGB").resize((width, height), Image.BICUBIC)
        self.img_array = np.array(self.img)

        self.max_steps = max_steps
        self.sigma = sigma
        self.weight_method = weight_method
        self.patch_radius = patch_radius

        self.seeds = seeds
        self.marche_convergente = 0
        self.graphe_image = Graph()

        self.segmented_image = [[0] * width for _ in range(height)]
        for (y, x), label in seeds.items():
            self.segmented_image[y][x] = label

        self.ajout_poids_graphe()

    # ------------------------------------------------------------------ poids

    def moyenne_patch(self, y, x):
        """Couleur RGB moyenne du carré de rayon patch_radius centré sur (y, x)."""
        r = self.patch_radius
        patch = self.img_array[max(0, y - r):min(self.height, y + r + 1),
                               max(0, x - r):min(self.width, x + r + 1)]
        return patch.mean(axis=(0, 1))

    def distance_couleur(self, p1, p2):
        if self.weight_method == "patch":
            desc1 = self.moyenne_patch(*p1)
            desc2 = self.moyenne_patch(*p2)
        else:
            desc1 = self.img_array[p1].astype(np.int16)
            desc2 = self.img_array[p2].astype(np.int16)
        return float(np.linalg.norm(desc1 - desc2))

    def poids_gauss(self, dist):
        """Poids gaussien : proche de 1 pour des couleurs similaires, de 0 sinon."""
        return exp(-(dist ** 2) / (self.sigma ** 2))

    def ajout_poids_graphe(self):
        """Relie chaque pixel à ses voisins de droite et du dessous."""
        for i in range(self.height):
            for j in range(self.width):
                if j < self.width - 1:
                    w = self.poids_gauss(self.distance_couleur((i, j), (i, j + 1)))
                    self.graphe_image.add_edge((i, j), (i, j + 1), w)
                if i < self.height - 1:
                    w = self.poids_gauss(self.distance_couleur((i, j), (i + 1, j)))
                    self.graphe_image.add_edge((i, j), (i + 1, j), w)

    # --------------------------------------------------------- marche aléatoire

    def marche_aleatoire(self, origin):
        cur_pixel = origin
        neighbors = self.graphe_image.get_neighbors(cur_pixel)

        for _ in range(self.max_steps):
            # Choix d'un voisin avec une probabilité proportionnelle au poids de l'arête
            cur_pixel = random.choices(
                population=[n[0] for n in neighbors],
                weights=[n[1] for n in neighbors],
            )[0]

            label = self.segmented_image[cur_pixel[0]][cur_pixel[1]]
            if label != 0:  # pixel déjà étiqueté : la marche converge
                self.marche_convergente += 1
                self.segmented_image[origin[0]][origin[1]] = label
                return

            neighbors = self.graphe_image.get_neighbors(cur_pixel)

        # Marche non convergente : étiquette de la seed la plus proche géométriquement
        nearest = min(self.seeds, key=lambda s: sqrt((origin[0] - s[0]) ** 2 + (origin[1] - s[1]) ** 2))
        self.segmented_image[origin[0]][origin[1]] = self.seeds[nearest]

    def color_all_pixels(self, verbose=True):
        """Lance une marche depuis chaque pixel non marqué, dans un ordre aléatoire."""
        all_pixels = [(i, j) for i in range(self.height) for j in range(self.width)]
        random.shuffle(all_pixels)
        n = len(all_pixels)
        for k, (i, j) in enumerate(all_pixels, 1):
            if verbose and k in (n // 4, n // 2, 3 * n // 4):
                print(f"{round(100 * k / n)}%")
            if (i, j) not in self.seeds:
                self.marche_aleatoire((i, j))
        if verbose:
            print("100%")
            print(f"Nombre de marches convergentes : {self.marche_convergente}")

    # ----------------------------------------------------------------- sortie

    def label_array(self):
        """Tableau (H, W) des étiquettes '#RRGGBB'."""
        return np.array(self.segmented_image, dtype=object)

    def to_image(self):
        arr = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for i in range(self.height):
            for j in range(self.width):
                arr[i, j] = hex_to_rgb(self.segmented_image[i][j])
        return Image.fromarray(arr)

    def save_image(self, path, show=False):
        img = self.to_image()
        img.save(path)
        if show:
            img.show()

    def get_stats_nb_marches(self):
        return (self.max_steps, self.marche_convergente)


# Ancien nom conservé pour compatibilité avec le code du TIPE
Image_Segmentation = RandomWalkerSegmentation

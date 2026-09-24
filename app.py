"""Interface Tkinter : placer des seeds à la main, les sauvegarder et segmenter l'image."""
from segmentation import RandomWalkerSegmentation
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import ast
import json
import os
import time
from datetime import datetime

# Nom du fichier unique où seront stockés tous les presets
ROOT = os.path.dirname(os.path.abspath(__file__))
PRESETS_FILENAME = os.path.join(ROOT, "data", "presets.json")
OUTPUT_DIR = os.path.join(ROOT, "results")
LOG_FILE = os.path.join(OUTPUT_DIR, "logs", "log_segment_times.txt")

class UI:
    def __init__(self, root):
        self.root = root
        self.image_path = os.path.join(ROOT, "data", "images", "cameleon.jpg")
        self.width = 256
        self.height = 256
        self.selected_color = None
        self.selected_color_code = None

        # Dictionnaire local : { (y, x): '#RRGGBB' }
        self.seeds = {}

        # Lecture du fichier JSON global des presets 
        self.all_presets = self.load_all_presets()

        # Paramètres pour la loupe
        self.magnifier_size = 50  # taille de la zone à extraire
        self.magnifier_scale = 2  # facteur de zoom (x2, x3, etc.)

    def load_all_presets(self):
        """Charge le contenu de presets.json (ou renvoie un dict vide si absent)."""
        if os.path.exists(PRESETS_FILENAME):
            with open(PRESETS_FILENAME, "r") as f:
                return json.load(f)  # Structure attendue: { "presetName": { "(y,x)": "#RRGGBB", ... }, ... }
        else:
            return {}

    def save_all_presets(self):
        """Sauvegarde le dictionnaire self.all_presets dans presets.json."""
        with open(PRESETS_FILENAME, "w") as f:
            json.dump(self.all_presets, f, indent=2)
        print(f"Tous les presets ont été sauvegardés dans {PRESETS_FILENAME}.")

    # =============== PARTIE INTERFACE ===============

    def run_ui(self):
        # Frame de gauche (boutons)
        self.color_frame = tk.Frame(self.root)
        self.color_frame.pack(side="left", fill="y", padx=5, pady=5)

        # Bouton pour choisir l'image
        self.choose_image_button = tk.Button(
            self.color_frame, text="Choisir une image", command=self.choose_image
        )
        self.choose_image_button.pack(pady=5)

        # Bouton choisir couleur
        self.choose_color_button = tk.Button(self.color_frame, text="Choisir couleur",
                                             command=self.create_color_palette)
        self.choose_color_button.pack(pady=5)

        # Bouton segmenter
        self.segment_button = tk.Button(self.color_frame, text="Segmenter l'image",
                                        command=self.segmentImage)
        self.segment_button.pack(pady=5)

        # Bouton sauver
        self.save_button = tk.Button(self.color_frame, text="Sauvegarder preset",
                                     command=self.show_save_preset_window)
        self.save_button.pack(pady=5)

        # Bouton charger
        self.load_button = tk.Button(self.color_frame, text="Charger preset",
                                     command=self.show_load_preset_window)
        self.load_button.pack(pady=5)

        # Bouton Reset Seeds
        self.reset_button = tk.Button(
            self.color_frame, text="Reset Seeds", command=self.reset_seeds
        )
        self.reset_button.pack(pady=5)

        # Label couleur sélectionnée
        self.color_label = tk.Label(self.color_frame, text="Aucune couleur sélectionnée")
        self.color_label.pack(pady=5)

        # Canvas pour l'image
        self.image_canvas = tk.Canvas(self.root, width=self.width, height=self.height)
        self.image_canvas.pack(side="left")

        # Charger l'image
        img = Image.open(self.image_path).resize((self.width, self.height), Image.BICUBIC)
        self.display_img = img  # Conserver l'objet PIL
        self.photo = ImageTk.PhotoImage(img)
        self.image_canvas.create_image(0, 0, anchor="nw", image=self.photo)

        # Clic sur l'image
        self.image_canvas.bind("<Button-1>", self.on_click)

        # Canvas loupe
        self.magnifier_canvas = tk.Canvas(self.root,
                                          width=self.magnifier_size * self.magnifier_scale,
                                          height=self.magnifier_size * self.magnifier_scale)
        self.magnifier_canvas.pack(side="left", padx=5)

        # Mouvement souris = loupe
        self.image_canvas.bind("<Motion>", self.show_magnifier)

    # ======================= RESET SEEDS ====================

    def reset_seeds(self):
        """Vide le dictionnaire des seeds et efface les points sur l'image."""
        self.seeds = {}
        # Réafficher seulement l'image sans les seeds
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor="nw", image=self.photo)
        print("Seeds réinitialisés")


    # ======================= CHARGER UNE IMAGE ====================

    def choose_image(self):
        """Ouvre une boîte de dialogue pour choisir un fichier image,
        puis recharge le canvas avec cette nouvelle image."""
        file_path = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            self.image_path = file_path
            # Charger et redimensionner la nouvelle image
            img = Image.open(self.image_path).resize((self.width, self.height), Image.BICUBIC)
            self.display_img = img
            self.photo = ImageTk.PhotoImage(img)
            # Mettre à jour le canvas
            self.image_canvas.delete("all")
            self.image_canvas.create_image(0, 0, anchor="nw", image=self.photo)
            # Réinitialiser les seeds (optionnel)
            self.seeds = {}
            # Effacer la loupe
            self.magnifier_canvas.delete("all")
            print(f"Nouvelle image chargée : {self.image_path}")

    # ======================= CHOIX COULEUR ====================

    def create_color_palette(self):
        self.color_palette = tk.Toplevel(self.root)
        self.color_palette.title("Palette de couleurs")
        self.colors = {
            "#ff0000": "Rouge",
            "#00ff00": "Vert",
            "#000000": "Noir",
            "#ffff00": "Jaune",
            "#ff00ff": "Magenta",
            "#00ffff": "Cyan",
            "#ffffff": "Blanc",
            "#0000ff": "Bleu",
            "#ffa500": "Orange",
        }
        row = 0
        col = 0
        for color_code, color_name in self.colors.items():
            btn = tk.Button(self.color_palette, bg=color_code, width=5, height=1,
                            command=lambda c=color_code: self.select_color(c))
            btn.grid(row=row, column=col, padx=5, pady=5)
            col += 1
            if col == 3:
                row += 1
                col = 0

    def select_color(self, color_code):
        self.selected_color = self.colors[color_code]
        self.selected_color_code = color_code
        self.color_palette.destroy()
        self.color_label['text'] = f"Couleur sélectionnée : {self.selected_color}"
        self.color_label['fg'] = color_code

    # ======================= SAUVEGARDER UN PRESET ====================

    def show_save_preset_window(self):
        """Ouvre une petite fenêtre pour demander le nom du preset."""
        self.save_win = tk.Toplevel(self.root)
        self.save_win.title("Sauvegarder un preset")
        tk.Label(self.save_win, text="Nom du preset :").pack(padx=5, pady=5)

        self.save_preset_entry = tk.Entry(self.save_win)
        self.save_preset_entry.pack(padx=5, pady=5)

        save_btn = tk.Button(self.save_win, text="Enregistrer", command=self.save_current_seeds_as_preset)
        save_btn.pack(pady=5)

    def save_current_seeds_as_preset(self):
        """Récupère le nom du preset, l’enregistre dans self.all_presets, puis réécrit presets.json."""
        preset_name = self.save_preset_entry.get().strip()
        if not preset_name:
            return

        # Convertir (y, x) en str pour JSON
        seeds_str_keys = {str(k): v for k, v in self.seeds.items()}

        # Stocker dans le gros dictionnaire 
        self.all_presets[preset_name] = seeds_str_keys

        # Sauver le gros dictionnaire en JSON
        self.save_all_presets()

        self.save_win.destroy()
        print(f"Preset '{preset_name}' sauvegardé avec {len(self.seeds)} points.")

    # ======================= CHARGER UN PRESET ====================

    def show_load_preset_window(self):
        """Ouvre une fenêtre pour lister tous les noms de preset et en charger un."""
        self.load_win = tk.Toplevel(self.root)
        self.load_win.title("Charger un preset")

        tk.Label(self.load_win, text="Choisissez un preset :").pack(padx=5, pady=5)

        # La liste des clés dans all_presets
        all_keys = list(self.all_presets.keys())

        self.preset_var = tk.StringVar()
        self.preset_box = ttk.Combobox(self.load_win, textvariable=self.preset_var,
                                       values=all_keys, state='readonly')
        self.preset_box.pack(padx=5, pady=5)

        load_btn = tk.Button(self.load_win, text="Charger", command=self.load_selected_preset)
        load_btn.pack(pady=5)

    def load_selected_preset(self):
        """Charge le preset choisi dans la combobox."""
        preset_name = self.preset_var.get()
        if not preset_name:
            return

        self.load_win.destroy()

        # Récupère le dictionnaire str->couleur
        seeds_str_keys = self.all_presets[preset_name]

        # Convertit à nouveau en (y,x)->couleur
        new_seeds = {}
        for k, color in seeds_str_keys.items():
            # k est une chaîne comme '(10, 20)', on le parse avec eval
            coord = ast.literal_eval(k)
            new_seeds[coord] = color

        self.seeds = new_seeds
        print(f"Preset '{preset_name}' chargé ({len(self.seeds)} points).")
        self.redraw_seeds()

    # ======================= REDESSINER L'IMAGE & POINTS ====================

    def redraw_seeds(self):
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor="nw", image=self.photo)
        for (y, x), color in self.seeds.items():
            self.image_canvas.create_oval(
                x - 2, y - 2, x + 2, y + 2,
                fill=color, outline=color
            )

    # ======================= GESTION DU CLIC ====================

    def on_click(self, event):
        x, y = event.x, event.y
        if x < 3 or y < 3 or x >= self.width - 3 or y >= self.height - 3:
            return

        print(f"Pixel coordinates: ({y}, {x})")
        if self.selected_color is not None:
            # On dessine un point
            self.image_canvas.create_oval(x-3, y-3, x+3, y+3, fill=self.selected_color_code)
            # On marque le voisinage 7x7 dans self.seeds
            for i in range(-3, 4):
                for j in range(-3, 4):
                    self.seeds[(y+i, x+j)] = self.selected_color_code

    # ======================= LOUPE ====================

    def show_magnifier(self, event):
        x, y = event.x, event.y
        # Si on sort de l'image
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            self.magnifier_canvas.delete("all")
            return

        half = self.magnifier_size // 2
        left = max(0, x - half)
        top = max(0, y - half)
        right = min(self.width, x + half)
        bottom = min(self.height, y + half)

        # Extraire et zoomer
        region = self.display_img.crop((left, top, right, bottom))
        zoomed_region = region.resize(
            ((right - left)*self.magnifier_scale,
             (bottom - top)*self.magnifier_scale),
            Image.NEAREST
        )
        self.magnifier_photo = ImageTk.PhotoImage(zoomed_region)

        self.magnifier_canvas.delete("all")
        self.magnifier_canvas.create_image(0, 0, anchor="nw", image=self.magnifier_photo)

        # Crosshair
        zw, zh = zoomed_region.size
        cx, cy = zw//2, zh//2
        self.magnifier_canvas.create_line(cx, 0, cx, zh, fill="red", width=1)
        self.magnifier_canvas.create_line(0, cy, zw, cy, fill="red", width=1)

    # ======================= SEGMENTATION ====================

    def segmentImage(self):
        image = RandomWalkerSegmentation(self.seeds, self.image_path, self.height, self.width)
        print("Segmenting image...")

        # Chronométrage
        start = time.perf_counter()
        image.color_all_pixels()
        end = time.perf_counter()

        elapsed = end - start
        print(f"Image segmented in {elapsed:.4f} seconds")

        output_path = os.path.join(OUTPUT_DIR, "segmented_image.png")
        image.save_image(output_path, show=True)
        print(f"Image saved as {output_path}")

        # Enregistrement dans un fichier log
        stats_marche = image.get_stats_nb_marches()
        log_line = f"{datetime.now().isoformat()} | Image: {self.image_path} | Max marches : {stats_marche[0]} Nb convergentes : {stats_marche[1]} | Time: {elapsed:.4f} seconds\n"
        with open(LOG_FILE, "a") as log_file:
            log_file.write(log_line)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Sélecteur de couleur et affichage d'image")
    ui = UI(root)
    ui.run_ui()
    root.mainloop()

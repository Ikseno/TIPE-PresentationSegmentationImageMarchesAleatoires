# Journal de bord du TIPE

Notes brutes prises pendant le projet (fusion des fichiers `plan/orga.md` des deux anciennes branches).

## Phase 1 : algorithme naïf (branche `tipe-naive-fonctionnalites`)

tous les problèmes :

les probabilités sont très mal faites en effet a partir d'une distance de 100 le calcul de l'exponentielle est assimilié à une valeur nulle ce qui casse le programme quand un pixel isolé au milieu de couleurs différentes.

le choix des pixels n'est pas aléatoire donc risque de fausser les résultats.

beaucoup trop d'étapes avant de trouver un pixel seed.

la détection de pixel seed n'est pas bien, il faut caper le nombre de pas assigner la couleur puis passer au suivant, avec une barrière d'absorption ça fera bien le boulot.

faire une image plus petite pour les tests du genre 200*150

fait !

08/03 : 
ajout d'une limite dans main.py pour ne pas prendre des pixels seeds qui sont en dehors de 200*150
paramètre actuel : sigma = 40 et marches max = 500
avec un bon seeding : on a 27885 marches convergentes sur 30000, je vais essayer d'augmenter un peu l'accuracy

sigma le plus petit qui marche : 8 permet d'avoir la gaussienne la plus discriminante, au dela python assimile les probabilités à 0 ce qui casse le programme

en augmentant le nombre de pas on se rend compte que ça ne change plus trop les résultats a partir d'un certain nombre.
les marches devant êtres coincées augmente le temps de calcul pour pas beauoup plus de résultats à 28000 les résultats sont plutôt satisfaisants avec un temps de calcul raisonnable.

## Phase 2 : apprentissage automatique (branche `tipe-ai-segmentation`)


on va utiliser une régression car les données sont continues 

méthodes éventuelles :
- k plus proches voisins
- régression linéaire

but segmenter le pokemon en dehors d'un décors
donc faut l'entrainer à appuyer sur le pokemon et dehors 

=+=+=+=+
=+=+=+=+
=+=+=+=+

utilisation d'un dataset type COCO et l'entrainer sur des chats par exemple

on va utiliser deux approches que l'on comparera en terme d'efficacité

la première est d'utiliser un clustering k mean avant d'utiliser un modèle CNN
la deuxième est d'utiliser un CNN directement sur l'image pour placer correctement les seeds dans les zones foreground et background de façon uniforme 

renseignons nous sur l'utilisation de COCO et comment faire un réseau de neurone en python

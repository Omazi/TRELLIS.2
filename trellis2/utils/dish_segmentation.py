"""Détourage des photos de plats pour TRELLIS 2 (menu AR).

Trois modes :
- cut_auto : (par défaut) garde automatiquement la nourriture, les sauces, l'assiette ou
             la boîte, et retire le fond autour. Le contenant sert de socle au modèle 3D,
             comme le fait Tripo. Aucun clic.
- cut_dish : garde uniquement la nourriture (détourage automatique BiRefNet via rembg).
- cut_all  : garde tout et retire seulement les zones désignées par des clics (SAM 2).
Les deux renvoient une image RGBA prête pour Trellis2ImageTo3DPipeline.run().
"""
from typing import List, Optional, Sequence, Tuple
import numpy as np
from PIL import Image, ImageFilter


def _weights(name: str) -> str:
    """Poids locaux (sauvegarde Drive) si MENU_AR_WEIGHTS les contient, sinon téléchargement normal."""
    import os
    d = os.environ.get('MENU_AR_WEIGHTS')
    return os.path.join(d, name) if d and os.path.exists(os.path.join(d, name)) else name


DISH_ITEMS = ['plate', 'bowl', 'sauce cup', 'cup', 'glass', 'takeout box',
              'cutting board', 'basket', 'food']
# Pas de « tray » : le plateau (ou la table) est le support, pas le plat. Un objet détecté qui
# touche deux bords de la photo ou en couvre plus de 45 % est traité comme support et ignoré.


def _feathered_rgba(rgb: np.ndarray, mask: np.ndarray, feather: float = 1.2) -> Image.Image:
    a = np.array(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(feather)))
    a[[0, -1], :] = 0; a[:, [0, -1]] = 0   # bord transparent : TRELLIS utilise alors notre détourage
    return Image.fromarray(np.dstack([rgb, a]))


def _trim_background(rgb: np.ndarray, hull: np.ndarray, keep: np.ndarray, tol: float = 0.12,
                     max_cut: float = 0.06) -> np.ndarray:
    """Retire de l'enveloppe convexe les poches de fond (plateau, table) qu'elle englobe entre
    deux débords (salade qui dépasse du bol, etc.) : pixels hors détourage, reliés au bord de
    l'enveloppe et de la couleur du fond juste autour. Le bord du bol rogné par le détourage
    (blanc, pas couleur fond) est gardé."""
    from scipy import ndimage
    on_border = hull[0].sum() + hull[-1].sum() + hull[:, 0].sum() + hull[:, -1].sum()
    perimeter = (hull & ~ndimage.binary_erosion(hull)).sum()
    if on_border > 0.05 * perimeter:
        return hull                       # plat coupé par le cadre : le « fond » autour serait le contenant
    lab = rgb.astype(np.float32) / 255
    ring = ndimage.binary_dilation(hull, iterations=12) & ~ndimage.binary_dilation(hull, iterations=3)
    if ring.sum() < 50:
        return hull
    bg = np.median(lab[ring], axis=0)
    spread = np.median(np.abs(lab[ring] - bg).sum(1)) + tol      # fond texturé -> tolérance plus large
    like_bg = np.abs(lab - bg).sum(2) < spread
    cand = hull & ~keep & like_bg
    edge = hull & ~ndimage.binary_erosion(hull, iterations=2)
    lab_c, n = ndimage.label(cand)
    if not n:
        return hull
    touching = np.unique(lab_c[edge & cand]); touching = touching[touching > 0]
    out = hull & ~np.isin(lab_c, touching)
    out = ndimage.binary_fill_holes(ndimage.binary_opening(out, iterations=2))
    return out if (hull & ~out).sum() <= max_cut * hull.sum() else hull   # au-delà : trop risqué


class AutoDishSegmenter:
    """Nourriture (rembg) + objets du plat détectés par texte (YOLOE) -> enveloppe convexe = plat + contenant."""

    def __init__(self, rembg_model: str = 'birefnet-general', detector_weights: str = 'yoloe-11l-seg.pt',
                 items: Sequence[str] = DISH_ITEMS, conf: float = 0.25):
        import rembg
        from ultralytics import YOLOE
        try:
            self.session = rembg.new_session(rembg_model)
        except Exception:                       # téléchargement coupé : modèle plus léger
            self.session = rembg.new_session('isnet-general-use')
        self.items = list(items); self.conf = conf; self.last_found = []
        self.detector = YOLOE(_weights(detector_weights))
        self.detector.set_classes(self.items, self.detector.get_text_pe(self.items))

    def masks(self, image: Image.Image):
        import rembg
        from PIL import ImageDraw
        from scipy import ndimage
        from skimage.morphology import convex_hull_image
        img = image.convert('RGB'); A = np.array(img); H, W = A.shape[:2]
        food = np.array(rembg.remove(img, session=self.session))[..., 3] > 127
        r = self.detector.predict(A, conf=self.conf, verbose=False)[0]
        items = np.zeros((H, W), bool); found = []
        if r.masks is not None:
            for c, poly in zip(r.boxes.cls.tolist(), r.masks.xy):
                if len(poly) < 3:
                    continue
                m = Image.new('L', (W, H), 0); ImageDraw.Draw(m).polygon([tuple(p) for p in poly], fill=255)
                m = np.array(m) > 0
                sides = int(m[0].any()) + int(m[-1].any()) + int(m[:, 0].any()) + int(m[:, -1].any())
                if sides >= 2 or m.mean() > 0.45:          # plateau, table, planche géante : support
                    continue
                items |= ndimage.binary_erosion(m, iterations=3)   # masques YOLOE un peu gras
                found.append(self.items[int(c)])
        elements = food | items
        lab, n = ndimage.label(elements)        # ignore les miettes isolées (< 0,5 % de l'image)
        if n:
            sizes = ndimage.sum(elements, lab, range(1, n + 1))
            elements = np.isin(lab, 1 + np.flatnonzero(sizes >= 0.005 * H * W))
        if not elements.any():
            return food, food, found
        # enveloppe convexe : referme le bord du bol/de la boîte que le détourage aurait rogné,
        # sans marge autour (une marge ajoutait un liseré de plateau, devenu un socle noir en 3D)
        hull = ndimage.binary_dilation(convex_hull_image(elements), iterations=1)
        hull = _trim_background(A, hull, ndimage.binary_dilation(elements, iterations=2))
        return food, hull, found

    def __call__(self, image: Image.Image, keep_container: bool = True) -> Image.Image:
        food, hull, self.last_found = self.masks(image)
        return _feathered_rgba(np.array(image.convert('RGB')), hull if keep_container else food)


def cut_auto(image: Image.Image, keep_container: bool = True, segmenter: Optional[AutoDishSegmenter] = None) -> Image.Image:
    return (segmenter or AutoDishSegmenter())(image, keep_container)


def cut_dish(image: Image.Image, session=None) -> Image.Image:
    import rembg
    if session is None:
        session = rembg.new_session('birefnet-general')
    return rembg.remove(image.convert('RGB'), session=session)


class ClickSegmenter:
    """Part de toute l'image ; chaque clic retire (ou rajoute) l'objet désigné, segmenté par SAM 2."""

    def __init__(self, image: Image.Image, sam_weights: str = 'sam2.1_b.pt', sam=None):
        self.image = image.convert('RGB')
        self.array = np.array(self.image)
        h, w = self.array.shape[:2]
        self.mask = np.ones((h, w), bool)
        self.history: List[Tuple[int, int, bool]] = []
        if sam is None:
            from ultralytics import SAM
            sam = SAM(_weights(sam_weights))
        self.sam = sam

    def object_at(self, x: int, y: int) -> np.ndarray:
        r = self.sam(self.array, points=[[int(x), int(y)]], labels=[1], verbose=False)[0]
        return r.masks.data.cpu().numpy()[0] > 0.5

    def click(self, x: int, y: int, remove: bool = True) -> float:
        """Retire (remove=True) ou rajoute l'objet sous (x, y). Renvoie la part de l'image gardée."""
        m = self.object_at(x, y)
        self.mask = (self.mask & ~m) if remove else (self.mask | m)
        self.history.append((int(x), int(y), remove))
        return float(self.mask.mean())

    def undo(self) -> float:
        clicks = self.history[:-1]
        self.mask[:] = True; self.history = []
        for x, y, rm in clicks:
            self.click(x, y, rm)
        return float(self.mask.mean())

    def rgba(self, feather: float = 1.2) -> Image.Image:
        return _feathered_rgba(self.array, self.mask, feather)

    def preview(self, max_size: int = 700) -> Image.Image:
        """Aperçu : zones retirées en damier vert."""
        img = self.image.copy(); img.thumbnail((max_size, max_size))
        m = Image.fromarray((self.mask * 255).astype(np.uint8)).resize(img.size)
        bg = Image.new('RGB', img.size, (70, 170, 70))
        return Image.composite(img, bg, m)


def cut_all(image: Image.Image, remove_points: Sequence[Tuple[int, int]] = (), sam=None) -> Image.Image:
    seg = ClickSegmenter(image, sam=sam)
    for x, y in remove_points:
        seg.click(x, y, remove=True)
    return seg.rgba()

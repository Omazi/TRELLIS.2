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


DISH_ITEMS = ['plate', 'bowl', 'sauce cup', 'cup', 'glass', 'tray', 'takeout box',
              'cutting board', 'basket', 'food']


def _feathered_rgba(rgb: np.ndarray, mask: np.ndarray, feather: float = 1.2) -> Image.Image:
    a = np.array(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(feather)))
    a[[0, -1], :] = 0; a[:, [0, -1]] = 0   # bord transparent : TRELLIS utilise alors notre détourage
    return Image.fromarray(np.dstack([rgb, a]))


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
        self.items = list(items); self.conf = conf
        self.detector = YOLOE(detector_weights)
        self.detector.set_classes(self.items, self.detector.get_text_pe(self.items))

    def masks(self, image: Image.Image):
        import rembg
        from PIL import ImageDraw
        from scipy import ndimage
        from skimage.morphology import convex_hull_image
        img = image.convert('RGB'); A = np.array(img); H, W = A.shape[:2]
        food = np.array(rembg.remove(img, session=self.session))[..., 3] > 127
        r = self.detector.predict(A, conf=self.conf, verbose=False)[0]
        items = Image.new('L', (W, H), 0); d = ImageDraw.Draw(items); found = []
        if r.masks is not None:
            for c, poly in zip(r.boxes.cls.tolist(), r.masks.xy):
                if len(poly) >= 3:
                    d.polygon([tuple(p) for p in poly], fill=255); found.append(self.items[int(c)])
        elements = food | (np.array(items) > 0)
        lab, n = ndimage.label(elements)        # ignore les miettes isolées (< 0,5 % de l'image)
        if n:
            sizes = ndimage.sum(elements, lab, range(1, n + 1))
            elements = np.isin(lab, 1 + np.flatnonzero(sizes >= 0.005 * H * W))
        if not elements.any():
            return food, food, found
        hull = convex_hull_image(elements)
        hull = ndimage.binary_dilation(hull, iterations=max(2, int(0.01 * max(H, W))))
        return food, hull, found

    def __call__(self, image: Image.Image, keep_container: bool = True) -> Image.Image:
        food, hull, _ = self.masks(image)
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
            sam = SAM(sam_weights)
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

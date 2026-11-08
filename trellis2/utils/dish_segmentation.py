"""Détourage des photos de plats pour TRELLIS 2 (menu AR).

Deux modes :
- cut_dish : garde uniquement la nourriture (détourage automatique BiRefNet via rembg).
- cut_all  : garde tout (contenant, papier, sauces...) et retire seulement les zones
             que l'utilisateur désigne par des clics (segmentation SAM 2).
             Le contenant sert alors de socle au modèle 3D, comme le fait Tripo.
Les deux renvoient une image RGBA prête pour Trellis2ImageTo3DPipeline.run().
"""
from typing import List, Optional, Sequence, Tuple
import numpy as np
from PIL import Image, ImageFilter


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
        a = Image.fromarray((self.mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(feather))
        a = np.array(a)
        # bord transparent : TRELLIS doit voir de la transparence pour utiliser notre détourage
        a[[0, -1], :] = 0; a[:, [0, -1]] = 0
        return Image.fromarray(np.dstack([self.array, a]))

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

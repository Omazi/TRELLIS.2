"""Détourage des photos de plats pour TRELLIS 2 (menu AR).

Trois modes :
- cut_auto : (par défaut) garde automatiquement la nourriture, les sauces, l'assiette ou
             la boîte, et retire le fond autour. Le contenant sert de socle au modèle 3D,
             comme le fait Tripo. Aucun clic.
- cut_dish : garde uniquement la nourriture (détourage automatique BiRefNet via rembg).
- cut_all  : garde tout et retire seulement les zones désignées par des clics (SAM 2).
Les deux renvoient une image RGBA prête pour Trellis2ImageTo3DPipeline.run().
"""
import os
from typing import List, Optional, Sequence, Tuple
import numpy as np
from PIL import Image, ImageFilter


def _weights(name: str) -> str:
    """Poids locaux (sauvegarde Drive) si MENU_AR_WEIGHTS les contient, sinon téléchargement normal."""
    import os
    d = os.environ.get('MENU_AR_WEIGHTS')
    return os.path.join(d, name) if d and os.path.exists(os.path.join(d, name)) else name


DISH_ITEMS = ['plate', 'bowl', 'sauce cup', 'cup', 'glass', 'takeout box',
              'cutting board', 'basket', 'food', 'sauce packet', 'condiment sachet']
CONTAINERS = {'plate', 'bowl', 'takeout box', 'cutting board', 'basket'}
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

    def detections(self, image: Image.Image):
        """[(nom, masque booléen)] des objets du plat repérés sur la photo."""
        from PIL import ImageDraw
        from scipy import ndimage
        A = np.array(image.convert('RGB')); H, W = A.shape[:2]
        r = self.detector.predict(A, conf=self.conf, verbose=False)[0]
        out = []
        if r.masks is not None:
            for c, poly in zip(r.boxes.cls.tolist(), r.masks.xy):
                if len(poly) < 3:
                    continue
                m = Image.new('L', (W, H), 0); ImageDraw.Draw(m).polygon([tuple(p) for p in poly], fill=255)
                out.append((self.items[int(c)], ndimage.binary_erosion(np.array(m) > 0, iterations=3)))  # un peu gras
        return out

    def masks(self, image: Image.Image):
        import rembg
        from PIL import ImageDraw
        from scipy import ndimage
        from skimage.morphology import convex_hull_image
        img = image.convert('RGB'); A = np.array(img); H, W = A.shape[:2]
        food = np.array(rembg.remove(img, session=self.session))[..., 3] > 127
        detections = self.detections(img)
        # Chaque objet (bol, boîte, sauce, sachet...) et chaque morceau de nourriture est refermé par
        # sa propre enveloppe convexe : le bord rogné par le détourage est récupéré, mais la table
        # entre deux objets (ou entre le plat et un sachet posé à côté) n'est pas englobée.
        parts = []; found = []; extra = []
        near_food = ndimage.binary_dilation(food, iterations=max(3, int(0.02 * max(H, W))))
        for name, m in detections:
            sides = int(m[0].any()) + int(m[-1].any()) + int(m[:, 0].any()) + int(m[:, -1].any())
            if sides >= 3 or m.mean() > 0.45:          # plateau, table, set de table : le support
                continue
            if m.sum() < 0.001 * H * W:
                continue
            solid = m.sum() / max(1, convex_hull_image(m).sum())
            if name in CONTAINERS and m.mean() > 0.03 and solid < 0.85:
                # grand contenant au masque en lambeaux (une assiette nette et pleine est gardée telle
                # quelle) : enveloppe convexe seulement autour de ce que le détourage a vu
                # (le masque YOLOE d'une assiette déborde en fines bavures sur le set de table) ;
                # les parties épaisses du masque loin de la nourriture (parois d'une boîte) restent.
                thick = ndimage.binary_opening(m, iterations=max(2, int(0.01 * max(H, W))))
                extra.append(thick & ~near_food)
                m &= near_food
            parts.append(m); found.append(name)
        n_items = len(parts)
        lab, n = ndimage.label(food)            # morceaux de nourriture (miettes < 0,5 % ignorées)
        for k in range(1, n + 1):
            m = lab == k
            if m.sum() >= 0.005 * H * W:
                parts.append(m)
        if not parts:
            return food, food, found
        elements = np.logical_or.reduce(parts)
        # objet isolé, petit et loin du plat (verre coupé en haut de la photo) : ignoré
        lab, n = ndimage.label(elements)
        sizes = ndimage.sum(elements, lab, range(1, n + 1))
        biggest = lab == 1 + int(np.argmax(sizes))
        # un petit morceau détaché (raisin, miette du décor) n'est gardé que s'il touche presque le plat ;
        # un objet détecté (sauce, sachet) peut être un peu plus loin
        touching = ndimage.binary_dilation(biggest, iterations=max(2, int(0.008 * max(H, W))))
        close_by = ndimage.binary_dilation(biggest, iterations=max(3, int(0.05 * max(H, W))))
        items_mask = np.logical_or.reduce(parts[:n_items]) if n_items else np.zeros((H, W), bool)
        def keep(k):
            comp = lab == k + 1
            if sizes[k] >= 0.02 * H * W or (touching & comp).any():
                return True
            return bool((close_by & comp).any() and (comp & items_mask).sum() > 0.5 * comp.sum())
        keep_ids = [k + 1 for k in range(n) if keep(k)]
        main = np.isin(lab, keep_ids)
        hull = np.zeros((H, W), bool)
        r_close = max(2, int(0.012 * max(H, W)))
        for m in parts[:n_items]:                  # objets détectés : enveloppe convexe (bol, boîte, sachet)
            if (m & main).any():
                hull |= convex_hull_image(m)
        for m in parts[n_items:]:                  # nourriture : on referme les trous du détourage sans
            if (m & main).any():                   # englober la table dans les creux (sachet qui dépasse)
                big = np.pad(m, r_close)
                hull |= ndimage.binary_closing(big, iterations=r_close)[r_close:-r_close, r_close:-r_close]
        for m in extra:
            hull |= m
        hull = ndimage.binary_fill_holes(ndimage.binary_dilation(hull, iterations=1))
        hull = _trim_background(A, hull, ndimage.binary_dilation(elements, iterations=2))
        return food, hull, found

    def __call__(self, image: Image.Image, keep_container: bool = True) -> Image.Image:
        food, hull, self.last_found = self.masks(image)
        return _feathered_rgba(np.array(image.convert('RGB')), hull if keep_container else food)


def cut_auto(image: Image.Image, keep_container: bool = True, segmenter: Optional[AutoDishSegmenter] = None) -> Image.Image:
    return (segmenter or AutoDishSegmenter())(image, keep_container)


class Sam2Masks:
    """SAM 2.1 (Meta, licence Apache 2.0) par la bibliothèque transformers : contour exact d'un objet
    désigné par un cadre ou par un point. Remplace la version Ultralytics (AGPL)."""

    def __init__(self, model_id: str = 'facebook/sam2.1-hiera-large', device: Optional[str] = None):
        import torch
        from transformers import Sam2Model, Sam2Processor
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(self.device).eval()

    def __call__(self, image: Image.Image, boxes=None, points=None) -> np.ndarray:
        """boxes : [[x0, y0, x1, y1], ...] ou points : [[x, y], ...] -> masques [n, H, W] booléens."""
        import torch
        kw = {}
        if boxes is not None:
            kw['input_boxes'] = [[[float(v) for v in b] for b in boxes]]
        else:
            kw['input_points'] = [[[[float(x), float(y)]] for x, y in points]]
            kw['input_labels'] = [[[1] for _ in points]]
        inputs = self.processor(images=image.convert('RGB'), return_tensors='pt', **kw).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), inputs['original_sizes'].cpu())[0]
        return masks[:, 0].numpy() > 0.5


class OpenDishSegmenter(AutoDishSegmenter):
    """Même détourage que AutoDishSegmenter, avec uniquement des modèles utilisables à la vente :
    BiRefNet (MIT) pour la nourriture, Grounding DINO (IDEA, licence Apache 2.0) pour repérer par
    texte assiette, bol, boîte, sauces, sachets, et SAM 2.1 (Meta, Apache 2.0) pour leur contour."""

    def __init__(self, rembg_model: str = 'birefnet-general', items: Sequence[str] = DISH_ITEMS,
                 detector_id: str = 'IDEA-Research/grounding-dino-base', box_threshold: float = 0.3,
                 text_threshold: float = 0.25, sam: Optional['Sam2Masks'] = None, device: Optional[str] = None):
        import rembg
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        try:
            self.session = rembg.new_session(rembg_model)
        except Exception:
            self.session = rembg.new_session('isnet-general-use')
        self.items = list(items); self.last_found = []
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.det_processor = AutoProcessor.from_pretrained(detector_id)
        self.det_model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(self.device).eval()
        self.sam = sam or Sam2Masks(device=self.device)

    def _label(self, text: str) -> Optional[str]:
        """Grounding DINO rend le morceau de phrase reconnu : on le ramène à nos noms d'objets."""
        t = ' '.join(str(text).lower().replace('.', ' ').split())
        if t in self.items:
            return t
        best = [it for it in self.items if it in t or t in it]
        return max(best, key=len) if best else None

    def detections(self, image: Image.Image):
        import torch
        img = image.convert('RGB')
        prompt = ' '.join(f'{it}.' for it in self.items)          # « plate. bowl. sauce cup. ... »
        inputs = self.det_processor(images=img, text=prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            out = self.det_model(**inputs)
        res = self.det_processor.post_process_grounded_object_detection(
            out, inputs['input_ids'], threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[img.size[::-1]])[0]
        labels = res.get('text_labels', res.get('labels'))
        boxes, names = [], []
        for box, lab in zip(res['boxes'].tolist(), labels):
            name = self._label(lab)
            if name is not None:
                boxes.append(box); names.append(name)
        if not boxes:
            return []
        masks = self.sam(img, boxes=boxes)
        return list(zip(names, masks))


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
            if os.environ.get('MENU_AR_LIBRE') == '1':          # version vente : SAM 2.1 par transformers
                sam = Sam2Masks()
            else:
                from ultralytics import SAM
                sam = SAM(_weights(sam_weights))
        self.sam = sam

    def object_at(self, x: int, y: int) -> np.ndarray:
        if isinstance(self.sam, Sam2Masks):
            return self.sam(self.image, points=[[int(x), int(y)]])[0]
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

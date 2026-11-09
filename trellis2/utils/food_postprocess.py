"""Post-traitement pour plats (menu AR).

solidify_base : la génération à partir d'une seule photo produit une peau dont le
dessous remonte en voûte (le modèle ne voit pas le dessous du plat). On rabat cette
face inférieure à plat sur le sol pour que le plat repose sur la table en AR.
"""
import numpy as np

def solidify_base(vertices, faces, up_axis=1, start=-0.15, full=-0.55, smooth_iters=2):
    """Rabat la face inférieure d'un maillage (peau fermée en voûte) à plat sur le sol.

    Les sommets dont la normale pointe vers le bas descendent vers le plan du sol,
    proportionnellement à quel point ils regardent vers le bas : n_y >= start -> pas bougé,
    n_y <= full -> posé au sol. Les coutures UV (sommets dupliqués) bougent ensemble.
    Retourne une copie des sommets.
    """
    V = np.asarray(vertices, dtype=np.float64); F = np.asarray(faces)
    # regrouper les sommets dupliqués (coutures de texture) par position
    key = np.round(V / (np.ptp(V, axis=0).max() * 1e-6)).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
    n_u = inv.max() + 1
    Fu = inv[F]
    Vu = np.zeros((n_u, 3)); np.add.at(Vu, inv, V); Vu /= np.bincount(inv, minlength=n_u)[:, None]
    # normales de sommets (pondérées par l'aire)
    fn = np.cross(Vu[Fu[:, 1]] - Vu[Fu[:, 0]], Vu[Fu[:, 2]] - Vu[Fu[:, 0]])
    vn = np.zeros((n_u, 3)); [np.add.at(vn, Fu[:, i], fn) for i in range(3)]
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    w = np.clip((start - vn[:, up_axis]) / (start - full), 0, 1)
    # lisser le poids sur le voisinage pour éviter les marches d'escalier
    for _ in range(smooth_iters):
        acc = np.zeros(n_u); cnt = np.zeros(n_u)
        for i in range(3):
            for j in range(3):
                np.add.at(acc, Fu[:, i], w[Fu[:, j]]); np.add.at(cnt, Fu[:, i], 1)
        w = np.maximum(w, 0.5 * acc / np.maximum(cnt, 1)) if _ == 0 else acc / np.maximum(cnt, 1)
    floor = Vu[:, up_axis].min()
    Vu[:, up_axis] += w * (floor - Vu[:, up_axis])
    out = V.copy(); out[:, up_axis] = Vu[inv, up_axis]
    return out, w[inv]



def underside_color(tex: np.ndarray, V, F, UV, up_axis=1, band=0.12, shade=0.95):
    """Couleur du dessous = celle du bas des flancs (bord de la planche, du bol, de la boîte),
    un peu assombrie comme à l'ombre. Le dessous n'est jamais vu sur la photo : on prolonge
    simplement le matériau visible juste au-dessus, au lieu d'un noir uniforme."""
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    y = V[:, up_axis]; y0, y1 = y.min(), y.max()
    side = np.abs(fn[:, up_axis]) < 0.35
    low = y[F].mean(1) < y0 + band * (y1 - y0)
    vid = np.unique(F[side & low])
    if len(vid) < 20:
        return None
    T = tex.shape[0]; uv = np.asarray(UV)[vid]
    c = tex[np.clip(((1 - uv[:, 1]) * (T - 1)).astype(int), 0, T - 1), np.clip((uv[:, 0] * (T - 1)).astype(int), 0, T - 1)]
    return tuple(np.median(c, axis=0) * shade)


def paint_underside(texture, V, F, UV, weights=None, color=None, up_axis=1, threshold=-0.7, grow=3):
    """Peint le dessous plat du modèle, pour qu'il ne montre plus la nourriture « en transparence »
    vu d'en bas. color=None : couleur du bas des flancs (bois de la planche, céramique du bol...).
    Retourne une nouvelle image PIL."""
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    from .photo_projection import _texel_positions, vertex_normals
    V = np.asarray(V, np.float64); F = np.asarray(F)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    down = (fn[:, up_axis] < threshold).astype(np.float32)         # faces tournées vers le sol
    w = np.zeros(len(V), np.float32); np.maximum.at(w, F.ravel(), np.repeat(down, 3))
    tex = np.asarray(texture.convert('RGB'), np.float32); T = tex.shape[0]
    if color is None:
        color = underside_color(tex, V, F, UV, up_axis=up_axis) or (46, 40, 34)
    _, wt, used = _texel_positions(V, F, np.asarray(UV), np.repeat(w[:, None], 3, 1), T)
    idx = distance_transform_edt(~used, return_distances=False, return_indices=True)
    a = wt[idx[0], idx[1], :1]
    if grow:                                                       # déborde un peu : efface les liserés
        from scipy.ndimage import grey_dilation
        a = grey_dilation(a[..., 0], size=(2 * grow + 1, 2 * grow + 1))[..., None]
    out = tex * (1 - a) + np.array(color, np.float32) * a
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def floating_bits(V, F, max_area: float = 0.02, min_gap: float = 0.02) -> np.ndarray:
    """Faces à retirer : petits morceaux détachés et éloignés du plat (un raisin du décor gardé
    par erreur au détourage devient un petit cube qui flotte à côté de l'assiette).
    Un morceau est retiré s'il fait moins de max_area de la surface totale ET s'il est à plus de
    min_gap (fraction de la taille du plat) du morceau principal. Une sauce ou un sachet posé
    contre le plat touche le reste : il est gardé. Renvoie un masque booléen par face."""
    import trimesh
    from scipy.spatial import cKDTree
    m = trimesh.Trimesh(np.asarray(V, np.float64), np.asarray(F), process=False)
    m.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=5)      # coutures UV recousues
    comps = trimesh.graph.connected_components(m.face_adjacency, min_len=1)
    area = m.area_faces; total = area.sum()
    if len(comps) < 2 or total <= 0:
        return np.zeros(len(F), bool)
    comps = sorted(comps, key=lambda c: -area[c].sum())
    main_v = np.unique(m.faces[comps[0]].ravel())
    tree = cKDTree(m.vertices[main_v])
    size = np.ptp(m.vertices, axis=0).max()
    drop = np.zeros(len(F), bool)
    for c in comps[1:]:
        if area[c].sum() / total >= max_area:
            continue
        d, _ = tree.query(m.vertices[np.unique(m.faces[c].ravel())], k=1)
        if d.min() > min_gap * size:
            drop[c] = True
    return drop


def rim_color_from_photo(rgba, band: float = 0.03, shade: float = 0.9):
    """Couleur du contenant vue SUR LA PHOTO : médiane des pixels de la bande extérieure du
    détourage (bord de l'assiette, de la planche, du bol, de la boîte). Sert au dessous du modèle,
    que la photo ne montre jamais : mieux vaut la vraie couleur du bord que celle des flancs
    inventés (striés, souvent sombres)."""
    from scipy import ndimage
    a = np.asarray(rgba.convert('RGBA'))
    m = a[..., 3] > 127
    if m.sum() < 500:
        return None
    k = max(2, int(band * max(m.shape)))
    ring = m & ~ndimage.binary_erosion(m, iterations=k)
    ring &= ndimage.binary_erosion(m, iterations=2)          # sans le liseré adouci du détourage
    if ring.sum() < 200:
        return None
    return tuple(float(c) for c in np.median(a[..., :3][ring].astype(np.float32), axis=0) * shade)

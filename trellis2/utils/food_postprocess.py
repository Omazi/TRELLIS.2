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



def paint_underside(texture, V, F, UV, weights=None, color=(46, 40, 34), up_axis=1, threshold=-0.7, grow=3):
    """Peint d'une couleur unie (carton/assiette sombre) le dessous plat du modèle, pour qu'il
    ne montre plus la nourriture « en transparence » vu d'en bas. Retourne une nouvelle image PIL.
    weights : poids optionnels par sommet (ex. ceux de solidify_base) ; sinon d'après les normales."""
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    from .photo_projection import _texel_positions, vertex_normals
    V = np.asarray(V, np.float64); F = np.asarray(F)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    down = (fn[:, up_axis] < threshold).astype(np.float32)         # faces tournées vers le sol
    w = np.zeros(len(V), np.float32); np.maximum.at(w, F.ravel(), np.repeat(down, 3))
    tex = np.asarray(texture.convert('RGB'), np.float32); T = tex.shape[0]
    _, wt, used = _texel_positions(V, F, np.asarray(UV), np.repeat(w[:, None], 3, 1), T)
    idx = distance_transform_edt(~used, return_distances=False, return_indices=True)
    a = wt[idx[0], idx[1], :1]
    if grow:                                                       # déborde un peu : efface les liserés
        from scipy.ndimage import grey_dilation
        a = grey_dilation(a[..., 0], size=(2 * grow + 1, 2 * grow + 1))[..., None]
    out = tex * (1 - a) + np.array(color, np.float32) * a
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))

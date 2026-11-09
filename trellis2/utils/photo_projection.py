"""Recalage de la texture générée sur la photo d'origine (menu AR).

TRELLIS 2 réinvente toute la texture : les couleurs dérivent (ex. pâte dorée rendue rose).
1. estimate_camera : retrouve l'angle de prise de vue (silhouette du modèle vs détourage
   de la photo, puis accord des couleurs pour lever l'ambiguïté des contenants symétriques).
2. recolor_from_photo : garde les détails de la texture générée (alignés sur la géométrie)
   mais recale la couleur de chaque zone sur la photo floutée ; les zones cachées héritent
   de la correction des zones voisines. Tolérant aux petits écarts de forme.
3. paste_photo : la vraie photo sur les parties qu'elle montre (dessus du plat), recalée sur les
   contours du modèle par flot optique ; la texture (IA ou recalée) garde les flancs et le reste.
project_photo (ancien plaquage direct, sans recalage) est conservé pour comparaison.
Tout tourne sur CPU (numpy/scipy).
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import minimum_filter


# ---------------------------------------------------------------- caméra
def _rotation(az, el, roll):
    """Caméra orbitale : az autour de Y, el au-dessus de l'horizon, roll dans l'image (degrés)."""
    a, e, r = np.radians([az, el, roll])
    # direction de vue (de la caméra vers l'objet) puis base caméra
    d = -np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(d, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([np.cos(a), 0.0, -np.sin(a)])
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, d)
    cr, sr = np.cos(r), np.sin(r)
    right, cam_up = cr * right + sr * cam_up, -sr * right + cr * cam_up
    return np.stack([right, cam_up, d])          # lignes : x écran, y écran (haut), profondeur


def _project(P, R):
    q = P @ R.T
    return q[:, 0], -q[:, 1], q[:, 2]            # x, y (vers le bas), profondeur


def _silhouette(V, F, R, size, fit=None):
    x, y, _ = _project(V, R)
    if fit is None:
        return x, y
    s, tx, ty = fit
    pts = np.stack([x * s + tx, y * s + ty], 1)
    img = Image.new('L', size, 0); d = ImageDraw.Draw(img)
    for t in pts[F]:
        d.polygon([tuple(p) for p in t], fill=255)
    return np.array(img) > 0


def _fit_to_mask(x, y, mask_stats):
    """Échelle + translation qui alignent l'étendue projetée sur celle du masque (perspective faible)."""
    (mx0, my0, mx1, my1) = mask_stats
    sx = (mx1 - mx0) / max(1e-9, x.max() - x.min()); sy = (my1 - my0) / max(1e-9, y.max() - y.min())
    s = np.sqrt(sx * sy)
    tx = (mx0 + mx1) / 2 - s * (x.max() + x.min()) / 2
    ty = (my0 + my1) / 2 - s * (y.max() + y.min()) / 2
    return s, tx, ty


def _color_agreement(C, vcol, photo, mask, R, fit):
    """Corrélation entre les couleurs du modèle vues sous cet angle et la photo (texels visibles)."""
    H, W = mask.shape
    x, y, z = _project(C, R); s, tx, ty = fit; x = x * s + tx; y = y * s + ty
    ins = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    ix, iy = x[ins].astype(int), y[ins].astype(int); zz = z[ins]; col = vcol[ins]
    zbuf = np.full((H, W), np.inf); np.minimum.at(zbuf, (iy, ix), zz)
    vis = (zz <= zbuf[iy, ix] + 1e-4 * np.ptp(z)) & mask[iy, ix]
    if vis.sum() < 200: return -1.0
    a = col[vis].mean(1); b = photo[iy[vis], ix[vis]].mean(1)
    return float(np.corrcoef(a, b)[0, 1])


def estimate_camera(V, F, mask, max_faces=20000, work=128, verbose=False, vertex_colors=None, photo_rgb=None):
    """Cherche (az, el, roll) : silhouette (IoU) puis, parmi les meilleures, accord des couleurs
    avec la photo (lève l'ambiguïté des contenants symétriques). Renvoie pose + ajustement image."""
    H, W = mask.shape
    k = work / max(H, W); size = (max(8, int(W * k)), max(8, int(H * k)))
    m = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(size)) > 127
    ys, xs = np.nonzero(m); stats = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    rng = np.random.default_rng(0)
    Fs = F if len(F) <= max_faces else F[rng.choice(len(F), max_faces, replace=False)]
    C = V - (V.min(0) + V.max(0)) / 2

    def score(az, el, roll):
        R = _rotation(az, el, roll); x, y = _silhouette(C, Fs, R, size)
        fit = _fit_to_mask(x, y, stats); sil = _silhouette(C, Fs, R, size, fit)
        return (sil & m).sum() / max(1, (sil | m).sum())

    coarse = sorted(((score(a, e, r), a, e, r) for a in range(0, 360, 15) for e in range(15, 91, 15)
                     for r in (-20, -10, 0, 10, 20)), key=lambda t: -t[0])

    def refine(best):
        for step in (5, 2):                      # affinage local
            _, a0, e0, r0 = best
            cand = [(score(a, e, r), a, e, r) for a in (a0 - step, a0, a0 + step)
                    for e in (e0 - step, e0, e0 + step) if 5 <= e <= 90 for r in (r0 - step, r0, r0 + step)]
            best = max(cand + [best], key=lambda t: t[0])
        return best

    if vertex_colors is None or photo_rgb is None:
        iou, az, el, roll = refine(coarse[0])
    else:
        # candidats distincts parmi les meilleures silhouettes, départagés par la couleur
        picks = []
        for c in coarse:
            if c[0] < coarse[0][0] - 0.08: break
            if all(min(abs(c[1] - p[1]) % 360, 360 - abs(c[1] - p[1]) % 360) > 30 or abs(c[3] - p[3]) > 15 for p in picks):
                picks.append(c)
            if len(picks) >= 6: break
        ys, xs = np.nonzero(mask); full_stats = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        scored = []
        for c in picks:
            iou, a, e, r = refine(c); R = _rotation(a, e, r); x, y = _project(C, R)[:2]
            agree = _color_agreement(C, vertex_colors, photo_rgb, mask, R, _fit_to_mask(x, y, full_stats))
            scored.append((iou + agree, iou, agree, a, e, r))
            if verbose: print(f'  candidat az {a}° el {e}° roulis {r}° : silhouette {iou:.2f}, couleurs {agree:.2f}')
        _, iou, _, az, el, roll = max(scored)
    if verbose:
        print(f'angle de la photo : azimut {az}°, élévation {el}°, roulis {roll}° (IoU {iou:.2f})')
    # ajustement en pleine résolution
    R = _rotation(az, el, roll); x, y = _project(C, R)[:2]
    ys, xs = np.nonzero(mask); fit = _fit_to_mask(x, y, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    return dict(az=az, el=el, roll=roll, iou=float(iou), R=R, fit=fit, center=(V.min(0) + V.max(0)) / 2)


# ---------------------------------------------------------------- texture
_BARY_CACHE = {}


def _texel_bary(F, UV, T):
    """Rasterisation des triangles dans l'espace UV (vectorisée, mise en cache) : pour chaque texel,
    le triangle qui le couvre et ses coordonnées barycentriques. Même résultat que l'ancienne boucle
    triangle par triangle, ~100x plus rapide (la texture fait 4096², le maillage ~300 000 faces)."""
    import hashlib
    key = (T, hashlib.sha1(np.ascontiguousarray(F).tobytes() + np.ascontiguousarray(UV, np.float32).tobytes()).hexdigest())
    if key in _BARY_CACHE:
        return _BARY_CACHE[key]
    P = UV * (T - 1); P = np.stack([P[:, 0], (T - 1) - P[:, 1]], 1).astype(np.float64)
    A, B, C = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    lo = np.floor(np.minimum(np.minimum(A, B), C)).astype(np.int64)
    hi = np.ceil(np.maximum(np.maximum(A, B), C)).astype(np.int64)
    lo = np.maximum(lo, 0); hi = np.minimum(hi, T - 1)
    den = (B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1])
    ok = (hi >= lo).all(1) & (np.abs(den) > 1e-12)
    span = (hi - lo + 1).max(1)
    face = np.full((T, T), -1, np.int64); bary = np.zeros((T, T, 3), np.float32)
    done = np.zeros(len(F), bool)
    for k in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
        sel = np.flatnonzero(ok & ~done & (span <= k))
        done[sel] = True
        if not len(sel):
            continue
        g = np.arange(k)
        chunk = max(1, 4_000_000 // (k * k))
        for c0 in range(0, len(sel), chunk):
            f = sel[c0:c0 + chunk]
            shape = (len(f), k, k)
            xs = np.broadcast_to(lo[f, 0][:, None, None] + g[None, None, :], shape)
            ys = np.broadcast_to(lo[f, 1][:, None, None] + g[None, :, None], shape)
            inb = (xs <= hi[f, 0][:, None, None]) & (ys <= hi[f, 1][:, None, None])
            px, py = xs + 0.5, ys + 0.5
            a, b, c, d = A[f][:, None, None], B[f][:, None, None], C[f][:, None, None], den[f][:, None, None]
            w0 = ((b[..., 1] - c[..., 1]) * (px - c[..., 0]) + (c[..., 0] - b[..., 0]) * (py - c[..., 1])) / d
            w1 = ((c[..., 1] - a[..., 1]) * (px - c[..., 0]) + (a[..., 0] - c[..., 0]) * (py - c[..., 1])) / d
            w2 = 1 - w0 - w1
            m = inb & (w0 >= -1e-3) & (w1 >= -1e-3) & (w2 >= -1e-3)
            fi, yi, xi = np.nonzero(m)
            yy, xx = ys[fi, yi, xi], xs[fi, yi, xi]
            face[yy, xx] = f[fi]            # en cas de recouvrement : le dernier triangle gagne (comme avant)
            bary[yy, xx] = np.stack([w0[m], w1[m], w2[m]], 1)
    used = face >= 0
    _BARY_CACHE.clear(); _BARY_CACHE[key] = (face, bary, used)
    return face, bary, used


def _texel_positions(V, F, UV, N, T):
    """Pour chaque texel : position 3D et normale interpolées (rasterisation dans l'espace UV)."""
    face, bary, used = _texel_bary(np.asarray(F), np.asarray(UV), T)
    pos = np.zeros((T, T, 3), np.float32); nrm = np.zeros((T, T, 3), np.float32)
    fu, bu = face[used], bary[used]
    tri = np.asarray(F)[fu]
    pos[used] = np.einsum('nk,nkd->nd', bu, np.asarray(V, np.float32)[tri])
    nrm[used] = np.einsum('nk,nkd->nd', bu, np.asarray(N, np.float32)[tri])
    return pos, nrm, used


def _texel_positions_loop(V, F, UV, N, T):
    """Ancienne version (référence pour les tests)."""
    pos = np.zeros((T, T, 3), np.float32); nrm = np.zeros((T, T, 3), np.float32); used = np.zeros((T, T), bool)
    P = UV * (T - 1); P = np.stack([P[:, 0], (T - 1) - P[:, 1]], 1)
    for f in F:
        a, b, c = P[f]
        x0, y0 = np.floor(np.minimum(np.minimum(a, b), c)).astype(int); x1, y1 = np.ceil(np.maximum(np.maximum(a, b), c)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, T - 1), min(y1, T - 1)
        if x1 < x0 or y1 < y0: continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        px, py = xs.ravel() + 0.5, ys.ravel() + 0.5
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-12: continue
        w0 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / den
        w1 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / den
        w2 = 1 - w0 - w1
        k = (w0 >= -1e-3) & (w1 >= -1e-3) & (w2 >= -1e-3)
        if not k.any(): continue
        W3 = np.stack([w0[k], w1[k], w2[k]], 1)
        yy, xx = ys.ravel()[k], xs.ravel()[k]
        pos[yy, xx] = W3 @ V[f]; nrm[yy, xx] = W3 @ N[f]; used[yy, xx] = True
    return pos, nrm, used


def _bilinear(img, x, y):
    H, W = img.shape[:2]
    x = np.clip(x, 0, W - 1.001); y = np.clip(y, 0, H - 1.001)
    x0, y0 = x.astype(int), y.astype(int); fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def project_photo(texture, V, F, UV, N, photo_rgba, cam, min_facing=0.15, full_facing=0.55, edge_erode=3):
    """Plaque la photo sur les texels visibles ; recale les couleurs du reste par régression."""
    tex = np.asarray(texture.convert('RGB'), np.float32) / 255; T = tex.shape[0]
    ph = np.asarray(photo_rgba.convert('RGBA'), np.float32) / 255; H, W = ph.shape[:2]
    alpha = np.array(Image.fromarray((ph[..., 3] * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(2 * edge_erode + 1)), np.float32) / 255
    pos, nrm, used = _texel_positions(V, F, UV, N, T)
    P = pos[used] - cam['center']; Nn = nrm[used]; Nn /= np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-9
    x, y, z = _project(P, cam['R']); s, tx, ty = cam['fit']; x = x * s + tx; y = y * s + ty
    # visibilité : tampon de profondeur à partir des texels eux-mêmes
    inside = (x >= 0) & (x < W - 1) & (y >= 0) & (y < H - 1)
    ix, iy = np.clip(x.astype(int), 0, W - 1), np.clip(y.astype(int), 0, H - 1)
    zbuf = np.full((H, W), np.inf, np.float32); np.minimum.at(zbuf, (iy[inside], ix[inside]), z[inside])
    zbuf = minimum_filter(zbuf, size=3)               # bouche les trous entre texels
    tol = 0.004 * (V.max(0) - V.min(0)).max()
    visible = inside & (z <= zbuf[iy, ix] + tol)
    facing = -(Nn @ cam['R'][2])                     # 1 = face à la caméra
    w = np.clip((facing - min_facing) / (full_facing - min_facing), 0, 1) * visible * _bilinear(alpha[..., None], x, y)[:, 0]
    photo_col = _bilinear(ph[..., :3], x, y)
    gen_col = tex[used]
    # recalage couleur des zones cachées : régression affine générée -> photo sur les texels bien vus
    good = w > 0.8
    if good.sum() > 500:
        A = np.c_[gen_col[good], np.ones(good.sum())]
        M, *_ = np.linalg.lstsq(A, photo_col[good], rcond=None)
        gen_col = np.clip(np.c_[gen_col, np.ones(len(gen_col))] @ M, 0, 1)
    out_col = w[:, None] * photo_col + (1 - w[:, None]) * gen_col
    new = tex.copy(); new[used] = out_col
    stats = dict(texels=int(used.sum()), from_photo=float((w > 0.5).mean()))
    return Image.fromarray((np.clip(new, 0, 1) * 255).round().astype(np.uint8)), stats


def _render_texels(x, y, z, col, keep, size):
    """Image de la texture vue depuis la caméra de la photo (texels projetés, le plus proche gagne)."""
    W, H = size
    ix, iy = np.clip(x.astype(int), 0, W - 1), np.clip(y.astype(int), 0, H - 1)
    ins = keep & (x >= 0) & (x < W) & (y >= 0) & (y < H)
    zb = np.full((H, W), np.inf, np.float32); np.minimum.at(zb, (iy[ins], ix[ins]), z[ins])
    front = ins & (z <= zb[iy, ix] + 1e-6)
    img = np.zeros((H, W, 3), np.float32); cnt = np.zeros((H, W), np.float32)
    np.add.at(img, (iy[front], ix[front]), col[front]); np.add.at(cnt, (iy[front], ix[front]), 1)
    filled = cnt > 0
    img[filled] /= cnt[filled][:, None]
    return img, filled


def _photo_flow(render, rmask, photo, pmask, max_shift=0.06):
    """Flot dense rendu -> photo : de combien décaler chaque point du rendu du modèle pour tomber sur
    le même endroit de la photo. Calculé sur les contours (distance signée au bord du plat) : la
    texture générée invente souvent d'autres garnitures que la photo, ses détails ne servent donc
    pas de repères ; les bords (croûte, planche, bol, sachet) si. Lissé et borné : un recalage,
    pas une déformation."""
    import cv2
    from scipy.ndimage import distance_transform_edt
    H, W = rmask.shape
    def sdf(m):
        d = distance_transform_edt(m) - distance_transform_edt(~m)
        return np.clip(128 + 4 * d, 0, 255).astype(np.uint8)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(sdf(rmask), sdf(pmask), None)
    sig = 0.02 * max(H, W)
    flow = np.stack([cv2.GaussianBlur(flow[..., k], (0, 0), sig) for k in range(2)], -1)
    mag = np.linalg.norm(flow, axis=2, keepdims=True); lim = max_shift * max(H, W)
    return flow * np.minimum(1, lim / np.maximum(mag, 1e-6))


def paste_photo(texture, V, F, UV, photo_rgba, cam, min_facing=0.55, full_facing=0.75, edge_erode=4,
                align=True, work=768):
    """La photo elle-même sur les parties du plat qu'elle montre ; la texture (IA ou recalée) partout
    ailleurs. Avant de plaquer, la photo est recalée sur le rendu du modèle par flot optique : sans
    ce recalage, une forme générée un peu différente du vrai plat donnait des détails dédoublés.
    Retourne (image PIL, stats)."""
    import cv2
    tex = np.asarray(texture.convert('RGB'), np.float32) / 255; T = tex.shape[0]
    ph = np.asarray(photo_rgba.convert('RGBA'), np.float32) / 255; H, W = ph.shape[:2]
    alpha = cv2.erode(ph[..., 3], np.ones((2 * edge_erode + 1,) * 2, np.uint8))
    V = np.asarray(V, np.float64); F = np.asarray(F)
    N = vertex_normals(V, F)
    pos, nrm, used = _texel_positions(V, F, UV, N, T)
    P = pos[used] - cam['center']; Nn = nrm[used]; Nn /= np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-9
    x, y, z = _project(P, cam['R']); s, tx, ty = cam['fit']; x = x * s + tx; y = y * s + ty
    inside = (x >= 0) & (x < W - 1) & (y >= 0) & (y < H - 1)
    ix, iy = np.clip(x.astype(int), 0, W - 1), np.clip(y.astype(int), 0, H - 1)
    zbuf = np.full((H, W), np.inf, np.float32); np.minimum.at(zbuf, (iy[inside], ix[inside]), z[inside])
    zbuf = minimum_filter(zbuf, size=3)
    tol = 0.004 * np.ptp(V, axis=0).max()
    visible = inside & (z <= zbuf[iy, ix] + tol)
    facing = -(Nn @ cam['R'][2])
    gen = tex[used]
    fx = fy = 0.0; shift = 0.0
    if align:
        k = work / max(H, W); size = (max(16, int(W * k)), max(16, int(H * k)))
        rimg, rmask = _render_texels(x * k, y * k, z, gen, visible & (facing > 0.05), size)
        rimg = cv2.inpaint((rimg * 255).astype(np.uint8), (~rmask & cv2.dilate(rmask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)).astype(np.uint8), 3, cv2.INPAINT_TELEA).astype(np.float32) / 255
        rmask = cv2.dilate(rmask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        small = cv2.resize(ph, size, interpolation=cv2.INTER_AREA)
        flow = _photo_flow(rimg, rmask, small[..., :3], small[..., 3] > 0.5)
        f = _bilinear(flow, x * k, y * k) / k
        fx, fy = f[:, 0], f[:, 1]
        shift = float(np.median(np.hypot(fx, fy)[visible]) / max(H, W)) if visible.any() else 0.0
    photo_col = _bilinear(ph[..., :3], x + fx, y + fy)
    a = _bilinear(alpha[..., None], x + fx, y + fy)[:, 0]
    t = np.clip((facing - min_facing) / (full_facing - min_facing), 0, 1)
    w = (t * t * (3 - 2 * t)) * visible * a
    # pas de recalage global des couleurs ici : la texture reçue est déjà accordée à la photo
    # (recolor_from_photo, ou MV-Adapter guidé par la photo) ; une correction affine globale
    # délavait les flancs et la planche.
    out = w[:, None] * photo_col + (1 - w[:, None]) * gen
    new = tex.copy(); new[used] = out
    stats = dict(from_photo=float((w > 0.5).mean()), shift=round(shift, 4))
    return Image.fromarray((np.clip(new, 0, 1) * 255).round().astype(np.uint8)), stats


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V); [np.add.at(vn, F[:, i], fn) for i in range(3)]
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------- correction de couleur locale
def _merge_vertices(V):
    key = np.round(V / (np.ptp(V, axis=0).max() * 1e-6)).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    return inv.ravel(), int(inv.max()) + 1


def _mesh_mean(values, Fu, n, fixed=None, iters=1):
    """Moyenne sur les voisins (graphe du maillage), en gardant éventuellement certains sommets fixes."""
    v = values.copy()
    deg = np.zeros(n); [np.add.at(deg, Fu[:, i], 2) for i in range(3)]
    for _ in range(iters):
        acc = np.zeros_like(v)
        for i, j in ((0, 1), (1, 2), (2, 0)):
            np.add.at(acc, Fu[:, i], v[Fu[:, j]]); np.add.at(acc, Fu[:, j], v[Fu[:, i]])
        new = acc / np.maximum(deg, 1)[:, None]
        v = np.where(fixed[:, None], v, new) if fixed is not None else new
    return v


def recolor_from_photo(texture, V, F, UV, photo_rgba, cam, blur=0.02, min_facing=0.25, max_gain=2.5, light_strength=0.35):
    """Garde les détails de la texture générée, mais recale la couleur de chaque zone sur la photo
    (version floutée : tolérante à un léger décalage entre le modèle et la photo).
    Les zones cachées reçoivent la correction des zones voisines visibles."""
    tex = np.asarray(texture.convert('RGB'), np.float32) / 255; T = tex.shape[0]
    ph = photo_rgba.convert('RGBA'); H, W = ph.size[1], ph.size[0]
    sig = blur * max(H, W)
    rgb = np.asarray(ph, np.float32)[..., :3] / 255; a = np.asarray(ph, np.float32)[..., 3:] / 255
    # flou pondéré par le masque (le fond retiré ne bave pas sur le plat)
    num = np.asarray(Image.fromarray(((rgb * a) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(sig)), np.float32)
    den = np.asarray(Image.fromarray((a[..., 0] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(sig)), np.float32)[..., None]
    photo_blur = num / np.maximum(den, 1) * 255     # = couleur moyenne locale du plat (0..255)
    inv, n = _merge_vertices(V); Fu = inv[F]
    Vu = np.zeros((n, 3)); np.add.at(Vu, inv, V); Vu /= np.bincount(inv, minlength=n)[:, None]
    Nu = vertex_normals(Vu, Fu)
    # couleur générée par sommet, lissée sur le maillage (même échelle de flou que la photo)
    vc = tex[np.clip(((1 - UV[:, 1]) * (T - 1)).astype(int), 0, T - 1), np.clip((UV[:, 0] * (T - 1)).astype(int), 0, T - 1)]
    gen = np.zeros((n, 3)); np.add.at(gen, inv, vc); gen /= np.bincount(inv, minlength=n)[:, None]
    edge = np.linalg.norm(Vu[Fu[:, 0]] - Vu[Fu[:, 1]], axis=1).mean()
    size_px = cam['fit'][0] * edge                               # longueur d'arête en pixels photo
    gen = _mesh_mean(gen, Fu, n, iters=int(np.clip((sig / max(size_px, 1e-6)) ** 2, 2, 60)))
    # visibilité des sommets depuis la photo
    x, y, z = _project(Vu - cam['center'], cam['R']); s, tx, ty = cam['fit']; x = x * s + tx; y = y * s + ty
    ins = (x >= 0) & (x < W - 1) & (y >= 0) & (y < H - 1)
    ix, iy = np.clip(x.astype(int), 0, W - 1), np.clip(y.astype(int), 0, H - 1)
    zbuf = np.full((H, W), np.inf); np.minimum.at(zbuf, (iy[ins], ix[ins]), z[ins])
    zbuf = minimum_filter(zbuf, size=5)
    facing = -(Nu @ cam['R'][2])
    vis = ins & (z <= zbuf[iy, ix] + 0.01 * np.ptp(z)) & (facing > min_facing) & (den[iy, ix, 0] > 200)
    target = _bilinear(photo_blur, x, y) / 255
    gain = np.ones((n, 3))
    gain[vis] = np.clip((target[vis] + 0.02) / (gen[vis] + 0.02), 1 / max_gain, max_gain)
    # zones cachées : correction des voisines visibles (diffusion), puis lissage d'ensemble
    gain[~vis] = np.median(gain[vis], axis=0) if vis.any() else 1
    gain = _mesh_mean(gain, Fu, n, fixed=vis, iters=80)
    gain = _mesh_mean(gain, Fu, n, iters=3)
    # application texel par texel (interpolation barycentrique des gains)
    _, g_tex, used = _texel_positions(V, F, UV, gain[inv].astype(np.float32), T)
    # étend la correction aux texels de bordure (sinon liseré non corrigé aux coutures)
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(~used, return_distances=False, return_indices=True)
    g_tex = g_tex[idx[0], idx[1]]
    out = tex * g_tex
    # luminosité : un seul réglage global (l'éclairage du viewer s'ajoute déjà) ; les teintes restent celles de la photo
    k = out[used].mean() / max(tex[used].mean(), 1e-6)
    out = np.clip(out / k * k ** light_strength, 0, 1)
    return Image.fromarray((out * 255).round().astype(np.uint8)), dict(visible_vertices=float(vis.mean()))

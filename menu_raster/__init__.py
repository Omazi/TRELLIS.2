"""menu_raster — rasteriseur de triangles en PyTorch pur, écrit pour Menu AR.

Remplace les quatre opérations dont notre pipeline a besoin (dessiner des triangles en pixels,
interpoler des attributs, lire une texture) avec les mêmes entrées et sorties que l'interface
publique documentée de nvdiffrast, pour pouvoir en changer sans toucher au code qui l'appelle.
Écrit de zéro à partir de la description des entrées/sorties ; aucune ligne n'est reprise
d'une autre bibliothèque. Licence : celle de ce dépôt (MIT).

Conventions (identiques à l'interface documentée de nvdiffrast) :
  * pos : positions en espace de découpage (x, y, z, w). Pixel (ligne i, colonne j) = centre
    ((j + 0.5) / W * 2 - 1, (i + 0.5) / H * 2 - 1) en coordonnées normalisées : la ligne 0 est
    en bas (y = -1), comme OpenGL.
  * rasterize -> rast[..., 4] = (u, v, z/w, id_triangle + 1) ; 0 = pas de triangle.
    Le point = u * sommet0 + v * sommet1 + (1 - u - v) * sommet2 (barycentres corrigés en
    perspective). Le triangle le plus proche (plus petit z/w) gagne ; pas d'élimination des faces
    arrière ; les points hors de -1 <= z/w <= 1 sont écartés.
  * interpolate -> attributs interpolés, 0 hors triangles.
  * texture -> échantillonnage « nearest » ou « linear » (bilinéaire), u le long de la largeur,
    v le long de la hauteur (v = 0 sur la première ligne), bords « wrap », « clamp » ou « zero ».

Limites assumées (non utilisées par le pipeline) : pas de dérivées (pas d'entraînement à travers
le rendu), pas de mipmaps, pas d'antialiasing analytique, triangles traversant w = 0 ignorés.
"""
from typing import Optional, Sequence, Tuple

import torch

__all__ = ['RasterizeCudaContext', 'RasterizeGLContext', 'rasterize', 'interpolate', 'texture', 'antialias',
           'backend']


def backend():
    """Module de rendu à utiliser : le nôtre par défaut, nvdiffrast seulement si on le demande
    explicitement (MENU_AR_RASTER=nvdiffrast), pour comparer les deux sur un même plat."""
    import os
    import sys
    if os.environ.get('MENU_AR_RASTER', 'libre').lower() == 'nvdiffrast':
        import nvdiffrast.torch as dr
        return dr
    return sys.modules[__name__]

_PAIRS_PER_CHUNK = 8_000_000      # couples (triangle, pixel candidat) traités à la fois


class RasterizeCudaContext:
    """Contexte factice : notre rasteriseur n'a pas d'état. Gardé pour la compatibilité."""

    def __init__(self, device=None):
        self.device = device


RasterizeGLContext = RasterizeCudaContext


def _screen(pos4: torch.Tensor, H: int, W: int):
    """Clip -> coordonnées pixel (centres de pixels sur les entiers) et z/w."""
    w = pos4[:, 3]
    inv_w = 1.0 / torch.where(w.abs() > 1e-12, w, torch.ones_like(w))
    x = (pos4[:, 0] * inv_w + 1) * (W * 0.5) - 0.5
    y = (pos4[:, 1] * inv_w + 1) * (H * 0.5) - 0.5
    z = pos4[:, 2] * inv_w
    return x, y, z, w


@torch.no_grad()
def _raster_one(pos4: torch.Tensor, tri: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Une image : renvoie rast [H, W, 4]."""
    dev = pos4.device
    out = torch.zeros(H, W, 4, dtype=torch.float32, device=dev)
    if tri.numel() == 0:
        return out
    pos4 = pos4.float(); tri = tri.long()
    sx, sy, sz, sw = _screen(pos4, H, W)
    X, Y, Z, Wc = sx[tri], sy[tri], sz[tri], sw[tri]          # [T, 3]
    # aire signée (double) dans l'écran ; triangles dégénérés ou derrière la caméra écartés
    area = (X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (X[:, 2] - X[:, 0]) * (Y[:, 1] - Y[:, 0])
    keep = (area.abs() > 1e-12) & (Wc > 0).all(1)
    x0 = torch.ceil(X.min(1).values).clamp(0, W - 1).long(); x1 = torch.floor(X.max(1).values).clamp(-1, W - 1).long()
    y0 = torch.ceil(Y.min(1).values).clamp(0, H - 1).long(); y1 = torch.floor(Y.max(1).values).clamp(-1, H - 1).long()
    keep &= (x1 >= x0) & (y1 >= y0) & (X.max(1).values >= 0) & (Y.max(1).values >= 0)
    keep &= (X.min(1).values <= W - 1) & (Y.min(1).values <= H - 1)
    ids = torch.nonzero(keep).squeeze(1)
    if ids.numel() == 0:
        return out
    bw = (x1 - x0 + 1)[ids]; bh = (y1 - y0 + 1)[ids]
    span = torch.maximum(bw, bh)

    zbuf = torch.full((H * W,), float('inf'), device=dev)
    win = torch.full((H * W,), -1, dtype=torch.long, device=dev)
    passes = []                                    # (indices pixels, triangles, profondeur) candidats

    def candidates(sel: torch.Tensor, k: int):
        """Pixels candidats (grille k x k depuis le coin de la boîte) des triangles sel."""
        g = torch.arange(k, device=dev)
        chunk = max(1, _PAIRS_PER_CHUNK // (k * k))
        for c in range(0, sel.numel(), chunk):
            t = sel[c:c + chunk]
            shape = (t.numel(), k, k)
            px = (x0[t, None, None] + g[None, None, :]).expand(shape)
            py = (y0[t, None, None] + g[None, :, None]).expand(shape)
            ok = (px <= x1[t, None, None]) & (py <= y1[t, None, None])
            Xt, Yt = X[t][:, :, None, None], Y[t][:, :, None, None]
            fx, fy = px.float(), py.float()
            # coordonnées barycentriques dans l'écran : poids du sommet i = aire du sous-triangle
            # opposé / aire totale
            a = area[t][:, None, None]
            b0 = ((Xt[:, 1] - fx) * (Yt[:, 2] - fy) - (Xt[:, 2] - fx) * (Yt[:, 1] - fy)) / a
            b1 = ((Xt[:, 2] - fx) * (Yt[:, 0] - fy) - (Xt[:, 0] - fx) * (Yt[:, 2] - fy)) / a
            b2 = 1 - b0 - b1
            eps = -1e-7
            ok &= (b0 >= eps) & (b1 >= eps) & (b2 >= eps)
            zz = b0 * Z[t][:, 0, None, None] + b1 * Z[t][:, 1, None, None] + b2 * Z[t][:, 2, None, None]
            ok &= (zz >= -1) & (zz <= 1)
            fi, yi, xi = torch.nonzero(ok, as_tuple=True)
            if fi.numel() == 0:
                continue
            pix = py[fi, yi, xi] * W + px[fi, yi, xi]
            yield pix, t[fi], zz[fi, yi, xi]

    sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    done = torch.zeros(ids.numel(), dtype=torch.bool, device=dev)
    for k in sizes:
        m = (~done) & (span <= k)
        if not m.any():
            continue
        done |= m
        for pix, t, zz in candidates(ids[m], k):
            zbuf.scatter_reduce_(0, pix, zz, reduce='amin')
            passes.append((pix, t, zz))
    # gagnant : profondeur minimale ; à égalité, le plus petit numéro de triangle (déterministe)
    big = torch.iinfo(torch.long).max
    win.fill_(big)
    for pix, t, zz in passes:
        front = zz <= zbuf[pix]
        win.scatter_reduce_(0, pix[front], t[front], reduce='amin')
    covered = torch.nonzero(win != big).squeeze(1)
    if covered.numel() == 0:
        return out
    t = win[covered]
    py = (covered // W).float(); px = (covered % W).float()
    Xt, Yt = X[t], Y[t]
    a = area[t]
    b0 = ((Xt[:, 1] - px) * (Yt[:, 2] - py) - (Xt[:, 2] - px) * (Yt[:, 1] - py)) / a
    b1 = ((Xt[:, 2] - px) * (Yt[:, 0] - py) - (Xt[:, 0] - px) * (Yt[:, 2] - py)) / a
    b2 = 1 - b0 - b1
    zz = b0 * Z[t, 0] + b1 * Z[t, 1] + b2 * Z[t, 2]
    # correction de perspective : poids divisés par w, renormalisés
    q0, q1, q2 = b0 / Wc[t, 0], b1 / Wc[t, 1], b2 / Wc[t, 2]
    s = q0 + q1 + q2
    flat = out.view(-1, 4)
    flat[covered, 0] = q0 / s
    flat[covered, 1] = q1 / s
    flat[covered, 2] = zz
    flat[covered, 3] = (t + 1).float()
    return out


def rasterize(glctx, pos: torch.Tensor, tri: torch.Tensor, resolution: Sequence[int],
              ranges: Optional[torch.Tensor] = None, grad_db: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """pos [n, V, 4] (mode instancié) ou [V, 4] avec ranges [n, 2] (début, nombre de triangles).
    Renvoie (rast [n, H, W, 4], dérivées) ; les dérivées sont toujours nulles ici."""
    H, W = int(resolution[0]), int(resolution[1])
    if pos.dim() == 3:
        imgs = [_raster_one(pos[b], tri, H, W) for b in range(pos.shape[0])]
    else:
        if ranges is None:
            raise ValueError('pos à 2 dimensions : ranges est obligatoire')
        imgs = []
        for start, count in ranges.tolist():
            sub = tri[start:start + count]
            r = _raster_one(pos, sub, H, W)
            r[..., 3] = torch.where(r[..., 3] > 0, r[..., 3] + start, r[..., 3])
            imgs.append(r)
    rast = torch.stack(imgs, 0)
    db = torch.zeros(1, device=rast.device).expand(*rast.shape[:3], 4 if grad_db else 0)
    return rast, db


def interpolate(attr: torch.Tensor, rast: torch.Tensor, tri: torch.Tensor, rast_db=None,
                diff_attrs=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """attr [n ou 1, V, A] ou [V, A] -> [n, H, W, A] ; 0 là où il n'y a pas de triangle."""
    if diff_attrs is not None:
        raise NotImplementedError('menu_raster : pas de dérivées d’attributs')
    n, H, Wd = rast.shape[:3]
    a = attr if attr.dim() == 3 else attr[None]
    A = a.shape[-1]
    out = torch.zeros(n, H, Wd, A, dtype=a.dtype, device=rast.device)
    tri = tri.long()
    for b in range(n):
        ab = a[b if a.shape[0] > 1 else 0]
        r = rast[b].reshape(-1, 4)
        idx = torch.nonzero(r[:, 3] > 0).squeeze(1)
        if idx.numel() == 0:
            continue
        t = r[idx, 3].long() - 1
        u, v = r[idx, 0:1].to(a.dtype), r[idx, 1:2].to(a.dtype)
        vt = tri[t]
        val = u * ab[vt[:, 0]] + v * ab[vt[:, 1]] + (1 - u - v) * ab[vt[:, 2]]
        out.view(n, -1, A)[b, idx] = val
    return out, torch.zeros(1, device=rast.device).expand(n, H, Wd, 0)


def _wrap_index(i: torch.Tensor, size: int, mode: str):
    if mode == 'wrap':
        return torch.remainder(i, size), torch.ones_like(i, dtype=torch.bool)
    inside = (i >= 0) & (i < size)
    return i.clamp(0, size - 1), (inside if mode == 'zero' else torch.ones_like(inside))


def texture(tex: torch.Tensor, uv: torch.Tensor, uv_da=None, mip_level_bias=None, mip=None,
            filter_mode: str = 'auto', boundary_mode: str = 'wrap', max_mip_level=None) -> torch.Tensor:
    """tex [n ou 1, Ht, Wt, C], uv [n, h, w, 2] -> [n, h, w, C]."""
    if filter_mode == 'auto':
        filter_mode = 'linear' if uv_da is None else 'linear-mipmap-linear'
    if filter_mode not in ('nearest', 'linear'):
        raise NotImplementedError(f'menu_raster : filtre {filter_mode} non pris en charge (pas de mipmaps)')
    if boundary_mode not in ('wrap', 'clamp', 'zero'):
        raise NotImplementedError(f'menu_raster : bord {boundary_mode} non pris en charge')
    n = uv.shape[0]; Ht, Wt, C = tex.shape[1:]
    out = []
    for b in range(n):
        t = tex[b if tex.shape[0] > 1 else 0]
        u = uv[b, ..., 0].float() * Wt; v = uv[b, ..., 1].float() * Ht
        if filter_mode == 'nearest':
            i, ok_i = _wrap_index(torch.floor(v).long(), Ht, boundary_mode)
            j, ok_j = _wrap_index(torch.floor(u).long(), Wt, boundary_mode)
            val = t[i, j] * (ok_i & ok_j)[..., None]
        else:
            x = u - 0.5; y = v - 0.5
            x0 = torch.floor(x); y0 = torch.floor(y)
            fx = (x - x0)[..., None].to(t.dtype); fy = (y - y0)[..., None].to(t.dtype)
            x0 = x0.long(); y0 = y0.long()
            val = 0
            for dy, wy in ((0, 1 - fy), (1, fy)):
                i, ok_i = _wrap_index(y0 + dy, Ht, boundary_mode)
                for dx, wx in ((0, 1 - fx), (1, fx)):
                    j, ok_j = _wrap_index(x0 + dx, Wt, boundary_mode)
                    val = val + t[i, j] * (wy * wx) * (ok_i & ok_j)[..., None]
        out.append(val)
    return torch.stack(out, 0)


def antialias(color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0):
    """Non utilisé par notre pipeline : l'antialiasing analytique n'est pas implémenté."""
    raise NotImplementedError('menu_raster : antialias non pris en charge')

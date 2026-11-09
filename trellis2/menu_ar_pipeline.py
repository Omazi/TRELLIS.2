"""Pipeline complet du menu AR : photo d'un plat -> GLB prêt pour le web et l'AR.

Même enchaînement que le notebook colab/menu_ar_trellis2.ipynb, regroupé en une fonction
pour être appelé par le site (webapp/server.py) ou par un script.

    pipe = MenuARPipeline()            # charge TRELLIS 2 + le détourage (une fois)
    result = pipe.run(Image.open('plat.jpg'), workdir='/content/jobs/abc', mode='tout')
    result['mobile_glb']               # -> /content/jobs/abc/plat_mobile.glb
"""
import os
import subprocess
import time
from typing import Callable, Optional

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOBILE_TOOLS = os.path.join(ROOT, 'tools', 'mobile_glb')


def code_version() -> str:
    """Commit du code réellement chargé (écrit dans chaque résultat, pour savoir quelle version a tourné)."""
    r = subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'], text=True, capture_output=True)
    return r.stdout.strip() or 'inconnue'


CODE_VERSION = code_version()

STEPS = [
    ('segmentation', 'Détourage du plat'),
    ('generation', 'Génération 3D (TRELLIS 2)'),
    ('mesh', 'Maillage et texture'),
    ('colors', 'Couleurs recalées sur la photo'),
    ('texture', 'Texture haute définition (IA)'),
    ('mobile', 'Optimisation pour téléphone'),
]


class MenuARPipeline:
    def __init__(self, trellis_pipeline=None, segmenter=None, texturer=None, model_id: str = 'microsoft/TRELLIS.2-4B',
                 hd_texture: bool = True):
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
        if trellis_pipeline is None:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline
            trellis_pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_id, load_rembg=False)
            trellis_pipeline.cuda()
        if segmenter is None:
            if os.environ.get('MENU_AR_LIBRE') == '1':      # version vente : uniquement des modèles libres
                from trellis2.utils.dish_segmentation import OpenDishSegmenter
                segmenter = OpenDishSegmenter()
            else:
                from trellis2.utils.dish_segmentation import AutoDishSegmenter
                segmenter = AutoDishSegmenter()
        self.texturer_error = None
        if hd_texture and texturer is None:
            try:
                from trellis2.utils.mv_texture import MVTexturer
                texturer = MVTexturer()
            except Exception as e:               # le site reste utilisable avec la texture photo
                import traceback; traceback.print_exc()
                self.texturer_error = f'chargement : {type(e).__name__}: {e}'[:400]
                print('⚠️ Texture IA indisponible : seule la texture photo sera produite.')
                texturer = None
        self.trellis = trellis_pipeline
        self.segmenter = segmenter
        self.texturer = texturer          # None = pas de texture IA (seulement la version recalée sur la photo)

    @staticmethod
    def _node(script: str, *args: str) -> str:
        r = subprocess.run(['node', script, *args], cwd=MOBILE_TOOLS, text=True, capture_output=True)
        if r.returncode:
            raise RuntimeError(f'{script} : {r.stderr[-800:]}')
        return r.stdout

    def run(self, image: Image.Image, workdir: str, mode: str = 'tout', size_m: float = 0.30,
            pipeline_type: str = '1536_cascade', seed: int = 42,
            progress: Optional[Callable[[str], None]] = None) -> dict:
        import o_voxel
        import trimesh
        from trellis2.utils.food_postprocess import solidify_base, paint_underside, floating_bits, rim_color_from_photo
        from trellis2.utils.photo_projection import estimate_camera, recolor_from_photo, paste_photo

        os.makedirs(workdir, exist_ok=True)
        say = progress or (lambda step: None)
        t0 = time.time()
        timings = {}

        def mark(step):
            timings[step] = round(time.time() - t0, 1)

        # 1. détourage
        say('segmentation')
        img = image.convert('RGB'); img.thumbnail((1536, 1536))
        img.save(os.path.join(workdir, 'photo.jpg'), quality=88)
        rgba = self.segmenter(img, keep_container=(mode != 'plat'))
        rgba_path = os.path.join(workdir, 'input_rgba.png'); rgba.save(rgba_path)
        mark('segmentation')

        # 2. génération
        say('generation')
        mesh = self.trellis.run(rgba, seed=seed, pipeline_type=pipeline_type)[0]
        mark('generation')

        # 3. maillage + texture
        say('mesh')
        mesh.simplify(16777216)
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
            attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=300000, texture_size=4096, remesh=True, remesh_band=1, remesh_project=0,
        )
        del mesh
        drop = floating_bits(glb.vertices, glb.faces)        # petits morceaux qui flottent à côté du plat
        if drop.any():
            glb.update_faces(~drop); glb.remove_unreferenced_vertices()
        glb.vertices, _ = solidify_base(glb.vertices, glb.faces)
        hd = os.path.join(workdir, 'plat_hd.glb'); glb.export(hd)
        del glb
        mark('mesh')

        # 4. couleurs recalées sur la photo + fond uni
        say('colors')
        scene = trimesh.load(hd); g = list(scene.geometry.values())[0]
        V, F, UV = np.asarray(g.vertices), np.asarray(g.faces), np.asarray(g.visual.uv)
        P = np.asarray(rgba)
        base = g.visual.material.baseColorTexture
        tex = np.asarray(base.convert('RGB'), np.float32) / 255; T = tex.shape[0]
        vcol = tex[np.clip(((1 - UV[:, 1]) * (T - 1)).astype(int), 0, T - 1), np.clip((UV[:, 0] * (T - 1)).astype(int), 0, T - 1)]
        cam = estimate_camera(V, F, P[..., 3] > 127, vertex_colors=vcol, photo_rgb=P[..., :3] / 255.)
        new_tex, stats = recolor_from_photo(base, V, F, UV, rgba, cam)
        new_tex, paste_stats = paste_photo(new_tex, V, F, UV, rgba, cam)   # la vraie photo sur le dessus
        under = rim_color_from_photo(rgba)                 # dessous = couleur du bord du contenant sur la photo
        new_tex = paint_underside(new_tex, V, F, UV, color=under)
        tex_path = os.path.join(workdir, 'texture_photo.png'); new_tex.save(tex_path)
        hd_photo = os.path.join(workdir, 'plat_hd_photo.glb')
        self._node('retexture.mjs', hd, tex_path, hd_photo)
        mark('colors')

        # 5. texture haute définition par MV-Adapter (6 vues guidées par la photo et la géométrie)
        hd_ia, texture_error = None, (self.texturer_error if self.texturer is None else None)
        if self.texturer is not None:
            say('texture')
            import torch
            torch.cuda.empty_cache()
            try:
                ia = self.texturer(hd, rgba, cam, workdir)
                ia, _ = paste_photo(ia, V, F, UV, rgba, cam)   # l'IA complète, la photo reste la photo
                ia = paint_underside(ia, V, F, UV, color=under)   # même dessous que la version photo
                ia_path = os.path.join(workdir, 'texture_ia.png'); ia.save(ia_path)
                out_path = os.path.join(workdir, 'plat_hd_ia.glb')
                self._node('retexture.mjs', hd, ia_path, out_path)
                hd_ia = out_path                             # seulement si tout a réussi
            except Exception as e:                       # on garde la version photo plutôt que d'échouer
                import traceback; traceback.print_exc()
                texture_error = f'{type(e).__name__}: {e}'[:400]
                hd_ia = None
                torch.cuda.empty_cache()
            mark('texture')

        # 6. versions mobiles : plat_mobile.glb = texture IA (sinon photo), plat_mobile_photo.glb = texture photo
        say('mobile')
        mobile = os.path.join(workdir, 'plat_mobile.glb')
        mobile_photo = os.path.join(workdir, 'plat_mobile_photo.glb')
        self._node('optimize.mjs', hd_photo, mobile_photo, str(size_m), '0.35')
        if hd_ia:
            self._node('optimize.mjs', hd_ia, mobile, str(size_m), '0.35')
        else:
            import shutil; shutil.copy(mobile_photo, mobile)
        mark('mobile')

        return dict(mobile_glb=mobile, mobile_photo_glb=mobile_photo, hd_glb=hd_ia or hd_photo,
                    photo=os.path.join(workdir, 'photo.jpg'), texture='ia' if hd_ia else 'photo',
                    texture_error=texture_error, version=CODE_VERSION,
                    objets=list(getattr(self.segmenter, 'last_found', []) or []),
                    camera=dict(az=cam['az'], el=cam['el'], roll=cam['roll'], iou=cam['iou']),
                    collage=dict(part=round(paste_stats['from_photo'], 3), recalage=paste_stats['shift']),
                    seconds=timings, size_mb=round(os.path.getsize(mobile) / 1e6, 2))

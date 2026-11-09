"""Texture nette par MV-Adapter (Apache-2.0) sur le maillage TRELLIS 2.

TRELLIS 2 produit une bonne forme mais une texture floue, aux couleurs qui dérivent.
MV-Adapter (SDXL) dessine 6 vues haute définition du plat, guidées à la fois par la photo
(référence) et par la géométrie de NOTRE maillage (cartes de position et de normales),
puis les reprojette sur nos UV (agrandissement Real-ESRGAN, retouche des zones cachées LaMa).

Alignement : MV-Adapter considère que l'image de référence est sa vue de face. On tourne donc
le maillage pour que sa caméra de face coïncide avec l'angle de prise de vue de la photo
(retrouvé par photo_projection.estimate_camera), inclinaison et roulis compris.
Seule la texture est récupérée : la géométrie et les UV du maillage d'origine ne changent pas.

    tex = MVTexturer()(hd_glb_path, rgba_photo, cam, workdir)   # -> PIL.Image (texture sur nos UV)
"""
import os
import shutil
import sys
import urllib.request
from typing import Optional

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MVA_DIR = os.path.join(ROOT, 'third_party', 'mvadapter')

CKPTS = {
    'RealESRGAN_x2plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    'big-lama.pt': 'https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt',
}
# Vues générées par MV-Adapter (mêmes valeurs que scripts/texture_i2tex.py du dépôt d'origine)
VIEW_AZIMUTHS = [0, 90, 180, 270, 180, 180]
VIEW_ELEVATIONS = [0, 0, 0, 0, 89.99, -89.99]


def alignment_matrix(cam: dict, mode: str = 'full') -> np.ndarray:
    """Rotation qui amène la caméra de la photo sur la caméra de face de MV-Adapter
    (placée en +Z, regard vers -Z, haut +Y dans le repère glTF du maillage).
    mode='full' : azimut + inclinaison + roulis ; 'azimuth' : seulement l'azimut (plat reste à plat)."""
    if mode == 'azimuth':
        a = np.radians(cam['az'])
        return np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    A = np.asarray(cam['R'], np.float64)          # lignes : droite, haut, direction de visée
    return np.diag([1.0, 1.0, -1.0]) @ A          # droite -> +X, haut -> +Y, visée -> -Z


class MVTexturer:
    def __init__(self, device: str = 'cuda', ckpt_dir: Optional[str] = None,
                 base_model: str = 'stabilityai/stable-diffusion-xl-base-1.0',
                 vae_model: str = 'madebyollin/sdxl-vae-fp16-fix', adapter: str = 'huanngzh/mv-adapter'):
        import torch
        from diffusers import AutoencoderKL
        if MVA_DIR not in sys.path:
            sys.path.insert(0, MVA_DIR)
        from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
        from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
        from mvadapter.pipelines.pipeline_texture import TexturePipeline
        from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler

        self.device, self.dtype, self.num_views = device, torch.float16, 6
        vae = AutoencoderKL.from_pretrained(vae_model, torch_dtype=self.dtype)
        pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(base_model, vae=vae, variant='fp16', torch_dtype=self.dtype)
        pipe.scheduler = ShiftSNRScheduler.from_scheduler(pipe.scheduler, shift_mode='interpolated', shift_scale=8.0,
                                                          scheduler_class=None)
        pipe.init_custom_adapter(num_views=self.num_views, self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0)
        pipe.load_custom_adapter(adapter, weight_name='mvadapter_ig2mv_sdxl.safetensors')
        pipe.to(device=device, dtype=self.dtype)
        pipe.cond_encoder.to(device=device, dtype=self.dtype)
        pipe.enable_vae_slicing()
        self.pipe = pipe

        ckpt_dir = ckpt_dir or os.environ.get('MENU_AR_WEIGHTS') or os.path.join(ROOT, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        paths = {}
        for name, url in CKPTS.items():
            p = os.path.join(ckpt_dir, name)
            if not os.path.exists(p) or os.path.getsize(p) < 1_000_000:
                tmp = os.path.join('/tmp', name + '.part')      # disque local d'abord (Drive coupe parfois)
                urllib.request.urlretrieve(url, tmp)
                shutil.copyfile(tmp, p + '.part'); os.replace(p + '.part', p); os.remove(tmp)
            paths[name] = p
        self.texture_pipe = TexturePipeline(upscaler_ckpt_path=paths['RealESRGAN_x2plus.pth'],
                                            inpaint_ckpt_path=paths['big-lama.pt'], device=device)

    def _views(self, mesh_path: str, reference: Image.Image, text: str, steps: int, guidance: float, seed: int,
               height: int = 768, width: int = 768):
        """6 vues générées (équivalent de scripts/inference_ig2mv_sdxl.run_pipeline)."""
        import torch
        from mvadapter.utils.mesh_utils import NVDiffRastContextWrapper, get_orthogonal_camera, load_mesh, render
        cameras = get_orthogonal_camera(
            elevation_deg=VIEW_ELEVATIONS, distance=[1.8] * self.num_views, left=-0.55, right=0.55, bottom=-0.55, top=0.55,
            azimuth_deg=[x - 90 for x in VIEW_AZIMUTHS], device=self.device)
        ctx = NVDiffRastContextWrapper(device=self.device)
        mesh = load_mesh(mesh_path, rescale=True, device=self.device)
        out = render(ctx, mesh, cameras, height=height, width=width, render_attr=False, normal_background=0.0)
        control = torch.cat([(out.pos + 0.5).clamp(0, 1), (out.normal / 2 + 0.5).clamp(0, 1)], dim=-1).permute(0, 3, 1, 2)
        front_mask = out.mask[0].squeeze(-1).detach().cpu().numpy() if out.mask is not None else None
        ref = _preprocess_reference(reference, height, width, front_mask)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        images = self.pipe(text, height=height, width=width, num_inference_steps=steps, guidance_scale=guidance,
                           num_images_per_prompt=self.num_views, control_image=control.to(self.device),
                           control_conditioning_scale=1.0, reference_image=ref, reference_conditioning_scale=1.0,
                           negative_prompt='watermark, ugly, deformed, noisy, blurry, low contrast, cartoon',
                           cross_attention_kwargs={'scale': 1.0}, generator=gen).images
        return images, ref

    def __call__(self, hd_glb: str, rgba: Image.Image, cam: dict, workdir: str, align: str = 'full',
                 text: str = 'high quality, realistic food photography, sharp details', steps: int = 50,
                 guidance: float = 3.0, seed: int = 42, uv_size: int = 4096) -> Image.Image:
        import torch
        import trimesh
        from mvadapter.pipelines.pipeline_texture import ModProcessConfig
        from mvadapter.utils import make_image_grid

        # 1. copie du maillage tournée vers la caméra de face (mêmes UV, même ordre des sommets)
        scene = trimesh.load(hd_glb, process=False)
        g = list(scene.geometry.values())[0]
        M = alignment_matrix(cam, align)
        V = (np.asarray(g.vertices) - cam['center']) @ M.T
        V -= (V.min(0) + V.max(0)) / 2                     # MV-Adapter met à l'échelle sans recentrer
        g.vertices = V
        aligned = os.path.join(workdir, 'mv_aligned.glb'); scene.export(aligned)

        # 2. 6 vues haute définition guidées par la photo + la géométrie
        images, ref = self._views(aligned, rgba, text, steps, guidance, seed)
        grid = os.path.join(workdir, 'mv_views.png'); make_image_grid(images, rows=1).save(grid)
        ref.save(os.path.join(workdir, 'mv_reference.png'))
        torch.cuda.empty_cache()

        # 3. reprojection sur nos UV (agrandissement x2, retouche des zones non vues).
        # La texture est interceptée avant l'écriture du GLB intermédiaire (qui la recompresse en JPEG 75).
        import mvadapter.pipelines.pipeline_texture as pt
        captured = {}
        original_save = pt.replace_mesh_texture_and_save
        pt.replace_mesh_texture_and_save = lambda *a, **k: captured.setdefault('texture', k.get('texture'))
        try:
            self.texture_pipe(mesh_path=aligned, save_dir=workdir, save_name='mv', uv_unwarp=False, uv_size=uv_size,
                                rgb_path=grid, rgb_process_config=ModProcessConfig(view_upscale=True, inpaint_mode='view'),
                                camera_azimuth_deg=[x - 90 for x in VIEW_AZIMUTHS])
        finally:
            pt.replace_mesh_texture_and_save = original_save
            torch.cuda.empty_cache()
        tex = captured.get('texture')
        if tex is None:
            raise RuntimeError('MV-Adapter n’a pas produit de texture')
        if not isinstance(tex, Image.Image):
            from mvadapter.utils import tensor_to_image
            tex = tensor_to_image(tex)
        return tex.convert('RGB')


def _preprocess_reference(image: Image.Image, height: int, width: int, front_mask: Optional[np.ndarray] = None) -> Image.Image:
    """Recadre sur le plat sur fond gris. Si front_mask (silhouette du maillage en vue de face) est donné,
    la photo est placée et dimensionnée exactement comme cette silhouette (même cadrage que le guidage
    géométrique) ; sinon 90 % du cadre, centré (comportement d'origine de MV-Adapter)."""
    a = np.array(image.convert('RGBA'))
    ys, xs = np.where(a[..., 3] > 0)
    y0, y1 = max(ys.min() - 1, 0), min(ys.max() + 1, a.shape[0])
    x0, x1 = max(xs.min() - 1, 0), min(xs.max() + 1, a.shape[1])
    crop = a[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    my, mx = (np.where(front_mask) if front_mask is not None and front_mask.any() else ((), ()))
    if len(my):
        bh, bw = my.max() - my.min() + 1, mx.max() - mx.min() + 1
        s = np.sqrt((bh / h) * (bw / w))                       # même surface que la silhouette
        s = min(s, 0.98 * min(height / h, width / w))
        cy, cx = (my.min() + my.max()) / 2, (mx.min() + mx.max()) / 2
    else:
        s = 0.9 * min(height / h, width / w)
        cy, cx = height / 2, width / 2
    crop = np.array(Image.fromarray(crop).resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS))
    h, w = crop.shape[:2]
    top = int(np.clip(round(cy - h / 2), 0, height - h)); left = int(np.clip(round(cx - w / 2), 0, width - w))
    canvas = np.zeros((height, width, 4), np.uint8)
    canvas[top:top + h, left:left + w] = crop
    c = canvas.astype(np.float32) / 255
    rgb = c[..., :3] * c[..., 3:] + (1 - c[..., 3:]) * 0.5
    return Image.fromarray((rgb * 255).round().clip(0, 255).astype(np.uint8))

# Branche « vente » : uniquement des composants utilisables commercialement

| Composant | Rôle | Licence |
|---|---|---|
| TRELLIS.2, o-voxel, utils3d | forme 3D, maillage | MIT |
| CuMesh (branche `vente`), FlexGEMM | calcul 3D GPU | MIT (gpu_memory.h réécrit) |
| menu_raster (ce dépôt) | rendu de triangles, remplace nvdiffrast | MIT, écrit de zéro |
| MV-Adapter | texture IA | Apache 2.0 |
| SDXL base 1.0 | modèle d'image de MV-Adapter | CreativeML Open RAIL++-M (commercial autorisé, avec restrictions d'usage) |
| VAE SDXL fp16-fix | | MIT |
| Real-ESRGAN, LaMa | agrandissement, retouche | BSD-3 / Apache 2.0 |
| BiRefNet (rembg) | détourage | MIT |
| Grounding DINO, SAM 2.1 | assiette, sauces, sachets | Apache 2.0 |
| DINOv3 (Meta) | utilisé par TRELLIS.2 | DINOv3 License (texte : third_party/DINOv3_LICENSE.md) : usage commercial non interdit, gratuit ; « Built with DINOv3 » affiché sur le site ; interdits : sanctions/ITAR/militaire, rétro-ingénierie ; Meta peut modifier la licence. À faire relire par un juriste. |

Retirés de cette branche : nvdiffrast (NVIDIA, non commercial), Ultralytics YOLOE/SAM (AGPL-3.0),
`cubvh/include/gpu/gpu_memory.h` d'origine (NVIDIA, non commercial).
Ce tableau n'est pas un avis juridique.

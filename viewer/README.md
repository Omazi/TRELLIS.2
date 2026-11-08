# Viewer menu AR

Page web statique : photo du plat → vue 3D (rotation au doigt) → bouton « Voir sur ma table » (AR).
Aucune app à installer : AR via Scene Viewer (Android/Chrome), Quick Look (iPhone/Safari), WebXR.

Pour tester un plat : copier `plat_mobile.glb` (sortie du notebook `colab/`) en `plat.glb` et la photo en
`photo.jpg` dans ce dossier, puis héberger le dossier en HTTPS (l'AR exige une adresse publique en HTTPS).

- `model-viewer.min.js` : @google/model-viewer 4.4.0 (Apache-2.0), embarqué pour ne dépendre d'aucun CDN
- `meshopt_decoder.js` : décodeur meshoptimizer 1.3 (MIT), nécessaire aux GLB compressés par `tools/mobile_glb`

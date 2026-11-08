// Remplace la texture de couleur (baseColor) d'un GLB par une image, sans toucher au reste.
// Usage : node retexture.mjs entree.glb texture.(png|jpg) sortie.glb
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import fs from 'fs';
const [,, inp, img, out] = process.argv;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS);
const doc = await io.read(inp);
const mime = img.toLowerCase().endsWith('.png') ? 'image/png' : 'image/jpeg';
let n = 0;
for (const m of doc.getRoot().listMaterials()) {
  const t = m.getBaseColorTexture();
  if (t) { t.setImage(new Uint8Array(fs.readFileSync(img))).setMimeType(mime); n++; }
}
if (!n) { console.error('Aucune texture de couleur trouvée'); process.exit(1); }
await io.write(out, doc);
console.log(`texture remplacée (${n} matériau)`);

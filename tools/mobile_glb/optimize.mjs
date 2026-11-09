// Usage : node optimize.mjs entree.glb sortie.glb [taille_m=0.30] [ombre=0.35]
// taille_m : plus grand côté horizontal du plat en mètres. ombre : part de la hauteur assombrie près de la table (0 = désactivé).
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { dedup, weld, prune, simplify, textureCompress, meshopt, getBounds } from '@gltf-transform/functions';
import { MeshoptSimplifier, MeshoptEncoder, MeshoptDecoder } from 'meshoptimizer';
import sharp from 'sharp';
const [,, inp, out, size = '0.30', aoArg = '0.35'] = process.argv;
const AO = parseFloat(aoArg);
await MeshoptSimplifier.ready; await MeshoptEncoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.encoder': MeshoptEncoder, 'meshopt.decoder': MeshoptDecoder });
const doc = await io.read(inp);
const scene = doc.getRoot().getDefaultScene() || doc.getRoot().listScenes()[0];
// taille réelle : plus grand côté horizontal = `size` m, posé sur y=0, centré
let b = getBounds(scene);
const s = parseFloat(size) / Math.max(b.max[0] - b.min[0], b.max[2] - b.min[2]);
const root = doc.createNode('plat');
for (const n of scene.listChildren()) { scene.removeChild(n); root.addChild(n); }
scene.addChild(root);
root.setScale([s, s, s]);
root.setTranslation([-(b.min[0] + b.max[0]) / 2 * s, -b.min[1] * s, -(b.min[2] + b.max[2]) / 2 * s]);
// ombre de contact : assombrit le bas du plat (COLOR_0 multiplie la couleur de la texture)
if (AO > 0) {
  for (const mesh of doc.getRoot().listMeshes()) for (const prim of mesh.listPrimitives()) {
    const pos = prim.getAttribute('POSITION'); const n = pos.getCount(); const p = [0, 0, 0];
    let ymin = Infinity, ymax = -Infinity;
    for (let i = 0; i < n; i++) { pos.getElement(i, p); ymin = Math.min(ymin, p[1]); ymax = Math.max(ymax, p[1]); }
    const H = ymax - ymin, col = new Float32Array(n * 3);
    const nor = prim.getAttribute('NORMAL'); const q = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      pos.getElement(i, p);
      let t = Math.min(1, Math.max(0, (p[1] - ymin) / (AO * H)));
      t = t * t * (3 - 2 * t);                    // smoothstep
      let k = 0.45 + 0.55 * t;                    // 45 % de luminosité au contact de la table
      if (nor) { nor.getElement(i, q); if (q[1] < -0.5) k = 0.9; }   // dessous : pas d'ombre de contact
      col[i * 3] = col[i * 3 + 1] = col[i * 3 + 2] = k;
    }
    prim.setAttribute('COLOR_0', doc.createAccessor().setType('VEC3').setArray(col).setBuffer(pos.getBuffer()));
  }
}
// texture métal/rugosité quasi uniforme (cas de TRELLIS sur la nourriture) -> remplacée par 2 valeurs
for (const m of doc.getRoot().listMaterials()) {
  const t = m.getMetallicRoughnessTexture(); if (!t) continue;
  const { channels } = await sharp(Buffer.from(t.getImage())).stats();
  if (channels[1].stdev < 25 && channels[2].stdev < 12) {   // rugosité peu variable, pas de métal
    m.setRoughnessFactor(m.getRoughnessFactor() * channels[1].mean / 255);
    m.setMetallicFactor(m.getMetallicFactor() * channels[2].mean / 255);
    m.setMetallicRoughnessTexture(null);
  }
}
// on ne simplifie que les très gros maillages : la simplification fait « fondre » les détails fins (frites)
let faces = 0;
for (const m of doc.getRoot().listMeshes()) for (const p of m.listPrimitives()) faces += p.getIndices().getCount() / 3;
const MAX_FACES = 400000;
await doc.transform(
  dedup(), weld(),
  ...(faces > MAX_FACES ? [simplify({ simplifier: MeshoptSimplifier, ratio: MAX_FACES / faces, error: 0.0005 })] : []),
  textureCompress({ encoder: sharp, targetFormat: 'jpeg', resize: [2048, 2048], quality: 85 }),
  prune(),
  meshopt({ encoder: MeshoptEncoder, level: 'medium' }),   // compression forte (décodeur inclus dans model-viewer)
);
await io.write(out, doc);
b = getBounds(scene);
console.log('taille (m) :', (b.max[0]-b.min[0]).toFixed(3), (b.max[1]-b.min[1]).toFixed(3), (b.max[2]-b.min[2]).toFixed(3), '| bas y =', b.min[1].toFixed(3));

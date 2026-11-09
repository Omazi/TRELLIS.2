"""Site du menu AR : photo -> génération 3D (GPU) -> vue 3D + AR.

    from webapp.server import create_app, serve
    app = create_app(run_job=pipe.run, data_dir='/content/menu_ar_jobs', access_key='abcd')
    serve(app, port=8000)

Les générations passent une par une dans une file (un seul GPU). Chaque plat est gardé dans
data_dir/<id>/ (photo, GLB, état) et reste consultable à l'adresse /j/<id>.
L'accès est protégé par une clé : le lien d'entrée /?k=<clé> pose un cookie.
"""
import io
import json
import os
import queue
import threading
import time
import traceback
import uuid
from typing import Callable, List

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWER_DIR = os.path.join(os.path.dirname(HERE), 'viewer')
STEP_LABELS = {
    'queued': 'En attente',
    'segmentation': 'Détourage du plat',
    'generation': 'Génération 3D',
    'mesh': 'Maillage et texture',
    'colors': 'Couleurs recalées sur la photo',
    'texture': 'Texture haute définition (IA)',
    'mobile': 'Optimisation pour téléphone',
    'done': 'Terminé',
    'error': 'Erreur',
}


TEXTURE_SWITCH = '''
<div id="texsw" style="position:absolute;top:62px;left:50%;transform:translateX(-50%);z-index:20;display:flex;
  background:#efe8dd;border-radius:10px;padding:3px;font:14px system-ui,sans-serif">
  <button data-f="plat_mobile.glb" class="on">Texture IA</button><button data-f="plat_mobile_photo.glb">Texture photo</button>
</div>
<style>#texsw button{white-space:nowrap;border:0;background:transparent;padding:8px 12px;border-radius:8px;color:#7a7268;cursor:pointer}
#texsw button.on{background:#fff;color:#1d1a16;font-weight:600;box-shadow:0 1px 2px rgba(0,0,0,.08)}</style>
<script>
(function () {
  const sw = document.getElementById('texsw'); document.getElementById('viewer').appendChild(sw);
  sw.querySelectorAll('button').forEach(b => b.onclick = () => {
    sw.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    document.getElementById('mv').src = '/files/__ID__/' + b.dataset.f;
  });
})();
</script>'''


MAX_PHOTOS = 6


def create_app(run_job: Callable, data_dir: str, access_key: str) -> FastAPI:
    os.makedirs(data_dir, exist_ok=True)
    app = FastAPI(title='Menu AR')
    jobs_q: 'queue.Queue[str]' = queue.Queue()

    def job_dir(jid: str) -> str:
        if not jid.isalnum():
            raise HTTPException(404)
        return os.path.join(data_dir, jid)

    def read_state(jid: str) -> dict:
        p = os.path.join(job_dir(jid), 'state.json')
        if not os.path.exists(p):
            raise HTTPException(404, 'Plat introuvable')
        with open(p) as f:
            return json.load(f)

    def write_state(jid: str, **upd) -> dict:
        p = os.path.join(job_dir(jid), 'state.json')
        st = {}
        if os.path.exists(p):
            with open(p) as f:
                st = json.load(f)
        st.update(upd)
        with open(p + '.tmp', 'w') as f:
            json.dump(st, f, ensure_ascii=False)
        os.replace(p + '.tmp', p)
        return st

    def upload_files(jid):
        names = [f for f in os.listdir(job_dir(jid)) if f.startswith('upload') and f.endswith('.jpg')]
        return sorted(names, key=lambda f: 1 if f == 'upload.jpg' else int(f[7:-4]))

    def worker():
        while True:
            jid = jobs_q.get()
            st = read_state(jid)
            write_state(jid, step='segmentation', started=time.time())
            try:
                imgs = [Image.open(os.path.join(job_dir(jid), f)) for f in upload_files(jid)]
                res = run_job(imgs if len(imgs) > 1 else imgs[0], workdir=job_dir(jid), mode=st['mode'], size_m=st['size_m'],
                              progress=lambda step: write_state(jid, step=step))
                write_state(jid, step='done', finished=time.time(), result={
                    k: v for k, v in res.items() if k in ('camera', 'seconds', 'size_mb', 'texture', 'texture_error', 'version', 'objets', 'collage',
                                                   'photos', 'cameras')})
            except Exception as e:                       # l'erreur est affichée sur la page
                traceback.print_exc()
                write_state(jid, step='error', error=f'{type(e).__name__}: {e}'[:600], finished=time.time())
            finally:
                jobs_q.task_done()

    threading.Thread(target=worker, daemon=True).start()

    def check(request: Request):
        if request.cookies.get('k') != access_key and request.query_params.get('k') != access_key:
            raise HTTPException(403, 'Lien d’accès invalide : ouvre le lien complet affiché dans Colab.')

    def page(name: str) -> str:
        with open(os.path.join(HERE, name), encoding='utf-8') as f:
            return f.read()

    @app.get('/', response_class=HTMLResponse)
    def home(request: Request):
        check(request)
        resp = HTMLResponse(page('upload.html'))
        if request.query_params.get('k') == access_key:
            resp.set_cookie('k', access_key, max_age=7 * 24 * 3600, httponly=True, samesite='lax', secure=True)
        return resp

    @app.post('/api/jobs')
    async def create_job(request: Request, photo: List[UploadFile] = File(...), name: str = Form('Mon plat'),
                         mode: str = Form('tout'), size_cm: float = Form(30)):
        """Une ou plusieurs photos du même plat (6 au plus) ; la première est la photo principale."""
        check(request)
        imgs = []
        for up in photo[:MAX_PHOTOS]:
            try:
                imgs.append(ImageOps.exif_transpose(Image.open(io.BytesIO(await up.read()))).convert('RGB'))
            except Exception:
                raise HTTPException(400, f'Image illisible : {up.filename}')
        if not imgs:
            raise HTTPException(400, 'Aucune photo')
        jid = uuid.uuid4().hex[:10]
        os.makedirs(job_dir(jid))
        for k, img in enumerate(imgs):
            img.thumbnail((2048, 2048))
            img.save(os.path.join(job_dir(jid), 'upload.jpg' if k == 0 else f'upload_{k + 1}.jpg'), quality=92)
        write_state(jid, id=jid, name=(name or 'Mon plat')[:60], mode='plat' if mode == 'plat' else 'tout', photos=len(imgs),
                    size_m=max(0.05, min(1.0, size_cm / 100)), step='queued', created=time.time())
        jobs_q.put(jid)
        return {'id': jid}

    @app.get('/api/jobs/{jid}')
    def job_status(jid: str, request: Request):
        check(request)
        st = read_state(jid)
        pos = None
        if st['step'] == 'queued':
            pos = sum(1 for j in all_states() if j['step'] == 'queued' and j['created'] < st['created']) + 1
        now = time.time()
        return {**st, 'label': STEP_LABELS.get(st['step'], st['step']), 'queue_position': pos,
                'elapsed': round((st.get('finished') or now) - st.get('started', now))}

    def all_states():
        out = []
        for jid in os.listdir(data_dir):
            p = os.path.join(data_dir, jid, 'state.json')
            if os.path.exists(p):
                with open(p) as f:
                    out.append(json.load(f))
        return sorted(out, key=lambda s: -s['created'])

    @app.get('/api/jobs')
    def job_list(request: Request):
        check(request)
        return [{k: s.get(k) for k in ('id', 'name', 'step', 'created')} for s in all_states()[:30]]

    @app.get('/j/{jid}', response_class=HTMLResponse)
    def viewer(jid: str, request: Request):
        check(request)
        st = read_state(jid)
        if st['step'] != 'done':
            return RedirectResponse(f'/attente/{jid}')
        html = page_viewer()
        name = (st.get('name') or 'Mon plat').replace('&', '&amp;').replace('<', '&lt;')
        html = html.replace('Wrap gratiné &amp; frites', name)
        html = html.replace("const MODEL_SRC = 'plat.glb', PHOTO_SRC = 'photo.jpg';",
                            f"const MODEL_SRC = '/files/{jid}/plat_mobile.glb', PHOTO_SRC = '/files/{jid}/photo.jpg';")
        html = html.replace('<main class="wrap">', '<main class="wrap"><a href="/" style="color:#7a7268;font-size:14px;text-decoration:none">← Nouveau plat</a>', 1)
        extra = "<script>document.getElementById('openBtn').click()</script>"      # ouvre la 3D directement
        if (st.get('result') or {}).get('texture') == 'ia' and os.path.exists(os.path.join(job_dir(jid), 'plat_mobile_photo.glb')):
            extra += TEXTURE_SWITCH.replace('__ID__', jid)
        r = st.get('result') or {}
        info = f"version {r.get('version', 'ancienne')} · texture {r.get('texture', 'photo')}"
        if r.get('texture_error'):
            info += ' · texture IA en échec : ' + r['texture_error'][:160]
        extra += ('<div style="position:fixed;left:8px;bottom:6px;z-index:99;font:11px system-ui;color:#9a9288">'
                  + info.replace('<', '&lt;') + '</div>')
        return html.replace('</body>', extra + '\n</body>')

    def page_viewer() -> str:
        with open(os.path.join(VIEWER_DIR, 'index.html'), encoding='utf-8') as f:
            html = f.read()
        html = html.replace('src="model-viewer.min.js"', 'src="/static/model-viewer.min.js"')
        return html.replace('meshoptDecoderLocation: "meshopt_decoder.js"', 'meshoptDecoderLocation: "/static/meshopt_decoder.js"')

    @app.get('/attente/{jid}', response_class=HTMLResponse)
    def waiting(jid: str, request: Request):
        check(request); read_state(jid)
        return page('progress.html')

    @app.get('/files/{jid}/{fname}')
    def files(jid: str, fname: str, request: Request):
        # pas de vérification de clé ici : Scene Viewer (Android) télécharge le GLB sans le cookie.
        # Les noms de dossier sont aléatoires (10 caractères) et les fichiers limités au résultat.
        if fname not in ('plat_mobile.glb', 'plat_mobile_photo.glb', 'photo.jpg'):
            raise HTTPException(404)
        p = os.path.join(job_dir(jid), fname)
        if not os.path.exists(p):
            raise HTTPException(404)
        mt = 'model/gltf-binary' if fname.endswith('.glb') else 'image/jpeg'
        return FileResponse(p, media_type=mt, headers={'Access-Control-Allow-Origin': '*'})

    @app.get('/static/{fname}')
    def static(fname: str):
        if fname not in ('model-viewer.min.js', 'meshopt_decoder.js'):
            raise HTTPException(404)
        return FileResponse(os.path.join(VIEWER_DIR, fname), media_type='text/javascript')

    @app.exception_handler(HTTPException)
    def http_error(request: Request, exc: HTTPException):
        if request.url.path.startswith('/api/'):
            return JSONResponse({'detail': exc.detail}, status_code=exc.status_code)
        return HTMLResponse(f'<meta name="viewport" content="width=device-width,initial-scale=1">'
                            f'<p style="font-family:system-ui;padding:24px">{exc.detail or "Introuvable"}</p>',
                            status_code=exc.status_code)

    return app


def serve(app: FastAPI, port: int = 8000) -> threading.Thread:
    """Lance le serveur en arrière-plan (le notebook reste utilisable)."""
    import uvicorn
    config = uvicorn.Config(app, host='0.0.0.0', port=port, log_level='warning')
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True); t.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    return t

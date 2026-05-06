"""
Génère runs/<run>/monitor.html à chaque step/epoch.
Données 100 % inline — aucun chargement externe, compatible file://.
Auto-refresh toutes les 8 secondes.
"""

import base64
import json
import os
import time
from datetime import datetime, timedelta

_RUN_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>ContourDiff — __RUN_LABEL__</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0f0f1a; color: #cdd6f4; font-family: 'Courier New', monospace; padding: 24px; }
    h1  { font-size: 18px; color: #89b4fa; margin-bottom: 20px; letter-spacing: 1px; }
    .stats { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
    .card  { background: #1e1e2e; border: 1px solid #313244; border-radius: 10px;
             padding: 14px 22px; min-width: 150px; }
    .card-blue   .val { font-size: 26px; font-weight: bold; color: #89b4fa; }
    .card-green  .val { font-size: 26px; font-weight: bold; color: #a6e3a1; }
    .card-red    .val { font-size: 26px; font-weight: bold; color: #f38ba8; }
    .card-yellow .val { font-size: 26px; font-weight: bold; color: #f9e2af; }
    .card-purple .val { font-size: 26px; font-weight: bold; color: #cba6f7; }
    .card-gray   .val { font-size: 26px; font-weight: bold; color: #9399b2; }
    .card .label { font-size: 11px; color: #6c7086; margin-top: 4px; }
    canvas { display: block; background: #1e1e2e; border: 1px solid #313244;
             border-radius: 10px; margin-bottom: 16px; }
    .chart-title { font-size: 12px; color: #6c7086; margin-bottom: 6px; letter-spacing: 1px; }
    .refresh-note { font-size: 11px; color: #45475a; margin-top: 16px; }
    .sample-nav { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
    .sample-nav button {
      background: #1e1e2e; border: 1px solid #313244; color: #cdd6f4;
      border-radius: 6px; padding: 4px 14px; font-family: 'Courier New', monospace;
      font-size: 16px; cursor: pointer;
    }
    .sample-nav button:hover { background: #313244; }
    .sample-nav button:disabled { opacity: 0.3; cursor: default; }
    .sample-nav input[type=range] { flex: 1; accent-color: #89b4fa; }
    .sample-nav .epoch-label { font-size: 13px; color: #89b4fa; min-width: 120px; text-align: right; }
    .img-clickable { cursor: zoom-in; transition: opacity 0.15s; }
    .img-clickable:hover { opacity: 0.85; }
    .ref-thumb { border-radius: 6px; border: 1px solid #313244; width: 420px; height: 420px;
                 object-fit: contain; background: #1e1e2e; }
    /* lightbox */
    #lb { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.88);
          z-index: 9999; align-items: center; justify-content: center; }
    #lb.open { display: flex; }
    #lb-img { width: 80vmin; height: 80vmin; object-fit: contain;
              border-radius: 10px; box-shadow: 0 0 60px rgba(0,0,0,0.8);
              image-rendering: pixelated; }
    #lb-close { position: fixed; top: 18px; right: 24px; font-size: 28px; color: #cdd6f4;
                cursor: pointer; background: none; border: none; line-height: 1; }
    #lb-close:hover { color: #f38ba8; }
    #lb-label { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
                font-size: 12px; color: #6c7086; font-family: 'Courier New', monospace;
                letter-spacing: 1px; }
    #lb-prev, #lb-next { position: fixed; top: 50%; transform: translateY(-50%);
                          font-size: 28px; color: #cdd6f4; background: rgba(30,30,46,0.7);
                          border: 1px solid #313244; border-radius: 8px; padding: 8px 14px;
                          cursor: pointer; }
    #lb-prev { left: 16px; } #lb-next { right: 16px; }
    #lb-prev:hover, #lb-next:hover { background: #313244; }
    #lb-prev:disabled, #lb-next:disabled { opacity: 0.2; cursor: default; }
  </style>
</head>
<body>
  <h1>&#x2B21; ContourDiff &#x2014; __RUN_LABEL__</h1>

  <div class="stats">
    <div class="card card-blue">
      <div class="val" id="stat-epoch">—</div>
      <div class="label">EPOCH</div>
    </div>
    <div class="card card-green">
      <div class="val" id="stat-loss">—</div>
      <div class="label">TRAIN LOSS (DERNIER STEP)</div>
    </div>
    <div class="card card-yellow">
      <div class="val" id="stat-avg">—</div>
      <div class="label">TRAIN LOSS MOY. ÉPOQUE</div>
    </div>
    <div class="card card-purple">
      <div class="val" id="stat-val">—</div>
      <div class="label">VAL LOSS MOY. ÉPOQUE</div>
    </div>
    <div class="card card-red">
      <div class="val" id="stat-lr">—</div>
      <div class="label">LEARNING RATE</div>
    </div>
    <div class="card card-gray">
      <div class="val" id="stat-eta">—</div>
      <div class="label">ETA</div>
    </div>
    <div class="card card-gray">
      <div class="val" id="stat-elapsed">—</div>
      <div class="label">TEMPS ÉCOULÉ</div>
    </div>
  </div>

  <p class="chart-title">LOSS PAR STEP (TRAIN)</p>
  <canvas id="c-step" width="960" height="300"></canvas>

  <p class="chart-title">LOSS MOYENNE PAR ÉPOQUE — TRAIN vs VAL</p>
  <canvas id="c-epoch" width="960" height="240"></canvas>

  <p class="chart-title">LEARNING RATE</p>
  <canvas id="c-lr" width="960" height="160"></canvas>

  <div id="sample-section" style="display:none; margin-top:24px;">
    <p class="chart-title">ÉCHANTILLONS GÉNÉRÉS &nbsp;(4 bruits × 4 patients)</p>
    <div class="sample-nav">
      <button id="btn-prev" onclick="sampleNav(-1)">&#9664;</button>
      <input type="range" id="sample-slider" min="0" max="0" value="0" oninput="sampleSlide(this.value)">
      <button id="btn-next" onclick="sampleNav(1)">&#9654;</button>
      <span class="epoch-label" id="sample-epoch-label">—</span>
    </div>
    <img id="sample-img" class="img-clickable" onclick="lbOpenSample(sampleIdx)"
         style="width:100%; max-width:800px; height:500px; object-fit:contain; image-rendering:pixelated;
                border-radius:10px; border:1px solid #313244; background:#1e1e2e; display:block; margin-bottom:16px;" />

    <div id="ref-section" style="display:none;">
      <p class="chart-title" style="margin-top:8px;">RÉFÉRENCES (fixes)</p>
      <div style="display:flex; gap:16px; flex-wrap:wrap; margin-top:8px;">
        <div id="ref-ori-wrap"     style="display:none;">
          <p class="chart-title">ORI</p>
          <img id="ref-ori" class="img-clickable ref-thumb" onclick="lbOpenRef('ori')" />
        </div>
        <div id="ref-contour-wrap" style="display:none;">
          <p class="chart-title">CONTOUR</p>
          <img id="ref-contour" class="img-clickable ref-thumb" onclick="lbOpenRef('contour')" />
        </div>
        <div id="ref-near-wrap"    style="display:none;">
          <p class="chart-title">NEAR</p>
          <img id="ref-near" class="img-clickable ref-thumb" onclick="lbOpenRef('near')" />
        </div>
      </div>
    </div>
  </div>

  <!-- lightbox -->
  <div id="lb" onclick="lbClose()">
    <button id="lb-close" onclick="lbClose()">&#x2715;</button>
    <button id="lb-prev"  onclick="event.stopPropagation(); lbNav(-1)">&#9664;</button>
    <img id="lb-img" onclick="event.stopPropagation()" />
    <button id="lb-next"  onclick="event.stopPropagation(); lbNav(1)">&#9654;</button>
    <span id="lb-label"></span>
  </div>

  <p class="refresh-note">&#x21BB; rafraîchissement auto toutes les 8 s — dernière mise à jour : <span id="updated"></span></p>

  <script>
  var D = __DATA__;

  var cfg = D.config;
  document.getElementById('stat-epoch').textContent =
      D.epoch + ' / ' + (cfg.num_epochs || '?');
  document.getElementById('stat-loss').textContent =
      D.step_losses.length ? D.step_losses[D.step_losses.length-1].toFixed(5) : '—';
  document.getElementById('stat-avg').textContent =
      D.epoch_avg.length ? D.epoch_avg[D.epoch_avg.length-1].toFixed(5) : '—';
  document.getElementById('stat-val').textContent =
      D.val_avg && D.val_avg.length ? D.val_avg[D.val_avg.length-1].toFixed(5) : '—';
  document.getElementById('stat-lr').textContent =
      D.lr.length ? D.lr[D.lr.length-1].toExponential(2) : '—';
  document.getElementById('stat-eta').textContent     = D.eta     || '—';
  document.getElementById('stat-elapsed').textContent = D.elapsed || '—';
  document.getElementById('updated').textContent      = D.updated_at;

  var PALETTE = ['#89b4fa', '#cba6f7', '#f9e2af', '#f38ba8', '#a6e3a1'];

  function chart(id, series, opts) {
    var cv = document.getElementById(id);
    if (!cv) return;
    var ctx = cv.getContext('2d');
    var W = cv.width, H = cv.height;
    var pad = {t: 24, r: 20, b: 36, l: 72};
    var cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#1e1e2e'; ctx.fillRect(0, 0, W, H);

    var allY = series.flatMap(function(s){ return s.y; });
    if (!allY.length) return;

    var yMin = (opts && opts.yMin != null) ? opts.yMin : Math.min.apply(null, allY);
    var yMax = Math.max.apply(null, allY);
    var yRange = (yMax - yMin) || 1;
    var xMax = Math.max.apply(null, series.map(function(s){
      return s.x ? s.x[s.x.length-1] : s.y.length - 1;
    })) || 1;

    function px(xi, yi) {
      return { x: pad.l + (xi / xMax) * cw,
               y: pad.t + ch - ((yi - yMin) / yRange) * ch };
    }

    for (var g = 0; g <= 5; g++) {
      var frac = g / 5, gy = pad.t + frac * ch, gv = yMax - frac * yRange;
      ctx.strokeStyle = '#313244'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.l, gy); ctx.lineTo(W - pad.r, gy); ctx.stroke();
      ctx.fillStyle = '#585b70'; ctx.font = '11px monospace'; ctx.textAlign = 'right';
      ctx.fillText(gv < 0.001 ? gv.toExponential(1) : gv.toFixed(4), pad.l - 6, gy + 4);
    }

    ctx.fillStyle = '#585b70'; ctx.font = '11px monospace'; ctx.textAlign = 'center';
    for (var t = 0; t <= 5; t++) {
      var tx = Math.round(t / 5 * xMax), p = px(tx, yMin);
      ctx.fillText(tx, p.x, H - pad.b + 16);
    }

    series.forEach(function(s, si) {
      var xs = s.x || s.y.map(function(_, i){ return i; });
      ctx.strokeStyle = PALETTE[si % PALETTE.length];
      ctx.lineWidth = s.width || 1.5; ctx.lineJoin = 'round';
      ctx.beginPath();
      xs.forEach(function(xi, i) {
        var p = px(xi, s.y[i]);
        if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
      });
      ctx.stroke();
      if (s.dots) {
        ctx.fillStyle = PALETTE[si % PALETTE.length];
        xs.forEach(function(xi, i) {
          var p = px(xi, s.y[i]);
          ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, 2*Math.PI); ctx.fill();
        });
      }
      ctx.fillStyle = PALETTE[si % PALETTE.length];
      ctx.font = '11px monospace'; ctx.textAlign = 'left';
      ctx.fillText('■ ' + s.label, pad.l + si * 200, pad.t - 6);
    });
  }

  chart('c-step',  [{label: 'train loss / step', y: D.step_losses, width: 1.2}], {});
  var epochSeries = [{label: 'train', y: D.epoch_avg, dots: true, width: 2}];
  if (D.val_avg && D.val_avg.length)
    epochSeries.push({label: 'val', y: D.val_avg, dots: true, width: 2});
  chart('c-epoch', epochSeries, {});
  chart('c-lr',    [{label: 'learning rate', y: D.lr, width: 1.5}], {yMin: 0});

  var samples = D.samples || [];
  var sampleIdx = samples.length - 1;

  function showSample(idx) {
    idx = Math.max(0, Math.min(samples.length - 1, idx));
    sampleIdx = idx;
    var slider = document.getElementById('sample-slider');
    slider.value = idx;
    document.getElementById('btn-prev').disabled = (idx === 0);
    document.getElementById('btn-next').disabled = (idx === samples.length - 1);
    document.getElementById('sample-epoch-label').textContent =
        'EPOCH ' + samples[idx].epoch + ' / ' + (cfg.num_epochs || '?');
    document.getElementById('sample-img').src =
        'data:image/png;base64,' + samples[idx].b64;
  }

  function sampleNav(dir) { showSample(sampleIdx + dir); }
  function sampleSlide(v) { showSample(parseInt(v)); }

  if (samples.length) {
    var section = document.getElementById('sample-section');
    var slider  = document.getElementById('sample-slider');
    section.style.display = 'block';
    slider.max = samples.length - 1;
    showSample(samples.length - 1);
  }

  var refs = D.ref_images || {};
  if (refs.ori || refs.contour || refs.near) {
    document.getElementById('ref-section').style.display = 'block';
    if (refs.ori) {
      document.getElementById('ref-ori-wrap').style.display = 'block';
      document.getElementById('ref-ori').src = 'data:image/png;base64,' + refs.ori;
    }
    if (refs.contour) {
      document.getElementById('ref-contour-wrap').style.display = 'block';
      document.getElementById('ref-contour').src = 'data:image/png;base64,' + refs.contour;
    }
    if (refs.near) {
      document.getElementById('ref-near-wrap').style.display = 'block';
      document.getElementById('ref-near').src = 'data:image/png;base64,' + refs.near;
    }
  }

  // ── lightbox ──────────────────────────────────────────────────────────────
  var lbMode = null; // 'sample' | 'ref'
  var lbRefKey = null;

  function lbOpen(src, label, hasPrev, hasNext) {
    document.getElementById('lb-img').src = src;
    document.getElementById('lb-label').textContent = label;
    document.getElementById('lb-prev').disabled = !hasPrev;
    document.getElementById('lb-next').disabled = !hasNext;
    document.getElementById('lb').classList.add('open');
  }

  function lbClose() {
    document.getElementById('lb').classList.remove('open');
    lbMode = null;
  }

  function lbOpenSample(idx) {
    if (!samples.length) return;
    lbMode = 'sample';
    var s = samples[idx];
    lbOpen('data:image/png;base64,' + s.b64,
           'EPOCH ' + s.epoch + ' / ' + (cfg.num_epochs || '?'),
           idx > 0, idx < samples.length - 1);
  }

  function lbOpenRef(key) {
    lbMode = 'ref';
    lbRefKey = key;
    lbOpen('data:image/png;base64,' + refs[key], key.toUpperCase(), false, false);
  }

  function lbNav(dir) {
    if (lbMode === 'sample') {
      var next = Math.max(0, Math.min(samples.length - 1, sampleIdx + dir));
      showSample(next);
      lbOpenSample(next);
    }
  }

  document.addEventListener('keydown', function(e) {
    var lb = document.getElementById('lb');
    if (!lb.classList.contains('open')) return;
    if (e.key === 'Escape')      lbClose();
    if (e.key === 'ArrowLeft')   lbNav(-1);
    if (e.key === 'ArrowRight')  lbNav(1);
  });

  // auto-refresh pausé quand la lightbox est ouverte
  setInterval(function() {
    if (!document.getElementById('lb').classList.contains('open')) {
      location.reload();
    }
  }, 8000);
  </script>
</body>
</html>
"""


class TrainingMonitor:
    """Réécrit runs/<run>/monitor.html (données inline) à chaque step et fin d'époque."""

    def __init__(self, output_dir, config):
        self.output_dir = output_dir
        self.label      = os.path.basename(os.path.abspath(output_dir))
        self.config = {
            "num_epochs":       config.num_epochs,
            "train_batch_size": config.train_batch_size,
            "img_size":         config.img_size,
            "learning_rate":    config.learning_rate,
            "noise_step":       config.noise_step,
        }
        self.step_losses = []
        self.epoch_avg   = []
        self.val_avg     = []
        self.lr          = []
        self._epoch_buf  = []
        self.epoch       = 0
        self._samples    = []   # list of {epoch, b64}
        self._ref_images = {}   # {ori, contour, near} — stored once
        self._start      = time.time()

        os.makedirs(output_dir, exist_ok=True)
        self._write_html()

    def step(self, loss: float, lr: float):
        self.step_losses.append(round(loss, 6))
        self.lr.append(round(lr, 8))
        self._epoch_buf.append(loss)
        self._write_html()

    def update_references(self, paths: dict):
        """Store reference images once. paths = {ori, contour, near} (values are file paths, missing keys ignored)."""
        if self._ref_images:
            return  # already stored
        for key, path in paths.items():
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    self._ref_images[key] = base64.b64encode(f.read()).decode("ascii")
        self._write_html()

    def update_sample(self, img_path: str, epoch: int):
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        self._samples.append({"epoch": epoch, "b64": b64})
        self._write_html()

    def end_epoch(self, epoch: int, val_avg: float = None):
        self.epoch = epoch
        if self._epoch_buf:
            self.epoch_avg.append(round(sum(self._epoch_buf) / len(self._epoch_buf), 6))
            self._epoch_buf = []
        if val_avg is not None:
            self.val_avg.append(round(val_avg, 6))
        self._write_html()

    def _write_html(self):
        elapsed_s  = time.time() - self._start
        num_epochs = self.config["num_epochs"]
        eta = (str(timedelta(seconds=int(elapsed_s / self.epoch * (num_epochs - self.epoch))))
               if self.epoch > 0 else "—")

        data = {
            "epoch":      self.epoch,
            "step_losses": self.step_losses,
            "epoch_avg":  self.epoch_avg,
            "val_avg":    self.val_avg,
            "lr":         self.lr,
            "config":     self.config,
            "elapsed":    str(timedelta(seconds=int(elapsed_s))),
            "eta":        eta,
            "updated_at": datetime.now().strftime("%H:%M:%S"),
            "samples":    self._samples,
            "ref_images": self._ref_images,
        }

        html = (_RUN_TEMPLATE
                .replace('__RUN_LABEL__', self.label)
                .replace('__DATA__', json.dumps(data)))

        with open(os.path.join(self.output_dir, "monitor.html"), "w", encoding="utf-8") as f:
            f.write(html)

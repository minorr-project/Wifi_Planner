let polling = null;
  let optimPolling = null;
  let activePanel = 'grid';
  let activePreview = 'display';
  let currentStrategy = 'ga';

  // ── Strategy selector ──
  function selectStrategy(s, el) {
    currentStrategy = s;
    document.querySelectorAll('.strategy-option').forEach(o => o.classList.remove('selected'));
    el.classList.add('selected');
  }

  // ── Panel switcher ──
  function switchPanel(panel) {
    activePanel = panel;
    document.getElementById('panelGrid').style.display = panel === 'grid' ? 'block' : 'none';
    document.getElementById('panelResult').style.display = panel === 'result' ? 'block' : 'none';
    document.querySelectorAll('.panel-tab').forEach(t => t.classList.remove('active'));
    const idx = panel === 'grid' ? 0 : 1;
    document.querySelectorAll('.panel-tab')[idx].classList.add('active');
  }

  // ── Grid sub-tabs ──
  function switchPreview(type) {
    activePreview = type;
    const img = document.getElementById('previewImg');
    const label = document.getElementById('previewLabel');
    if (type === 'display') {
      img.src = '/api/preview?ts=' + Date.now();
      label.textContent = 'Display Grid';
    } else {
      img.src = '/api/preview/opt?ts=' + Date.now();
      label.textContent = 'Optimization Grid (4× downsampled)';
    }
    document.querySelectorAll('.sub-tab').forEach((t, i) => {
      t.classList.toggle('active', (i === 0 && type === 'display') || (i === 1 && type === 'opt'));
    });
  }

  // ── File upload ──
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');

  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag-over'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault(); dropzone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) uploadFile(fileInput.files[0]);
  });

  function uploadFile(file) {
    showStatus('info', `Uploading ${file.name}…`);
    const fd = new FormData();
    fd.append('file', file);
    fetch('/api/upload', { method: 'POST', body: fd })
      .then(r => r.json())
      .then(d => {
        if (d.error) { showStatus('error', d.error); return; }
        showStatus('success', `${d.filename} uploaded successfully.`);
        document.getElementById('fileLabel').textContent = d.filename;
        document.getElementById('fileBadge').style.display = 'flex';
        document.getElementById('runBtn').disabled = false;
      })
      .catch(() => showStatus('error', 'Upload failed.'));
  }

  // ── Grid pipeline ──
  function runPipeline() {
    document.getElementById('runBtn').disabled = true;
    showStatus('info', '<span class="spinner"></span> Starting pipeline…');
    document.getElementById('progressWrap').style.display = 'block';
    setProgress('progressBar', 5);

    fetch('/api/run', { method: 'POST' })
      .then(r => r.json())
      .then(d => {
        if (d.error) {
          showStatus('error', d.error);
          document.getElementById('runBtn').disabled = false;
          return;
        }
        startPolling();
      })
      .catch(() => {
        showStatus('error', 'Failed to start pipeline.');
        document.getElementById('runBtn').disabled = false;
      });
  }

  function startPolling() {
    if (polling) clearInterval(polling);
    polling = setInterval(pollStatus, 1500);
  }

  function pollStatus() {
    fetch('/api/status')
      .then(r => r.json())
      .then(data => {
        const p = data.pipeline;
        if (p.error) {
          clearInterval(polling);
          showStatus('error', '❌ ' + p.error);
          document.getElementById('runBtn').disabled = false;
          return;
        }
        if (p.running) {
          showStatus('info', `<span class="spinner"></span> ${p.stage}: ${p.message}`);
          setProgress('progressBar', p.progress || 40);
        }
        if (p.done) {
          clearInterval(polling);
          setProgress('progressBar', 100);
          showStatus('success', '✅ Pipeline complete! You can now run optimization.');
          document.getElementById('runBtn').disabled = false;
          loadGridResults(data);
        }
        if (!p.running && !p.done && !p.error && data.results_ready) {
          clearInterval(polling);
          loadGridResults(data);
        }
      })
      .catch(() => {});
  }

  function loadGridResults(data) {
    document.getElementById('subTabs').classList.remove('hidden');
    const img = document.getElementById('previewImg');
    img.src = '/api/preview?ts=' + Date.now();
    img.style.display = 'block';
    document.getElementById('previewPlaceholder').style.display = 'none';
    document.getElementById('previewLabel').classList.remove('hidden');
    document.getElementById('legendRow').classList.remove('hidden');
    document.getElementById('downloadCard').classList.remove('hidden');
    document.getElementById('optimCard').classList.remove('hidden');
    if (data.meta && Object.keys(data.meta).length) {
      renderMeta(data.meta);
      document.getElementById('metaCard').classList.remove('hidden');
    }
  }

  // ── Optimization ──
  function runOptimization() {
    const strategy = currentStrategy;
    const numRouters = parseInt(document.getElementById('numRouters').value) || 2;

    document.getElementById('optimBtn').disabled = true;
    showOptimStatus('info', '<span class="spinner"></span> Starting optimization…');
    document.getElementById('optimProgressWrap').style.display = 'block';
    setProgress('optimProgressBar', 5);

    fetch('/api/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ strategy, num_routers: numRouters }),
    })
      .then(r => r.json())
      .then(d => {
        if (d.error) {
          showOptimStatus('error', d.error);
          document.getElementById('optimBtn').disabled = false;
          return;
        }
        startOptimPolling(strategy);
      })
      .catch(() => {
        showOptimStatus('error', 'Failed to start optimization.');
        document.getElementById('optimBtn').disabled = false;
      });
  }

  function startOptimPolling(strategy) {
    if (optimPolling) clearInterval(optimPolling);
    optimPolling = setInterval(() => pollOptimStatus(strategy), 1500);
  }

  function pollOptimStatus(strategy) {
    fetch('/api/optimize/status')
      .then(r => r.json())
      .then(data => {
        if (data.error) {
          clearInterval(optimPolling);
          showOptimStatus('error', '❌ ' + data.error);
          document.getElementById('optimBtn').disabled = false;
          return;
        }
        if (data.running) {
          showOptimStatus('info', `<span class="spinner"></span> ${data.stage}: ${data.message}`);
          setProgress('optimProgressBar', data.progress || 50);
        }
        if (data.done && data.result) {
          clearInterval(optimPolling);
          setProgress('optimProgressBar', 100);
          const strat = data.result.strategy;
          const cov = data.result.coverage_percent.toFixed(1);
          const sig = data.result.average_signal_dBm.toFixed(1);
          showOptimStatus('success', `✅ Done · Coverage: ${cov}% · Avg signal: ${sig} dBm`);
          document.getElementById('optimBtn').disabled = false;
          loadOptimResult(data.result);
        }
      })
      .catch(() => {});
  }

  const STRATEGY_NAMES = { ga: 'Genetic Algorithm', random: 'Random', uniform: 'Uniform Grid' };

  function loadOptimResult(result) {
    const strategy = result.strategy;

    document.getElementById('mCoverage').textContent = result.coverage_percent.toFixed(1) + '%';
    document.getElementById('mSignal').textContent = result.average_signal_dBm.toFixed(1) + ' dBm';
    document.getElementById('metricsRow').classList.remove('hidden');

    const img = document.getElementById('resultImg');
    img.src = `/api/optimize/image/${strategy}?ts=` + Date.now();
    img.style.display = 'block';
    document.getElementById('resultPlaceholder').style.display = 'none';
    const lbl = document.getElementById('resultLabel');
    lbl.textContent = STRATEGY_NAMES[strategy] || strategy;
    lbl.classList.remove('hidden');

    document.getElementById('tabResult').classList.remove('hidden');
    switchPanel('result');
  }

  // ── Helpers ──
  function setProgress(id, pct) {
    document.getElementById(id).style.width = pct + '%';
  }

  function showStatus(type, msg) {
    const box = document.getElementById('statusBox');
    box.className = 'status-box ' + type;
    box.innerHTML = msg;
    box.style.display = 'block';
  }

  function showOptimStatus(type, msg) {
    const box = document.getElementById('optimStatusBox');
    box.className = 'status-box ' + type;
    box.innerHTML = msg;
    box.style.display = 'block';
  }

  function renderMeta(meta) {
    const grid = document.getElementById('metaGrid');
    const cs = meta.cell_size ? meta.cell_size + ' units' : '–';
    const shape = meta.crop && meta.crop.cropped_shape
      ? meta.crop.cropped_shape.join(' × ')
      : (meta.grid_shape ? meta.grid_shape.join(' × ') : '–');
    const bounds = meta.bounds;
    let width = '–', height = '–';
    if (bounds) {
      const bmaxx = bounds.maxx !== undefined ? bounds.maxx : bounds.max_x;
      const bminx = bounds.minx !== undefined ? bounds.minx : bounds.min_x;
      const bmaxy = bounds.maxy !== undefined ? bounds.maxy : bounds.max_y;
      const bminy = bounds.miny !== undefined ? bounds.miny : bounds.min_y;
      width = Math.round(bmaxx - bminx);
      height = Math.round(bmaxy - bminy);
    }
    const items = [
      { label: 'Grid Size', value: shape, unit: 'cells' },
      { label: 'Cell Size', value: cs, unit: '' },
      { label: 'Width', value: width, unit: 'units' },
      { label: 'Height', value: height, unit: 'units' },
    ];
    grid.innerHTML = items.map(i => `
      <div class="meta-item">
        <div class="label">${i.label}</div>
        <div class="value">${i.value} <span class="unit">${i.unit}</span></div>
      </div>`).join('');
  }

  // ── Init ──
  fetch('/api/status')
    .then(r => r.json())
    .then(data => {
      if (data.dxf_loaded) {
        document.getElementById('fileLabel').textContent = data.dxf_filename;
        document.getElementById('fileBadge').style.display = 'flex';
        document.getElementById('runBtn').disabled = false;
      }
      if (data.results_ready) {
        loadGridResults(data);
        document.getElementById('progressWrap').style.display = 'none';
      }
    });
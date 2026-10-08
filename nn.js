/* ============================================================
   nn.js · 인공신경망 탭 (회귀 분석기에 덧붙이는 파일)
   - 라이브러리 없이 신경망(순전파, 역전파, Adam)을 직접 구현
   - index.html 에는 <script src="nn.js"></script> 한 줄만 추가하면 된다
   ============================================================ */
(function (root) {
'use strict';

/* ==========================================================
   1. 수학 도구
   ========================================================== */
function mulberry32(seed) {                 // 시드를 주면 항상 같은 난수가 나온다 (결과 재현용)
  let a = seed | 0;
  return function () {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randn(rng) {                       // 표준정규분포 난수 (Box-Muller)
  let u = 0;
  while (u === 0) u = rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function shuffle(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
  }
  return arr;
}
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const stdev = (a, m) => Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length) || 1;
function median(a) {
  const s = [...a].sort((x, y) => x - y), n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}
function r2score(y, yp) {
  const m = mean(y);
  let ssr = 0, sst = 0;
  for (let i = 0; i < y.length; i++) { ssr += (y[i] - yp[i]) ** 2; sst += (y[i] - m) ** 2; }
  return sst === 0 ? 0 : 1 - ssr / sst;
}
function rmse(y, yp) {
  let s = 0;
  for (let i = 0; i < y.length; i++) s += (y[i] - yp[i]) ** 2;
  return Math.sqrt(s / y.length);
}
function corr(a, b) {
  const ma = mean(a), mb = mean(b);
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) {
    sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2;
  }
  return sab / (Math.sqrt(saa * sbb) || 1);
}

/* 최소제곱법 : (FᵀF)θ = Fᵀy 를 가우스 소거법으로 푼다 (회귀 탭과 같은 방법) */
function lstsq(F, y) {
  const k = F[0].length;
  const A = Array.from({ length: k }, () => Array(k).fill(0));
  const b = Array(k).fill(0);
  for (let r = 0; r < F.length; r++) {
    for (let p = 0; p < k; p++) {
      b[p] += F[r][p] * y[r];
      for (let q = 0; q < k; q++) A[p][q] += F[r][p] * F[r][q];
    }
  }
  for (let p = 0; p < k; p++) A[p][p] += 1e-9;           // 수치 안정용
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < k; c++) {
    let piv = c;
    for (let r = c + 1; r < k; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    const t = M[c]; M[c] = M[piv]; M[piv] = t;
    for (let r = 0; r < k; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let q = c; q <= k; q++) M[r][q] -= f * M[c][q];
    }
  }
  return M.map((row, i) => row[k] / row[i]);
}

/* ==========================================================
   2. 신경망 (다층 퍼셉트론)
      입력층 → 은닉층 … → 출력층
      각 노드 : 입력 × 가중치(weight) 를 모두 더하고 편향(bias)을 더한 뒤
               활성화 함수에 넣어서 비선형으로 만든다.
      출력층은 값을 그대로 내보낸다 (회귀 문제이므로).
   ========================================================== */
const ACT = {
  relu:    { label: 'ReLU',      f: x => (x > 0 ? x : 0),         d: (z, a) => (z > 0 ? 1 : 0) },
  sigmoid: { label: '시그모이드', f: x => 1 / (1 + Math.exp(-x)),  d: (z, a) => a * (1 - a) },
  tanh:    { label: 'tanh',      f: Math.tanh,                    d: (z, a) => 1 - a * a },
};

class MLP {
  constructor(sizes, actName, seed) {
    this.sizes = sizes;
    this.actName = actName;
    this.act = ACT[actName];
    this.L = sizes.length - 1;                       // 가중치가 있는 층의 개수
    const rng = mulberry32(seed);
    this.W = []; this.b = []; this.gW = []; this.gb = [];
    this.mW = []; this.vW = []; this.mb = []; this.vb = [];
    for (let l = 1; l <= this.L; l++) {
      const nin = sizes[l - 1], nout = sizes[l];
      // 처음 가중치는 작은 난수로 시작한다. (ReLU 는 He 방식, 나머지는 Xavier 방식)
      const std = (l < this.L && actName === 'relu') ? Math.sqrt(2 / nin) : Math.sqrt(1 / nin);
      const W = new Float64Array(nin * nout);
      for (let i = 0; i < W.length; i++) W[i] = randn(rng) * std;
      this.W[l] = W;
      this.b[l] = new Float64Array(nout);
      this.gW[l] = new Float64Array(W.length);
      this.gb[l] = new Float64Array(nout);
      this.mW[l] = new Float64Array(W.length); this.vW[l] = new Float64Array(W.length);
      this.mb[l] = new Float64Array(nout);     this.vb[l] = new Float64Array(nout);
    }
    this.t = 0;                                       // Adam 의 걸음 수
    this.a = sizes.map(n => new Float64Array(n));     // 각 층의 출력값
    this.z = sizes.map(n => new Float64Array(n));     // 활성화 함수에 넣기 전 값
    this.delta = sizes.map(n => new Float64Array(n)); // 역전파로 내려오는 오차
  }

  /* 순전파 : 입력 → 출력 */
  forward(x) {
    const { a, z, W, b, sizes, L, act } = this;
    a[0].set(x);
    for (let l = 1; l <= L; l++) {
      const nin = sizes[l - 1], nout = sizes[l], Wl = W[l], bl = b[l], prev = a[l - 1];
      for (let o = 0; o < nout; o++) {
        let s = bl[o];
        const base = o * nin;
        for (let i = 0; i < nin; i++) s += Wl[base + i] * prev[i];
        z[l][o] = s;
        a[l][o] = (l === L) ? s : act.f(s);
      }
    }
    return a[L][0];
  }

  /* 한 번의 학습 : 미니배치의 평균 기울기를 구해(역전파) Adam 으로 가중치를 고친다 */
  step(X, Y, idx, lr) {
    const { sizes, L, W, gW, gb, a, z, delta, act } = this;
    for (let l = 1; l <= L; l++) { gW[l].fill(0); gb[l].fill(0); }
    const B = idx.length;
    let loss = 0;
    for (let n = 0; n < B; n++) {
      const k = idx[n];
      const err = this.forward(X[k]) - Y[k];
      loss += err * err;
      delta[L][0] = (2 * err) / B;                    // 오차(MSE)를 출력값으로 미분
      for (let l = L; l >= 1; l--) {                  // 출력층에서 입력층 쪽으로 거꾸로
        const nin = sizes[l - 1], nout = sizes[l];
        const dl = delta[l], prev = a[l - 1], gWl = gW[l], gbl = gb[l];
        for (let o = 0; o < nout; o++) {
          const d = dl[o];
          gbl[o] += d;
          const base = o * nin;
          for (let i = 0; i < nin; i++) gWl[base + i] += d * prev[i];
        }
        if (l > 1) {                                  // 합성함수 미분법(체인 룰)으로 한 층 아래로 전달
          const Wl = W[l], dprev = delta[l - 1];
          for (let i = 0; i < nin; i++) {
            let s = 0;
            for (let o = 0; o < nout; o++) s += Wl[o * nin + i] * dl[o];
            dprev[i] = s * act.d(z[l - 1][i], a[l - 1][i]);
          }
        }
      }
    }
    // Adam : 기울기의 평균(m)과 크기(v)를 보고 걸음 크기를 알아서 조절하는 경사하강법
    this.t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    for (let l = 1; l <= L; l++) {
      const Wl = W[l], g = gW[l], m = this.mW[l], v = this.vW[l];
      for (let i = 0; i < Wl.length; i++) {
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        Wl[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps);
      }
      const bl = this.b[l], gbl = gb[l], mb = this.mb[l], vb = this.vb[l];
      for (let i = 0; i < bl.length; i++) {
        mb[i] = b1 * mb[i] + (1 - b1) * gbl[i];
        vb[i] = b2 * vb[i] + (1 - b2) * gbl[i] * gbl[i];
        bl[i] -= lr * (mb[i] / c1) / (Math.sqrt(vb[i] / c2) + eps);
      }
    }
    return loss / B;
  }
}

/* ==========================================================
   3. 학습 준비 : 데이터 나누기, 표준화, 비교용 회귀
   ========================================================== */
/* 학습용 / 시험용으로 나눈다. mode 'group' 이면 같은 그룹(같은 선체)이 한쪽에만 들어간다. */
function makeSplit(n, keys, mode, seed, frac) {
  const rng = mulberry32(seed);
  const tr = [], te = [];
  if (mode === 'group' && keys) {
    const counts = {};
    keys.forEach(k => { counts[k] = (counts[k] || 0) + 1; });
    const groups = shuffle(Object.keys(counts), rng);
    const testSet = new Set();
    let cnt = 0;
    const target = Math.round(n * frac);
    for (const g of groups) { if (cnt >= target) break; testSet.add(g); cnt += counts[g]; }
    for (let i = 0; i < n; i++) (testSet.has(keys[i]) ? te : tr).push(i);
  } else {
    const idx = shuffle(Array.from({ length: n }, (_, i) => i), rng);
    const nte = Math.round(n * frac);
    for (let i = 0; i < n; i++) (i < nte ? te : tr).push(idx[i]);
  }
  return { tr, te };
}

class Trainer {
  /* cfg : { X(원본 입력 행렬), y(원본 출력), keys, hidden:[노드수…], act, lr, epochs, seed, split } */
  constructor(cfg) {
    this.cfg = cfg;
    const n = cfg.X.length, d = cfg.X[0].length;
    this.n = n; this.d = d;
    const sp = makeSplit(n, cfg.keys, cfg.split, cfg.seed, 0.2);
    this.tr = sp.tr; this.te = sp.te;

    // 표준화 : 평균을 0, 표준편차를 1로 맞춘다 (학습용 데이터의 평균과 표준편차만 사용)
    this.xm = []; this.xs = [];
    for (let j = 0; j < d; j++) {
      const col = this.tr.map(i => cfg.X[i][j]);
      const m = mean(col);
      this.xm.push(m); this.xs.push(stdev(col, m));
    }
    const ytr = this.tr.map(i => cfg.y[i]);
    this.ym = mean(ytr); this.ys = stdev(ytr, this.ym);
    this.Xn = cfg.X.map(r => r.map((v, j) => (v - this.xm[j]) / this.xs[j]));
    this.Yn = cfg.y.map(v => (v - this.ym) / this.ys);

    this.net = new MLP([d, ...cfg.hidden, 1], cfg.act, cfg.seed);
    this.rng = mulberry32(cfg.seed + 1);
    this.epoch = 0;
    this.hist = { epoch: [], train: [], test: [] };   // 오차(MSE, 원래 단위)
    this.batch = 32;
    this.buildBaselines();
  }

  normRow(row) { return row.map((v, j) => (v - this.xm[j]) / this.xs[j]); }

  predictRaw(row) { return this.net.forward(this.normRow(row)) * this.ys + this.ym; }

  mse(idx) {
    let s = 0;
    for (const i of idx) { const e = (this.net.forward(this.Xn[i]) - this.Yn[i]); s += e * e; }
    return (s / idx.length) * this.ys * this.ys;
  }

  runEpochs(k) {
    const order = this.tr.slice();
    for (let e = 0; e < k && this.epoch < this.cfg.epochs; e++) {
      shuffle(order, this.rng);
      for (let s = 0; s < order.length; s += this.batch) {
        this.net.step(this.Xn, this.Yn, order.slice(s, s + this.batch), this.cfg.lr);
      }
      this.epoch++;
      this.hist.epoch.push(this.epoch);
      this.hist.train.push(this.mse(this.tr));
      this.hist.test.push(this.mse(this.te));
    }
  }
  get finished() { return this.epoch >= this.cfg.epochs; }

  /* 학습용 / 시험용 점수 */
  evaluate() {
    const out = {};
    for (const [name, idx] of [['train', this.tr], ['test', this.te]]) {
      const act = idx.map(i => this.cfg.y[i]);
      const pred = idx.map(i => this.net.forward(this.Xn[i]) * this.ys + this.ym);
      out[name] = { r2: r2score(act, pred), rmse: rmse(act, pred), act, pred };
    }
    return out;
  }

  /* 비교용 모델 3개 : 직선(입력 1개), 곡선(입력 1개), 선형회귀(입력 전체) */
  buildBaselines() {
    const { cfg, d } = this;
    const ytr = this.tr.map(i => cfg.y[i]);
    // 1) 입력 전체를 쓰는 선형회귀
    const Fm = idx => idx.map(i => [1, ...this.Xn[i]]);
    const cMulti = lstsq(Fm(this.tr), ytr);
    const multi = {
      label: `선형 회귀 (입력 ${d}개 전부)`,
      predictNorm: xn => cMulti[0] + xn.reduce((s, v, j) => s + cMulti[j + 1] * v, 0),
    };
    // 2) 출력과 가장 관련이 큰 입력 1개로 직선, 3차 곡선 (회귀 탭과 같은 방식)
    let best = 0, bc = -1;
    for (let j = 0; j < d; j++) {
      const c = Math.abs(corr(this.tr.map(i => this.Xn[i][j]), ytr));
      if (c > bc) { bc = c; best = j; }
    }
    this.bestVar = best;
    const polyModel = deg => {
      const cols = [];
      for (let k = 1; k <= deg; k++) {
        const raw = this.tr.map(i => Math.pow(this.Xn[i][best], k));
        const m = mean(raw);
        cols.push({ m, s: stdev(raw, m) });
      }
      const feat = xv => [1, ...cols.map((c, k) => (Math.pow(xv, k + 1) - c.m) / c.s)];
      const th = lstsq(this.tr.map(i => feat(this.Xn[i][best])), ytr);
      return xn => feat(xn[best]).reduce((s, v, k) => s + v * th[k], 0);
    };
    const lin1 = { label: `직선 (입력 1개)`, predictNorm: polyModel(1) };
    const cub1 = { label: `3차 곡선 (입력 1개)`, predictNorm: polyModel(3) };
    this.baselines = [lin1, cub1, multi];
    for (const m of this.baselines) {
      m.predictRaw = row => m.predictNorm(this.normRow(row));
      for (const [name, idx] of [['train', this.tr], ['test', this.te]]) {
        const act = idx.map(i => cfg.y[i]);
        const pred = idx.map(i => m.predictNorm(this.Xn[i]));
        m[name] = { r2: r2score(act, pred), rmse: rmse(act, pred) };
      }
    }
  }
}

/* ==========================================================
   4. 데이터 읽기 (회귀 탭의 함수가 있으면 그걸 쓰고, 없으면 간단히 직접)
   ========================================================== */
const YACHT_COLS = ['부심위치(LCB)', '주형계수(Cp)', '길이-배수량비', '폭-흘수비', '길이-폭비',
                    '배의 속도(Froude)', '물의 저항(Resistance)'];
const isNum = v => v !== '' && v !== null && v !== undefined && !isNaN(Number(v));

function parseTextTable(text) {
  const lines = text.split(/\r?\n/).filter(l => l.trim());
  if (!lines.length) throw new Error('파일에 데이터가 없어요.');
  const delim = /[,;\t]/.test(lines[0]) ? /[,;\t]/ : /\s+/;
  const raw = lines.map(l => l.trim().split(delim).map(s => s.trim()));
  const hasHeader = !raw[0].every(isNum);
  let columns = hasHeader ? raw[0] : raw[0].map((_, i) => `${i + 1}번 열`);
  if (!hasHeader && raw[0].length === 7) columns = YACHT_COLS.slice();
  const rows = (hasHeader ? raw.slice(1) : raw).map(r => columns.map((_, i) => r[i]));
  return { columns, rows };
}

/* 숫자 열만 골라낸다 */
function numericColumns(t) {
  return t.columns.map((_, i) => i).filter(i => {
    const vals = t.rows.map(r => r[i]).filter(v => String(v === undefined || v === null ? '' : v).trim() !== '');
    return vals.length >= 3 && vals.filter(isNum).length / vals.length > 0.9;
  });
}

/* ==========================================================
   5. 화면 (브라우저에서만 실행)
   ========================================================== */
function initUI() {
  const $ = id => document.getElementById(id);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const fmt = v => {
    if (!isFinite(v)) return '-';
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(3);
    return v.toLocaleString('ko-KR', { maximumFractionDigits: a < 1 ? 4 : a < 100 ? 3 : 1 });
  };

  const tabs = Array.from(document.querySelectorAll('header nav button'));
  const regMain = document.querySelector('main');
  const headerP = document.querySelector('header p');
  if (tabs.length < 2 || !regMain) return;                // 회귀 분석기 페이지가 아니면 아무것도 안 한다

  /* ── 스타일 (회귀 탭에 있는 클래스를 최대한 그대로 쓴다) ── */
  const st = document.createElement('style');
  st.textContent = `
    .chk { display:flex; align-items:center; gap:8px; margin:4px 0; color:var(--ink); font-size:.85rem; cursor:pointer; }
    .chk input { width:auto; accent-color:var(--signal); }
    .nn-svg { width:100%; min-width:600px; height:auto; display:block; }
    .nn-svg text { fill:var(--ink); font-size:11px; font-family:inherit; }
    .nn-svg .dim { fill:var(--muted); }
    .nn-grid4 { display:grid; grid-template-columns:repeat(2,1fr); gap:12px; margin-top:14px; }
    .nn-grid4 div { background:var(--paper); border-radius:6px; padding:10px 12px; }
    .nn-grid4 b { display:block; font-size:1.3rem; font-variant-numeric:tabular-nums; }
    .nn-grid4 .nn b { color:var(--signal); }
    .nn-inputs { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:10px; }
    tr.nnrow td { font-weight:700; color:var(--signal); }
    .nn-prog { font-size:.85rem; color:var(--muted); margin:8px 0 0; font-variant-numeric:tabular-nums; }
    .btn:disabled { opacity:.5; cursor:not-allowed; }
  `;
  document.head.appendChild(st);

  /* ── 인공신경망 탭 화면 ── */
  const nnMain = document.createElement('main');
  nnMain.id = 'nnMain';
  nnMain.style.display = 'none';
  nnMain.innerHTML = `
  <aside>
    <div class="box step">
      <h2><span>1</span>데이터 열기</h2>
      <button class="btn" id="nnSample">예제 데이터 : 배 속도와 저항</button>
      <label for="nnFile" class="btn ghost" style="margin-top:8px;cursor:pointer">다른 파일 열기</label>
      <input type="file" id="nnFile" accept=".csv,.xlsx,.xls,.data,.txt">
      <p class="hint">회귀 탭에서 연 데이터가 있으면 자동으로 가져와요.</p>
      <p class="status" id="nnStatus" role="status"></p>
    </div>

    <div class="box step">
      <h2><span>2</span>입력과 출력 고르기</h2>
      <label for="nnY">맞히고 싶은 값 (출력)</label>
      <select id="nnY" disabled></select>
      <label>알려주는 값 (입력)</label>
      <div id="nnXs"></div>
    </div>

    <div class="box step">
      <h2><span>3</span>신경망 모양 고르기</h2>
      <label>은닉층 개수</label>
      <div class="radio" id="nnLayers">
        <label><input type="radio" name="nnlay" value="1"><span>1층</span></label>
        <label><input type="radio" name="nnlay" value="2" checked><span>2층</span></label>
        <label><input type="radio" name="nnlay" value="3"><span>3층</span></label>
      </div>
      <label for="nnNodes">층마다 노드 개수</label>
      <select id="nnNodes">
        <option value="4">4개</option><option value="8">8개</option>
        <option value="16" selected>16개</option><option value="32">32개</option>
      </select>
      <label>활성화 함수</label>
      <div class="radio" id="nnAct">
        <label><input type="radio" name="nnact" value="relu" checked><span>ReLU</span></label>
        <label><input type="radio" name="nnact" value="tanh"><span>tanh</span></label>
        <label><input type="radio" name="nnact" value="sigmoid"><span>시그모이드</span></label>
      </div>
      <p class="hint">활성화 함수가 직선을 곡선으로 바꿔 줘요. 은닉층이 많고 노드가 많을수록 복잡한 모양도 그릴 수 있지만, 데이터에만 억지로 맞출(과적합) 수도 있어요.</p>
    </div>

    <div class="box step">
      <h2><span>4</span>학습시키기</h2>
      <label for="nnEpochs">학습 횟수 (데이터를 몇 바퀴 볼지)</label>
      <select id="nnEpochs">
        <option value="200">200번</option><option value="500">500번</option>
        <option value="1000" selected>1000번</option><option value="2000">2000번</option>
        <option value="3000">3000번</option>
      </select>
      <label for="nnLr">학습 속도 (한 걸음의 크기)</label>
      <select id="nnLr">
        <option value="0.001">0.001 (느리고 안정적)</option><option value="0.003">0.003</option>
        <option value="0.01" selected>0.01</option><option value="0.03">0.03 (빠르지만 흔들림)</option>
      </select>
      <div id="nnSplitWrap" hidden>
        <label for="nnSplit">시험 데이터 고르는 방법</label>
        <select id="nnSplit">
          <option value="random">무작위 20% 떼어두기</option>
          <option value="group">선체 단위로 떼어두기 (처음 보는 선체 맞히기)</option>
        </select>
      </div>
      <button class="btn" id="nnTrain" style="margin-top:12px" disabled>학습 시작</button>
      <button class="btn ghost" id="nnStop" disabled>멈추기</button>
    </div>
  </aside>

  <section class="results">
    <div class="box empty" id="nnEmpty">
      <strong>데이터를 열어주세요</strong>
      왼쪽에서 파일을 선택하거나, 예제 데이터(배 속도와 저항)를 눌러보세요.
    </div>

    <div id="nnOut" hidden>
      <div class="box">
        <h3>신경망 구조</h3>
        <p class="sub">동그라미가 노드, 선이 가중치예요. 주황색은 +, 파란색은 − 가중치이고 진할수록 영향이 커요. 출력과 상관없는 입력은 학습하면서 가중치가 0에 가까워지는 경향이 있어요.</p>
        <div class="scroll"><svg class="nn-svg" id="nnSvg" viewBox="0 0 720 330" role="img" aria-label="신경망 구조 그림"></svg></div>
      </div>

      <div class="box" style="margin-top:20px">
        <h3>학습 곡선</h3>
        <p class="sub">학습할수록 오차(MSE)가 줄어드는 모습이에요. 시험 오차가 오히려 올라가면 과적합이에요.</p>
        <div class="chart-wrap small"><canvas id="nnLoss" aria-label="학습 곡선"></canvas></div>
        <p class="nn-prog" id="nnProg"></p>
      </div>

      <div class="box" style="margin-top:20px">
        <h3>모델 비교</h3>
        <p class="sub" id="nnCmpSub">R²는 1에 가까울수록, RMSE는 0에 가까울수록 잘 맞아요. 시험 점수는 학습에 쓰지 않은 데이터로 매긴 점수예요.</p>
        <div class="scroll"><table id="nnTable"></table></div>
        <p class="check" id="nnNote"></p>
      </div>

      <div class="two" style="margin-top:20px">
        <div class="box">
          <h3>예측 vs 실제 (시험 데이터)</h3>
          <p class="sub">점이 대각선에 가까울수록 잘 맞춘 거예요.</p>
          <div class="chart-wrap small"><canvas id="nnScatter" aria-label="예측값과 실제값 비교"></canvas></div>
        </div>
        <div class="box">
          <h3>모델별 시험 R²</h3>
          <p class="sub">같은 시험 데이터로 매긴 점수를 비교해요.</p>
          <div class="chart-wrap small"><canvas id="nnBar" aria-label="모델별 R² 비교"></canvas></div>
        </div>
      </div>

      <div class="box" style="margin-top:20px">
        <h3>예측 계산기</h3>
        <p class="sub">입력 값을 바꾸면 네 모델이 각각 출력 값을 예측해요.</p>
        <div class="nn-inputs" id="nnPredIn"></div>
        <div class="nn-grid4" id="nnPredOut"></div>
        <p class="warn" id="nnWarn"></p>
      </div>
    </div>
  </section>`;
  regMain.parentNode.insertBefore(nnMain, regMain.nextSibling);

  /* ── 상태 ── */
  let data = null;            // { columns, rows, label, numericCols, isYacht }
  let trainer = null, running = false, raf = 0;
  let charts = {};
  const SEED = 42;

  const setStatus = (msg, kind) => { const s = $('nnStatus'); s.textContent = msg; s.className = 'status ' + (kind || ''); };
  const radioVal = name => document.querySelector(`input[name="${name}"]:checked`).value;

  /* ── 탭 전환 ── */
  const origP = headerP ? headerP.textContent : '';
  tabs[1].disabled = false;
  tabs[1].innerHTML = '인공신경망';
  function setTab(i) {
    tabs.forEach((b, k) => b.setAttribute('aria-selected', k === i ? 'true' : 'false'));
    regMain.style.display = i === 0 ? '' : 'none';
    nnMain.style.display = i === 1 ? '' : 'none';
    if (headerP) headerP.textContent = i === 1
      ? '데이터를 열고 입력과 출력을 고르면, 인공신경망이 스스로 규칙을 배워서 예측해요.' : origP;
    if (i === 1 && !data) adoptRegressionTable();
    if (i === 1 && data) resizeCharts();
  }
  tabs[0].addEventListener('click', () => setTab(0));
  tabs[1].addEventListener('click', () => setTab(1));

  function resizeCharts() { Object.values(charts).forEach(c => c && c.resize()); }

  /* 회귀 탭에서 이미 연 데이터가 있으면 가져온다 */
  function adoptRegressionTable() {
    try {
      if (typeof table !== 'undefined' && table && table.rows && table.columns) {
        loadData({ columns: table.columns, rows: table.rows }, table.label || '회귀 탭의 데이터');
      }
    } catch (e) { /* 회귀 탭 데이터가 없으면 그냥 넘어간다 */ }
  }

  /* ── 데이터 불러오기 ── */
  function loadData(t, label) {
    const numericCols = numericColumns(t);
    if (numericCols.length < 2) throw new Error('숫자로 된 열이 2개 이상 있어야 해요.');
    const isYacht = t.columns.length === 7 && t.rows.length === 308;
    data = { ...t, label, numericCols, isYacht };
    stopTraining();
    trainer = null;
    $('nnY').innerHTML = numericCols.map(i => `<option value="${i}">${esc(t.columns[i])}</option>`).join('');
    $('nnY').disabled = false;
    $('nnY').value = String(numericCols[numericCols.length - 1]);
    buildXChecks();
    $('nnSplitWrap').hidden = !isYacht;
    $('nnTrain').disabled = false;
    $('nnEmpty').hidden = true;
    $('nnOut').hidden = false;
    setStatus(`${label} 을(를) 열었어요 (${t.rows.length.toLocaleString()}행).`, 'ok');
    ensureCharts();
    resetViews();
  }

  function buildXChecks() {
    const y = +$('nnY').value;
    $('nnXs').innerHTML = data.numericCols.filter(i => i !== y).map(i =>
      `<label class="chk"><input type="checkbox" value="${i}" checked> ${esc(data.columns[i])}</label>`).join('');
  }

  function selected() {
    const y = +$('nnY').value;
    const xs = Array.from(document.querySelectorAll('#nnXs input:checked')).map(c => +c.value);
    return { y, xs };
  }

  /* 선택한 열로 숫자 표를 만든다 */
  function buildMatrix() {
    const { y, xs } = selected();
    if (!xs.length) throw new Error('입력을 1개 이상 골라주세요.');
    const cols = [...xs, y];
    const X = [], Y = [], keys = [];
    data.rows.forEach(r => {
      if (cols.every(c => isNum(r[c]))) {
        X.push(xs.map(c => Number(r[c])));
        Y.push(Number(r[y]));
        keys.push(r.slice(0, 5).join(','));          // 예제 데이터에서 같은 선체를 묶는 열쇠
      }
    });
    if (X.length < 30) throw new Error('쓸 수 있는 행이 너무 적어요 (최소 30행).');
    return { X, y: Y, keys, xs, yIdx: y };
  }

  /* ── 그래프 ── */
  function ensureCharts() {
    if (charts.loss) return;
    const ink = cssVar('--ink'), muted = cssVar('--muted'), grid = cssVar('--line');
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.color = muted;
    const axis = title => ({ title: { display: true, text: title, color: ink }, grid: { color: grid } });
    charts.loss = new Chart($('nnLoss'), {
      type: 'line',
      data: { datasets: [
        { label: '학습 오차', data: [], borderColor: cssVar('--steel'), pointRadius: 0, borderWidth: 2 },
        { label: '시험 오차', data: [], borderColor: cssVar('--signal'), pointRadius: 0, borderWidth: 2 },
      ] },
      options: { maintainAspectRatio: false, animation: false, parsing: false,
        scales: { x: { type: 'linear', ...axis('학습 횟수') }, y: { type: 'logarithmic', ...axis('오차 (MSE)') } },
        plugins: { legend: { labels: { color: ink } } } },
    });
    charts.scatter = new Chart($('nnScatter'), {
      type: 'scatter',
      data: { datasets: [
        { label: '시험 데이터', data: [], backgroundColor: cssVar('--water') + '99', pointRadius: 3.5, order: 2 },
        { type: 'line', label: '완벽한 예측', data: [], borderColor: cssVar('--signal'), borderWidth: 2, pointRadius: 0, order: 1 },
      ] },
      options: { maintainAspectRatio: false, animation: false, parsing: false,
        scales: { x: { type: 'linear', ...axis('실제 값') }, y: { type: 'linear', ...axis('예측 값') } },
        plugins: { legend: { labels: { color: ink, usePointStyle: true } } } },
    });
    charts.bar = new Chart($('nnBar'), {
      type: 'bar',
      data: { labels: [], datasets: [{ label: '시험 R²', data: [], backgroundColor: [] }] },
      options: { maintainAspectRatio: false, animation: false,
        scales: { y: { ...axis('R²'), suggestedMax: 1 }, x: { grid: { display: false } } },
        plugins: { legend: { display: false } } },
    });
  }

  function resetViews() {
    ['loss', 'scatter', 'bar'].forEach(k => {
      const c = charts[k]; if (!c) return;
      c.data.datasets.forEach(d => { d.data = []; });
      if (k === 'bar') c.data.labels = [];
      c.update('none');
    });
    $('nnTable').innerHTML = '';
    $('nnProg').textContent = '';
    $('nnNote').textContent = '';
    $('nnPredIn').innerHTML = '';
    $('nnPredOut').innerHTML = '';
    $('nnWarn').textContent = '';
    drawNet(null);
  }

  /* 신경망 구조 그림 */
  function drawNet(tr) {
    let sizes;
    let names = [];
    if (tr) { sizes = tr.net.sizes; names = selected().xs.map(i => data.columns[i]); }
    else {
      const { xs } = selected();
      sizes = [xs.length || 1, ...Array(+radioVal('nnlay')).fill(+$('nnNodes').value), 1];
      names = xs.map(i => data.columns[i]);
    }
    const W = 720, H = 330, padL = 150, padR = 130, padT = 30, padB = 56;
    const shown = sizes.map(n => Math.min(n, 10));
    const xp = l => padL + l * (W - padL - padR) / (sizes.length - 1);
    const yp = (l, i) => { const m = shown[l], h = H - padT - padB; return padT + (m === 1 ? h / 2 : i * h / (m - 1)); };
    let svg = '';
    for (let l = 1; l < sizes.length; l++) {
      let maxAbs = 1e-9;
      if (tr) for (let o = 0; o < shown[l]; o++) for (let i = 0; i < shown[l - 1]; i++)
        maxAbs = Math.max(maxAbs, Math.abs(tr.net.W[l][o * sizes[l - 1] + i]));
      for (let o = 0; o < shown[l]; o++) for (let i = 0; i < shown[l - 1]; i++) {
        let color = cssVar('--steel'), op = 0.25, sw = 1;
        if (tr) {
          const w = tr.net.W[l][o * sizes[l - 1] + i];
          color = w >= 0 ? cssVar('--signal') : cssVar('--water');
          op = Math.min(1, 0.06 + 0.94 * Math.abs(w) / maxAbs);
          sw = 0.6 + 1.6 * Math.abs(w) / maxAbs;
        }
        svg += `<line x1="${xp(l - 1)}" y1="${yp(l - 1, i)}" x2="${xp(l)}" y2="${yp(l, o)}" stroke="${color}" stroke-opacity="${op.toFixed(2)}" stroke-width="${sw.toFixed(2)}"/>`;
      }
    }
    for (let l = 0; l < sizes.length; l++) {
      for (let i = 0; i < shown[l]; i++)
        svg += `<circle cx="${xp(l)}" cy="${yp(l, i)}" r="7" fill="${cssVar('--panel')}" stroke="${cssVar('--ink')}" stroke-width="1.5"/>`;
      if (sizes[l] > shown[l])
        svg += `<text class="dim" x="${xp(l)}" y="${H - padB + 26}" text-anchor="middle">… 외 ${sizes[l] - shown[l]}개</text>`;
      const title = l === 0 ? `입력층 (${sizes[l]})` : l === sizes.length - 1 ? `출력층 (${sizes[l]})` : `은닉층 ${l} (${sizes[l]})`;
      svg += `<text x="${xp(l)}" y="${H - 8}" text-anchor="middle" style="font-weight:700">${title}</text>`;
    }
    for (let i = 0; i < shown[0]; i++)
      svg += `<text x="${xp(0) - 14}" y="${yp(0, i) + 4}" text-anchor="end">${esc(names[i] || '')}</text>`;
    const yName = data ? data.columns[+$('nnY').value] : '';
    svg += `<text x="${xp(sizes.length - 1) + 14}" y="${yp(sizes.length - 1, 0) + 4}">${esc(yName.length > 11 ? yName.slice(0, 11) + '…' : yName)}</text>`;
    $('nnSvg').innerHTML = svg;
  }

  /* ── 화면 갱신 (학습 도중 계속 호출) ── */
  function refresh(final) {
    const ev = trainer.evaluate();
    const h = trainer.hist;
    const pts = a => h.epoch.map((e, i) => ({ x: e, y: Math.max(a[i], 1e-9) }));
    charts.loss.data.datasets[0].data = pts(h.train);
    charts.loss.data.datasets[1].data = pts(h.test);
    charts.loss.update('none');

    const lo = Math.min(...ev.test.act, ...ev.test.pred), hi = Math.max(...ev.test.act, ...ev.test.pred);
    charts.scatter.data.datasets[0].data = ev.test.act.map((v, i) => ({ x: v, y: ev.test.pred[i] }));
    charts.scatter.data.datasets[1].data = [{ x: lo, y: lo }, { x: hi, y: hi }];
    charts.scatter.update('none');

    const rows = trainer.baselines.map(m => ({ label: m.label, tr: m.train, te: m.test, nn: false }));
    rows.push({ label: `인공신경망 (${trainer.net.sizes.slice(1, -1).join('-')} · ${ACT[trainer.cfg.act].label})`,
                tr: ev.train, te: ev.test, nn: true });
    $('nnTable').innerHTML =
      '<thead><tr><th>모델</th><th>학습 R²</th><th>시험 R²</th><th>시험 RMSE</th></tr></thead><tbody>' +
      rows.map(r => `<tr class="${r.nn ? 'nnrow' : ''}"><td>${esc(r.label)}</td><td>${r.tr.r2.toFixed(3)}</td><td>${r.te.r2.toFixed(3)}</td><td>${fmt(r.te.rmse)}</td></tr>`).join('') +
      '</tbody>';

    charts.bar.data.labels = rows.map(r => r.label.replace(/ \(.*$/, '').replace('선형 회귀', '선형(전체)'));
    charts.bar.data.datasets[0].data = rows.map(r => +r.te.r2.toFixed(4));
    charts.bar.data.datasets[0].backgroundColor = rows.map(r => r.nn ? cssVar('--signal') : cssVar('--steel'));
    charts.bar.update('none');

    const bv = data.columns[selected().xs[trainer.bestVar]];
    $('nnProg').textContent =
      `${final ? '학습 끝' : '학습 중'} · ${trainer.epoch.toLocaleString()} / ${trainer.cfg.epochs.toLocaleString()}번 · ` +
      `학습 오차 ${fmt(h.train[h.train.length - 1])} · 시험 오차 ${fmt(h.test[h.test.length - 1])}`;
    $('nnNote').textContent =
      `직선과 3차 곡선은 출력과 가장 관련이 큰 입력 1개(${bv})만 써요. ` +
      `학습 ${trainer.tr.length}행 / 시험 ${trainer.te.length}행` +
      (trainer.cfg.split === 'group' ? ' · 시험 데이터는 학습에 한 번도 나오지 않은 선체예요.' :
       data.isYacht ? ' · 같은 선체가 학습과 시험에 섞여 있어서 점수가 다소 후하게 나올 수 있어요.' : '.');
    drawNet(trainer);
    updatePredict();
  }

  /* ── 예측 계산기 ── */
  function buildPredictInputs() {
    const { xs } = selected();
    const m = buildMatrix();
    $('nnPredIn').innerHTML = xs.map((ci, j) => {
      const col = m.X.map(r => r[j]);
      return `<div><label for="nnp${j}" style="margin-top:0">${esc(data.columns[ci])}</label>` +
             `<input type="number" id="nnp${j}" step="any" value="${+median(col).toPrecision(4)}"></div>`;
    }).join('');
    trainer.ranges = xs.map((_, j) => { const c = m.X.map(r => r[j]); return [Math.min(...c), Math.max(...c)]; });
    xs.forEach((_, j) => $('nnp' + j).addEventListener('input', updatePredict));
  }
  function updatePredict() {
    if (!trainer || !trainer.ranges) return;
    const row = trainer.ranges.map((_, j) => parseFloat($('nnp' + j).value));
    if (row.some(v => !isFinite(v))) { $('nnPredOut').innerHTML = ''; return; }
    const yName = data.columns[+$('nnY').value];
    const box = (cls, label, v) => `<div class="${cls}">${esc(label)}<b>${fmt(v)}</b></div>`;
    $('nnPredOut').innerHTML =
      trainer.baselines.map(m => box('', m.label, m.predictRaw(row))).join('') +
      box('nn', '인공신경망', trainer.predictRaw(row));
    const out = [];
    trainer.ranges.forEach(([a, b], j) => { if (row[j] < a || row[j] > b) out.push(data.columns[selected().xs[j]]); });
    $('nnWarn').textContent = out.length
      ? `${out.join(', ')} 값이 데이터 범위 밖이라 예측이 부정확할 수 있어요.` : '';
    $('nnPredOut').setAttribute('aria-label', `${yName} 예측값`);
  }

  /* ── 학습 시작 / 멈춤 ── */
  function startTraining() {
    try {
      const m = buildMatrix();
      stopTraining();
      const cfg = {
        X: m.X, y: m.y, keys: m.keys,
        hidden: Array(+radioVal('nnlay')).fill(+$('nnNodes').value),
        act: radioVal('nnact'), lr: +$('nnLr').value, epochs: +$('nnEpochs').value,
        seed: SEED, split: data.isYacht ? $('nnSplit').value : 'random',
      };
      trainer = new Trainer(cfg);
      buildPredictInputs();
      running = true;
      $('nnTrain').disabled = true; $('nnStop').disabled = false;
      setStatus('학습 중…');
      const perFrame = Math.max(1, Math.round(cfg.epochs / 120));
      const tick = () => {
        if (!running) return;
        trainer.runEpochs(perFrame);
        const fin = trainer.finished;
        refresh(fin);
        if (fin) finish(); else raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
    } catch (err) {
      setStatus(err.message, 'err');
    }
  }
  function finish() {
    running = false;
    $('nnTrain').disabled = false; $('nnStop').disabled = true;
    setStatus('학습이 끝났어요. 설정을 바꿔서 다시 학습해 보세요.', 'ok');
  }
  function stopTraining() {
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
    if (running) {
      running = false;
      $('nnTrain').disabled = false; $('nnStop').disabled = true;
      setStatus('학습을 멈췄어요.');
    }
  }

  /* ── 이벤트 ── */
  $('nnSample').addEventListener('click', async () => {
    try {
      setStatus('예제 데이터를 불러오는 중…');
      const res = await fetch('yacht_hydrodynamics.data');
      if (!res.ok) throw new Error();
      loadData(parseTextTable(await res.text()), '배 속도와 저항 (UCI 요트 저항 실험)');
    } catch (e) {
      setStatus('예제 파일(yacht_hydrodynamics.data)이 이 페이지와 같은 폴더에 없어요. [다른 파일 열기]로 직접 열어주세요.', 'err');
    }
  });
  $('nnFile').addEventListener('change', async e => {
    const f = e.target.files[0]; if (!f) return;
    try {
      setStatus('읽는 중…');
      let t;
      if (typeof readFile === 'function') t = await readFile(f);            // 회귀 탭의 파일 읽기 함수 (엑셀 포함)
      else t = parseTextTable(await f.text());
      loadData(t, f.name);
    } catch (err) { setStatus(`파일을 열지 못했어요. ${err.message}`, 'err'); }
    e.target.value = '';
  });
  $('nnY').addEventListener('change', () => { buildXChecks(); stopTraining(); trainer = null; resetViews(); });
  $('nnXs').addEventListener('change', () => { stopTraining(); trainer = null; resetViews(); });
  ['nnLayers', 'nnAct'].forEach(id => $(id).addEventListener('change', () => { if (data && !trainer) drawNet(null); }));
  $('nnNodes').addEventListener('change', () => { if (data && !trainer) drawNet(null); });
  $('nnTrain').addEventListener('click', startTraining);
  $('nnStop').addEventListener('click', stopTraining);
}

/* ==========================================================
   6. 시작
   ========================================================== */
const api = { MLP, Trainer, makeSplit, lstsq, mulberry32, parseTextTable, numericColumns, r2score, rmse, ACT };
if (typeof module !== 'undefined' && module.exports) module.exports = api;
if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initUI);
  else initUI();
}
root.NN = api;

})(typeof window !== 'undefined' ? window : globalThis);

/**
 * Hero: bbox BCE classifier. Warmup learns a baseline; Retrain perturbs it with noise
 * for fast convergence. Loss chart uses a wide rolling average of per-step BCE.
 */

const CONFIG = {
    targetText: "hi",
    font: "bold 140px Outfit, sans-serif",
    hiddenSize: 8,
    learningRate: 0.42,
    batchSize: 4096,
    trainStepsPerFrame: 14,
    gridCols: 56,
    gridRows: 34,
    rollWindow: 160,
    rawLossCap: 600,
    plotPoints: 260,
    initNoiseStd: 0.065,
    warmupSteps: 1100,
    warmupLr: 0.55
};

const MathUtils = {
    random: (min, max) => Math.random() * (max - min) + min
};

function gaussian() {
    let u = 0;
    let v = 0;
    while (u === 0) {
        u = Math.random();
    }
    while (v === 0) {
        v = Math.random();
    }
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function sigmoid(x) {
    if (x >= 0) {
        const t = Math.exp(-x);
        return 1 / (1 + t);
    }
    const t = Math.exp(x);
    return t / (1 + t);
}

class Tensor {
    constructor(rows, cols, data = null) {
        this.rows = rows;
        this.cols = cols;
        this.data = data || new Float32Array(rows * cols);
    }

    static random(rows, cols, scale = 1.0) {
        const t = new Tensor(rows, cols);
        for (let i = 0; i < t.data.length; i++) {
            t.data[i] = (Math.random() * 2 - 1) * scale;
        }
        return t;
    }

    static zeros(rows, cols) {
        return new Tensor(rows, cols);
    }
}

class SmallMLP {
    constructor(hiddenSize) {
        this.inputSize = 2;
        this.hiddenSize = hiddenSize;

        this.W1 = Tensor.random(2, hiddenSize, Math.sqrt(2 / 2));
        this.b1 = Tensor.zeros(1, hiddenSize);
        this.W2 = Tensor.random(hiddenSize, 1, Math.sqrt(2 / hiddenSize));
        this.b2 = Tensor.zeros(1, 1);
    }

    static fromSnapshot(snap, noiseStd) {
        const H = snap.hiddenSize;
        const m = new SmallMLP(H);
        for (let i = 0; i < m.W1.data.length; i++) {
            m.W1.data[i] = snap.W1[i] + gaussian() * noiseStd;
        }
        for (let i = 0; i < m.b1.data.length; i++) {
            m.b1.data[i] = snap.b1[i] + gaussian() * noiseStd * 0.5;
        }
        for (let i = 0; i < m.W2.data.length; i++) {
            m.W2.data[i] = snap.W2[i] + gaussian() * noiseStd;
        }
        m.b2.data[0] = snap.b2[0] + gaussian() * noiseStd * 0.5;
        return m;
    }

    forward(inputs) {
        this.inputs = inputs;
        const batchSize = inputs.rows;
        const H = this.hiddenSize;

        this.pre1 = new Tensor(batchSize, H);
        this.h1 = new Tensor(batchSize, H);
        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < H; j++) {
                let sum = this.b1.data[j];
                sum += inputs.data[i * 2] * this.W1.data[0 * H + j];
                sum += inputs.data[i * 2 + 1] * this.W1.data[1 * H + j];
                this.pre1.data[i * H + j] = sum;
                this.h1.data[i * H + j] = sum > 0 ? sum : 0;
            }
        }

        this.logits = new Tensor(batchSize, 1);
        for (let i = 0; i < batchSize; i++) {
            let sum = this.b2.data[0];
            for (let j = 0; j < H; j++) {
                sum += this.h1.data[i * H + j] * this.W2.data[j];
            }
            this.logits.data[i] = sum;
        }

        return this.logits;
    }

    trainStep(inputs, labels, lr) {
        this.forward(inputs);
        const batchSize = inputs.rows;
        const H = this.hiddenSize;

        let totalLoss = 0;
        const dLogit = new Tensor(batchSize, 1);

        for (let i = 0; i < batchSize; i++) {
            const z = this.logits.data[i];
            const s = sigmoid(z);
            const y = labels[i];
            totalLoss += -(y * Math.log(s + 1e-7) + (1 - y) * Math.log(1 - s + 1e-7));
            dLogit.data[i] = (s - y) / batchSize;
        }
        totalLoss /= batchSize;

        const dH1 = new Tensor(batchSize, H);
        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < H; j++) {
                const chain = dLogit.data[i] * this.W2.data[j];
                const pre = this.pre1.data[i * H + j];
                dH1.data[i * H + j] = pre > 0 ? chain : 0;
            }
        }

        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < H; j++) {
                this.W2.data[j] -= lr * this.h1.data[i * H + j] * dLogit.data[i] / batchSize;
            }
            this.b2.data[0] -= lr * dLogit.data[i] / batchSize;
        }

        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < H; j++) {
                for (let k = 0; k < 2; k++) {
                    const inVal = inputs.data[i * 2 + k];
                    this.W1.data[k * H + j] -= lr * inVal * dH1.data[i * H + j] / batchSize;
                }
            }
            for (let j = 0; j < H; j++) {
                this.b1.data[j] -= lr * dH1.data[i * H + j] / batchSize;
            }
        }

        return totalLoss;
    }
}

function snapshotFromMLP(net) {
    return {
        hiddenSize: net.hiddenSize,
        W1: new Float32Array(net.W1.data),
        b1: new Float32Array(net.b1.data),
        W2: new Float32Array(net.W2.data),
        b2: new Float32Array(net.b2.data)
    };
}

const canvas = document.getElementById('nn-canvas');
const ctx = canvas.getContext('2d');
const lossCanvas = document.getElementById('loss-canvas');
const lossCtx = lossCanvas.getContext('2d');

let mlp;
let batchInputs;
let batchLabels;
let gridSampleInput = null;
let epoch = 0;
let exploreModalOpen = false;

let warmupBaselineSnapshot = null;
let warmBaselineBboxKey = null;

const lossRawRing = [];
let smoothedPlotHistory = [];

let bbox = { xmin: -0.5, xmax: 0.5, ymin: -0.5, ymax: 0.5 };

function bboxToKey() {
    return [bbox.xmin, bbox.xmax, bbox.ymin, bbox.ymax].map((v) => v.toFixed(5)).join("|");
}

function insideBbox(x, y) {
    return x >= bbox.xmin && x <= bbox.xmax && y >= bbox.ymin && y <= bbox.ymax ? 1 : 0;
}

function resize() {
    const container = canvas.parentElement;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    const panel = lossCanvas.parentElement;
    const headerEl = panel.querySelector('h3');
    const statsEl = panel.querySelector('.stats');
    const btnRow = panel.querySelector('.hero-btn-row');
    const used = (headerEl ? headerEl.offsetHeight : 0) + (statsEl ? statsEl.offsetHeight : 0) + (btnRow ? btnRow.offsetHeight : 0) + 28;
    lossCanvas.width = panel.clientWidth - 30;
    lossCanvas.height = Math.max(90, panel.clientHeight - used);

    computeBboxFromText();
    initNetwork();
}

window.addEventListener('resize', resize);

function computeBboxFromText() {
    const osc = document.createElement('canvas');
    osc.width = Math.max(1, canvas.width);
    osc.height = Math.max(1, canvas.height);
    const osctx = osc.getContext('2d');

    osctx.font = CONFIG.font;
    osctx.fillStyle = 'black';
    osctx.textAlign = 'center';
    osctx.textBaseline = 'middle';
    osctx.fillText(CONFIG.targetText, osc.width / 2, osc.height / 2);

    const imgData = osctx.getImageData(0, 0, osc.width, osc.height);
    const data = imgData.data;

    let minPxX = osc.width;
    let minPxY = osc.height;
    let maxPxX = 0;
    let maxPxY = 0;
    let any = false;

    for (let y = 0; y < osc.height; y++) {
        for (let x = 0; x < osc.width; x++) {
            if (data[(y * osc.width + x) * 4 + 3] > 128) {
                any = true;
                if (x < minPxX) {
                    minPxX = x;
                }
                if (x > maxPxX) {
                    maxPxX = x;
                }
                if (y < minPxY) {
                    minPxY = y;
                }
                if (y > maxPxY) {
                    maxPxY = y;
                }
            }
        }
    }

    if (!any) {
        bbox = { xmin: -0.3, xmax: 0.3, ymin: -0.3, ymax: 0.3 };
        return;
    }

    const padX = Math.max(2, (maxPxX - minPxX) * 0.04);
    const padY = Math.max(2, (maxPxY - minPxY) * 0.04);
    minPxX = Math.max(0, minPxX - padX);
    maxPxX = Math.min(osc.width - 1, maxPxX + padX);
    minPxY = Math.max(0, minPxY - padY);
    maxPxY = Math.min(osc.height - 1, maxPxY + padY);

    bbox = {
        xmin: (minPxX / osc.width) * 2 - 1,
        xmax: (maxPxX / osc.width) * 2 - 1,
        ymin: (minPxY / osc.height) * 2 - 1,
        ymax: (maxPxY / osc.height) * 2 - 1
    };
}

function allocBatch() {
    const n = CONFIG.batchSize;
    batchInputs = new Tensor(n, 2);
    batchLabels = new Float32Array(n);
}

function sampleBatch() {
    const n = CONFIG.batchSize;
    for (let i = 0; i < n; i++) {
        const x = MathUtils.random(-1, 1);
        const y = MathUtils.random(-1, 1);
        batchInputs.data[i * 2] = x;
        batchInputs.data[i * 2 + 1] = y;
        batchLabels[i] = insideBbox(x, y);
    }
}

function buildGridSampleTensor() {
    const cols = CONFIG.gridCols;
    const rows = CONFIG.gridRows;
    const n = cols * rows;
    gridSampleInput = new Tensor(n, 2);
    let idx = 0;
    for (let gy = 0; gy < rows; gy++) {
        for (let gx = 0; gx < cols; gx++) {
            gridSampleInput.data[idx * 2] = ((gx + 0.5) / cols) * 2 - 1;
            gridSampleInput.data[idx * 2 + 1] = ((gy + 0.5) / rows) * 2 - 1;
            idx++;
        }
    }
}

function runWarmupTraining() {
    const temp = new SmallMLP(CONFIG.hiddenSize);
    for (let step = 0; step < CONFIG.warmupSteps; step++) {
        sampleBatch();
        temp.trainStep(batchInputs, batchLabels, CONFIG.warmupLr);
    }
    return snapshotFromMLP(temp);
}

function ensureWarmBaseline() {
    const key = bboxToKey();
    if (warmupBaselineSnapshot && warmBaselineBboxKey === key) {
        return;
    }
    warmupBaselineSnapshot = runWarmupTraining();
    warmBaselineBboxKey = key;
}

function initNetwork() {
    allocBatch();
    ensureWarmBaseline();
    mlp = SmallMLP.fromSnapshot(warmupBaselineSnapshot, CONFIG.initNoiseStd);
    buildGridSampleTensor();
    lossRawRing.length = 0;
    smoothedPlotHistory.length = 0;
    epoch = 0;
}

function rollingAverage() {
    const n = lossRawRing.length;
    if (n === 0) {
        return 0;
    }
    const w = Math.min(CONFIG.rollWindow, n);
    let s = 0;
    for (let i = n - w; i < n; i++) {
        s += lossRawRing[i];
    }
    return s / w;
}

function normToCanvasX(nx) {
    return (nx + 1) / 2 * canvas.width;
}

function normToCanvasY(ny) {
    return (ny + 1) / 2 * canvas.height;
}

function drawMainCanvas() {
    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const cols = CONFIG.gridCols;
    const rows = CONFIG.gridRows;
    const cellW = canvas.width / cols;
    const cellH = canvas.height / rows;

    if (gridSampleInput && gridSampleInput.rows === cols * rows) {
        mlp.forward(gridSampleInput);
    }

    let idx = 0;
    for (let gy = 0; gy < rows; gy++) {
        for (let gx = 0; gx < cols; gx++) {
            const p = gridSampleInput ? sigmoid(mlp.logits.data[idx]) : 0;
            idx++;
            const t = Math.floor(40 + p * 180);
            ctx.fillStyle = `rgb(${255 - t},${200 - Math.floor(p * 80)},${180 - Math.floor(p * 100)})`;
            ctx.fillRect(gx * cellW, gy * cellH, cellW + 0.5, cellH + 0.5);
        }
    }

    const x0 = normToCanvasX(bbox.xmin);
    const x1 = normToCanvasX(bbox.xmax);
    const y0 = normToCanvasY(bbox.ymin);
    const y1 = normToCanvasY(bbox.ymax);
    ctx.strokeStyle = '#2c2c2c';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
    ctx.setLineDash([]);

    ctx.strokeStyle = 'rgba(210, 105, 30, 0.85)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
}

function drawLoss() {
    const w = lossCanvas.width;
    const h = lossCanvas.height;
    lossCtx.clearRect(0, 0, w, h);

    if (smoothedPlotHistory.length < 2) {
        return;
    }

    lossCtx.beginPath();
    lossCtx.strokeStyle = '#D2691E';
    lossCtx.lineWidth = 2;

    const maxLoss = Math.max(...smoothedPlotHistory, 0.02);
    const minLoss = 0;

    for (let i = 0; i < smoothedPlotHistory.length; i++) {
        const x = (i / (smoothedPlotHistory.length - 1)) * w;
        const y = h - ((smoothedPlotHistory[i] - minLoss) / (maxLoss - minLoss)) * (h * 0.8) - 10;
        if (i === 0) {
            lossCtx.moveTo(x, y);
        } else {
            lossCtx.lineTo(x, y);
        }
    }
    lossCtx.stroke();
}

function tensorStats(arr) {
    let minV = arr[0];
    let maxV = arr[0];
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
        const v = arr[i];
        if (v < minV) {
            minV = v;
        }
        if (v > maxV) {
            maxV = v;
        }
        sum += v;
    }
    return { minV, maxV, mean: sum / arr.length };
}

function fmt(v) {
    return v.toFixed(4);
}

function refreshExploreWeights() {
    const el = document.getElementById('explore-weights');
    if (!el || !mlp) {
        return;
    }
    const H = mlp.hiddenSize;
    const w1 = mlp.W1.data;
    const b1 = mlp.b1.data;
    const w2 = mlp.W2.data;

    const hs = [];
    for (let j = 0; j < H; j++) {
        hs.push(`h${j}`);
    }

    let w1Rows = `<tr><th></th>${hs.map((h) => `<th>${h}</th>`).join('')}</tr>`;
    w1Rows += `<tr><th>x</th>${Array.from({ length: H }, (_, j) => `<td>${fmt(w1[j])}</td>`).join('')}</tr>`;
    w1Rows += `<tr><th>y</th>${Array.from({ length: H }, (_, j) => `<td>${fmt(w1[H + j])}</td>`).join('')}</tr>`;

    const b1Header = `<tr>${Array.from({ length: H }, (_, j) => `<th>h${j}</th>`).join('')}</tr>`;
    const b1Row = `<tr>${Array.from({ length: H }, (_, j) => `<td>${fmt(b1[j])}</td>`).join('')}</tr>`;
    const w2Cells = Array.from({ length: H }, (_, j) => `<tr><th>h${j} &rarr; logit</th><td>${fmt(w2[j])}</td></tr>`).join('');

    const s1 = tensorStats(w1);
    const s2 = tensorStats(w2);
    const sb1 = tensorStats(b1);

    el.innerHTML = `
        <div class="explore-weight-summary">
            <div><strong>W1</strong> min ${fmt(s1.minV)}, max ${fmt(s1.maxV)}, mean ${fmt(s1.mean)}</div>
            <div><strong>b1</strong> min ${fmt(sb1.minV)}, max ${fmt(sb1.maxV)}, mean ${fmt(sb1.mean)}</div>
            <div><strong>W2</strong> min ${fmt(s2.minV)}, max ${fmt(s2.maxV)}, mean ${fmt(s2.mean)}</div>
            <div><strong>b2 (logit bias)</strong> ${fmt(mlp.b2.data[0])}</div>
        </div>
        <p class="explore-table-label">W1 (rows = inputs x, y)</p>
        <table class="explore-weight-table">${w1Rows}</table>
        <p class="explore-table-label">b1 (per hidden unit)</p>
        <table class="explore-weight-table">${b1Header}${b1Row}</table>
        <p class="explore-table-label">W2 (hidden to logit)</p>
        <table class="explore-weight-table">${w2Cells}</table>
    `;
}

function drawNetworkViz() {
    const viz = document.getElementById('nn-viz-canvas');
    if (!viz || !mlp) {
        return;
    }
    const vctx = viz.getContext('2d');
    const W = viz.width;
    const H = viz.height;
    vctx.clearRect(0, 0, W, H);
    vctx.fillStyle = '#fffef8';
    vctx.fillRect(0, 0, W, H);

    const xIn = 55;
    const xH = W / 2;
    const xOut = W - 55;
    const inCount = 2;
    const hCount = mlp.hiddenSize;

    function yFor(idx, total) {
        return (H / (total + 1)) * (idx + 1);
    }

    const inPos = [];
    const hidPos = [];
    for (let i = 0; i < inCount; i++) {
        inPos.push({ x: xIn, y: yFor(i, inCount) });
    }
    for (let j = 0; j < hCount; j++) {
        hidPos.push({ x: xH, y: yFor(j, hCount) });
    }
    const outPos = { x: xOut, y: H / 2 };

    const maxW = 2.5;
    function lineForWeight(wv) {
        const a = Math.min(1, Math.abs(wv) * 0.8);
        return { w: 0.4 + a * maxW, neg: wv < 0 };
    }

    for (let k = 0; k < inCount; k++) {
        for (let j = 0; j < hCount; j++) {
            const wv = mlp.W1.data[k * hCount + j];
            const { w, neg } = lineForWeight(wv);
            vctx.beginPath();
            vctx.strokeStyle = neg ? 'rgba(85,107,47,0.45)' : 'rgba(210,105,30,0.5)';
            vctx.lineWidth = w;
            vctx.moveTo(inPos[k].x, inPos[k].y);
            vctx.lineTo(hidPos[j].x, hidPos[j].y);
            vctx.stroke();
        }
    }

    for (let j = 0; j < hCount; j++) {
        const wv = mlp.W2.data[j];
        const { w, neg } = lineForWeight(wv);
        vctx.beginPath();
        vctx.strokeStyle = neg ? 'rgba(85,107,47,0.5)' : 'rgba(210,105,30,0.55)';
        vctx.lineWidth = w;
        vctx.moveTo(hidPos[j].x, hidPos[j].y);
        vctx.lineTo(outPos.x, outPos.y);
        vctx.stroke();
    }

    const nodeR = 9;
    vctx.font = '11px Outfit, sans-serif';
    vctx.textAlign = 'center';

    for (let k = 0; k < inCount; k++) {
        vctx.beginPath();
        vctx.fillStyle = '#e8dcc8';
        vctx.strokeStyle = '#8B4513';
        vctx.lineWidth = 1.5;
        vctx.arc(inPos[k].x, inPos[k].y, nodeR, 0, Math.PI * 2);
        vctx.fill();
        vctx.stroke();
        vctx.fillStyle = '#333';
        vctx.fillText(k === 0 ? 'x' : 'y', inPos[k].x, inPos[k].y + 4);
    }

    for (let j = 0; j < hCount; j++) {
        vctx.beginPath();
        vctx.fillStyle = '#f5ebe0';
        vctx.strokeStyle = '#556B2F';
        vctx.lineWidth = 1.5;
        vctx.arc(hidPos[j].x, hidPos[j].y, nodeR, 0, Math.PI * 2);
        vctx.fill();
        vctx.stroke();
    }

    vctx.beginPath();
    vctx.fillStyle = '#fdebd0';
    vctx.strokeStyle = '#D2691E';
    vctx.lineWidth = 2;
    vctx.arc(outPos.x, outPos.y, nodeR + 2, 0, Math.PI * 2);
    vctx.fill();
    vctx.stroke();
    vctx.fillStyle = '#333';
    vctx.fillText('p', outPos.x, outPos.y + 4);

    vctx.textAlign = 'left';
    vctx.fillStyle = '#555';
    vctx.font = '10px Outfit, sans-serif';
    vctx.fillText('Input (2D point)', 8, 16);
    vctx.textAlign = 'center';
    vctx.fillText(`Hidden (${hCount}, ReLU)`, xH, 16);
    vctx.textAlign = 'right';
    vctx.fillText('Logit (BCE)', W - 8, 16);
}

function refreshExploreText() {
    const archEl = document.getElementById('explore-arch');
    if (!archEl || !mlp) {
        return;
    }
    archEl.textContent = `Architecture: 2 inputs -> ${mlp.hiddenSize} ReLU -> 1 logit (batch ${CONFIG.batchSize}). A long offline warmup fits the bbox task; Retrain copies those weights with small Gaussian noise so the heatmap snaps back quickly. Chart = rolling mean BCE over the last ${CONFIG.rollWindow} optimizer steps.`;
}

function animate() {
    if (!mlp || !batchInputs) {
        requestAnimationFrame(animate);
        return;
    }

    let lastStepLoss = 0;
    for (let s = 0; s < CONFIG.trainStepsPerFrame; s++) {
        sampleBatch();
        lastStepLoss = mlp.trainStep(batchInputs, batchLabels, CONFIG.learningRate);
        epoch++;
        lossRawRing.push(lastStepLoss);
        if (lossRawRing.length > CONFIG.rawLossCap) {
            lossRawRing.shift();
        }
    }

    const roll = rollingAverage();
    smoothedPlotHistory.push(roll);
    if (smoothedPlotHistory.length > CONFIG.plotPoints) {
        smoothedPlotHistory.shift();
    }

    document.getElementById('epoch-counter').innerText = epoch;
    document.getElementById('loss-value').innerText = roll.toFixed(4);
    const instEl = document.getElementById('loss-instant');
    if (instEl) {
        instEl.textContent = lastStepLoss.toFixed(4);
    }

    drawMainCanvas();
    drawLoss();

    if (exploreModalOpen) {
        drawNetworkViz();
        refreshExploreWeights();
    }

    requestAnimationFrame(animate);
}

function setExploreOpen(open) {
    const modal = document.getElementById('explore-modal');
    if (!modal) {
        return;
    }
    exploreModalOpen = open;
    modal.hidden = !open;
    modal.setAttribute('aria-hidden', open ? 'false' : 'true');
    if (open && mlp) {
        refreshExploreText();
        drawNetworkViz();
        refreshExploreWeights();
    }
}

async function init() {
    try {
        await document.fonts.ready;
    } catch (e) {
        console.warn("Font loading API not supported or failed", e);
    }

    setTimeout(() => {
        resize();
        document.getElementById('retrain-btn').addEventListener('click', initNetwork);
        const exploreBtn = document.getElementById('explore-btn');
        const modal = document.getElementById('explore-modal');
        if (exploreBtn) {
            exploreBtn.addEventListener('click', () => setExploreOpen(true));
        }
        if (modal) {
            modal.querySelector('.explore-modal-backdrop').addEventListener('click', () => setExploreOpen(false));
            modal.querySelector('.explore-modal-close').addEventListener('click', () => setExploreOpen(false));
        }
        animate();
    }, 100);
}

init();

/**
 * Neural Network Visualization
 * MLP with 1:1 supervised MSE: fixed noise inputs map to paired targets forming "hi".
 */

const CONFIG = {
    maxParticles: 800,
    learningRate: 0.08,
    hiddenSize: 64,
    targetText: "hi",
    font: "bold 140px Outfit, sans-serif"
};

const MathUtils = {
    random: (min, max) => Math.random() * (max - min) + min,
    randomGaussian: () => {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
};

class Tensor {
    constructor(rows, cols, data = null) {
        this.rows = rows;
        this.cols = cols;
        this.data = data || new Float32Array(rows * cols);
    }

    static random(rows, cols, scale = 1.0) {
        const t = new Tensor(rows, cols);
        for (let i = 0; i < t.data.length; i++) {
            t.data[i] = MathUtils.randomGaussian() * scale;
        }
        return t;
    }

    static zeros(rows, cols) {
        return new Tensor(rows, cols);
    }
}

class MLP {
    constructor(inputSize, hiddenSize, outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        this.W1 = Tensor.random(inputSize, hiddenSize, Math.sqrt(2 / inputSize));
        this.b1 = Tensor.zeros(1, hiddenSize);

        this.W2 = Tensor.random(hiddenSize, hiddenSize, Math.sqrt(2 / hiddenSize));
        this.b2 = Tensor.zeros(1, hiddenSize);

        this.W3 = Tensor.random(hiddenSize, outputSize, Math.sqrt(2 / hiddenSize));
        this.b3 = Tensor.zeros(1, outputSize);
    }

    forward(inputs) {
        this.inputs = inputs;
        const batchSize = inputs.rows;

        this.z1 = new Tensor(batchSize, this.W1.cols);
        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < this.W1.cols; j++) {
                let sum = this.b1.data[j];
                for (let k = 0; k < this.W1.rows; k++) {
                    sum += inputs.data[i * inputs.cols + k] * this.W1.data[k * this.W1.cols + j];
                }
                this.z1.data[i * this.z1.cols + j] = sum > 0 ? sum : 0;
            }
        }

        this.z2 = new Tensor(batchSize, this.W2.cols);
        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < this.W2.cols; j++) {
                let sum = this.b2.data[j];
                for (let k = 0; k < this.W2.rows; k++) {
                    sum += this.z1.data[i * this.z1.cols + k] * this.W2.data[k * this.W2.cols + j];
                }
                this.z2.data[i * this.z2.cols + j] = sum > 0 ? sum : 0;
            }
        }

        this.output = new Tensor(batchSize, this.W3.cols);
        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < this.W3.cols; j++) {
                let sum = this.b3.data[j];
                for (let k = 0; k < this.W3.rows; k++) {
                    sum += this.z2.data[i * this.z2.cols + k] * this.W3.data[k * this.W3.cols + j];
                }
                this.output.data[i * this.output.cols + j] = sum;
            }
        }

        return this.output;
    }

    /**
     * Mean MSE: (1/N) sum_i 0.5 * ||out_i - t_i||^2
     */
    trainStep(inputs, targets, lr) {
        this.forward(inputs);
        const batchSize = this.output.rows;
        const outputDim = this.output.cols;

        if (!targets || targets.length !== batchSize) {
            return 0;
        }

        let totalLoss = 0;
        const dLoss_dOut = new Tensor(batchSize, outputDim);

        for (let i = 0; i < batchSize; i++) {
            const ox = this.output.data[i * 2];
            const oy = this.output.data[i * 2 + 1];
            const tx = targets[i].x;
            const ty = targets[i].y;
            const ex = ox - tx;
            const ey = oy - ty;
            totalLoss += 0.5 * (ex * ex + ey * ey);
            dLoss_dOut.data[i * 2] = ex / batchSize;
            dLoss_dOut.data[i * 2 + 1] = ey / batchSize;
        }
        totalLoss /= batchSize;

        const dZ2 = new Tensor(batchSize, this.W2.cols);
        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < this.W2.cols; j++) {
                let sum = 0;
                for (let k = 0; k < outputDim; k++) {
                    sum += dLoss_dOut.data[i * outputDim + k] * this.W3.data[j * this.W3.cols + k];
                }
                dZ2.data[i * this.W2.cols + j] = (this.z2.data[i * this.W2.cols + j] > 0) ? sum : 0;
            }
        }

        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < this.W3.rows; j++) {
                for (let k = 0; k < this.W3.cols; k++) {
                    this.W3.data[j * this.W3.cols + k] -= lr * this.z2.data[i * this.z2.cols + j] * dLoss_dOut.data[i * outputDim + k] / batchSize;
                }
            }
            for (let k = 0; k < this.b3.cols; k++) {
                this.b3.data[k] -= lr * dLoss_dOut.data[i * outputDim + k] / batchSize;
            }
        }

        const dZ1 = new Tensor(batchSize, this.W1.cols);
        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < this.W1.cols; j++) {
                let sum = 0;
                for (let k = 0; k < this.W2.cols; k++) {
                    sum += dZ2.data[i * this.W2.cols + k] * this.W2.data[j * this.W2.cols + k];
                }
                dZ1.data[i * this.W1.cols + j] = (this.z1.data[i * this.W1.cols + j] > 0) ? sum : 0;
            }
        }

        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < this.W2.rows; j++) {
                for (let k = 0; k < this.W2.cols; k++) {
                    this.W2.data[j * this.W2.cols + k] -= lr * this.z1.data[i * this.z1.cols + j] * dZ2.data[i * this.W2.cols + k] / batchSize;
                }
            }
            for (let k = 0; k < this.b2.cols; k++) {
                this.b2.data[k] -= lr * dZ2.data[i * this.W2.cols + k] / batchSize;
            }
        }

        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < this.W1.rows; j++) {
                for (let k = 0; k < this.W1.cols; k++) {
                    this.W1.data[j * this.W1.cols + k] -= lr * this.inputs.data[i * this.inputs.cols + j] * dZ1.data[i * this.W1.cols + k] / batchSize;
                }
            }
            for (let k = 0; k < this.b1.cols; k++) {
                this.b1.data[k] -= lr * dZ1.data[i * this.W1.cols + k] / batchSize;
            }
        }

        return totalLoss;
    }
}

function frobNorm2D(tensor) {
    let s = 0;
    for (let i = 0; i < tensor.data.length; i++) {
        s += tensor.data[i] * tensor.data[i];
    }
    return Math.sqrt(s);
}

function tensorMinMaxMean(tensor) {
    let minV = tensor.data[0];
    let maxV = tensor.data[0];
    let sum = 0;
    for (let i = 0; i < tensor.data.length; i++) {
        const v = tensor.data[i];
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
        sum += v;
    }
    const mean = sum / tensor.data.length;
    return { minV, maxV, mean };
}

const canvas = document.getElementById('nn-canvas');
const ctx = canvas.getContext('2d');
const lossCanvas = document.getElementById('loss-canvas');
const lossCtx = lossCanvas.getContext('2d');

let mlp;
let fixedInput;
let targetPoints = [];
let lossHistory = [];
let epoch = 0;
let particleCount = 0;

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

    initTargets();
}

window.addEventListener('resize', resize);

function subsampleEvenly(sorted, maxN) {
    if (sorted.length <= maxN) {
        return sorted;
    }
    if (maxN <= 1) {
        return [sorted[0]];
    }
    const out = [];
    const step = (sorted.length - 1) / (maxN - 1);
    for (let i = 0; i < maxN; i++) {
        const idx = Math.round(i * step);
        out.push(sorted[Math.min(idx, sorted.length - 1)]);
    }
    return out;
}

function initTargets() {
    const osc = document.createElement('canvas');
    osc.width = canvas.width;
    osc.height = canvas.height;
    const osctx = osc.getContext('2d');

    osctx.font = CONFIG.font;
    osctx.fillStyle = 'black';
    osctx.textAlign = 'center';
    osctx.textBaseline = 'middle';
    osctx.fillText(CONFIG.targetText, osc.width / 2, osc.height / 2);

    const imgData = osctx.getImageData(0, 0, osc.width, osc.height);
    const data = imgData.data;

    const raw = [];
    for (let y = 0; y < osc.height; y += 2) {
        for (let x = 0; x < osc.width; x += 2) {
            if (data[(y * osc.width + x) * 4 + 3] > 128) {
                raw.push({
                    x: (x / osc.width) * 2 - 1,
                    y: (y / osc.height) * 2 - 1
                });
            }
        }
    }

    raw.sort((a, b) => (a.y !== b.y ? a.y - b.y : a.x - b.x));
    targetPoints = subsampleEvenly(raw, CONFIG.maxParticles);

    if (targetPoints.length === 0) {
        for (let i = 0; i < 200; i++) {
            const angle = (i / 200) * Math.PI * 2;
            targetPoints.push({
                x: Math.cos(angle) * 0.5,
                y: Math.sin(angle) * 0.5
            });
        }
    }

    particleCount = targetPoints.length;
    initNetwork();
}

function initNetwork() {
    const inputSize = 4;
    const n = particleCount || targetPoints.length;
    if (n === 0) {
        return;
    }

    fixedInput = new Tensor(n, inputSize);
    for (let i = 0; i < n; i++) {
        const x = MathUtils.random(-1, 1);
        const y = MathUtils.random(-1, 1);
        fixedInput.data[i * 4] = x;
        fixedInput.data[i * 4 + 1] = y;
        fixedInput.data[i * 4 + 2] = Math.sqrt(x * x + y * y);
        fixedInput.data[i * 4 + 3] = Math.atan2(y, x);
    }

    mlp = new MLP(inputSize, CONFIG.hiddenSize, 2);
    lossHistory = [];
    epoch = 0;
}

function drawLoss() {
    const w = lossCanvas.width;
    const h = lossCanvas.height;
    lossCtx.clearRect(0, 0, w, h);

    if (lossHistory.length < 2) {
        return;
    }

    lossCtx.beginPath();
    lossCtx.strokeStyle = '#D2691E';
    lossCtx.lineWidth = 2;

    const maxLoss = Math.max(...lossHistory);
    const minLoss = 0;

    for (let i = 0; i < lossHistory.length; i++) {
        const x = (i / (lossHistory.length - 1)) * w;
        const y = h - ((lossHistory[i] - minLoss) / (maxLoss - minLoss || 1)) * (h * 0.8) - 10;
        if (i === 0) {
            lossCtx.moveTo(x, y);
        } else {
            lossCtx.lineTo(x, y);
        }
    }
    lossCtx.stroke();
}

function animate() {
    if (!mlp || !fixedInput || targetPoints.length === 0) {
        animationId = requestAnimationFrame(animate);
        return;
    }

    const loss = mlp.trainStep(fixedInput, targetPoints, CONFIG.learningRate);
    lossHistory.push(loss);
    if (lossHistory.length > 200) {
        lossHistory.shift();
    }

    epoch++;
    document.getElementById('epoch-counter').innerText = epoch;
    document.getElementById('loss-value').innerText = loss.toFixed(4);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const output = mlp.output;
    ctx.fillStyle = '#D2691E';

    for (let i = 0; i < output.rows; i++) {
        const x = (output.data[i * 2] + 1) / 2 * canvas.width;
        const y = (output.data[i * 2 + 1] + 1) / 2 * canvas.height;

        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fill();
    }

    drawLoss();

    animationId = requestAnimationFrame(animate);
}

function refreshExplorePanel() {
    const archEl = document.getElementById('explore-arch');
    const statsEl = document.getElementById('explore-stats');
    if (!archEl || !statsEl) {
        return;
    }
    if (!mlp) {
        archEl.textContent = 'Network not initialized yet.';
        statsEl.innerHTML = '';
        return;
    }

    const n = particleCount || (mlp.output ? mlp.output.rows : 0);
    archEl.textContent = `Layers: ${mlp.inputSize} -> ${mlp.hiddenSize} (ReLU) -> ${mlp.hiddenSize} (ReLU) -> ${mlp.outputSize} (linear). One output row per point (${n} points). Each row gets the same fixed 4D noise vector for the whole run; Retrain resamples noise and reinitializes weights.`;

    const w1 = frobNorm2D(mlp.W1);
    const w2 = frobNorm2D(mlp.W2);
    const w3 = frobNorm2D(mlp.W3);
    const mm = tensorMinMaxMean(mlp.W1);
    statsEl.innerHTML = [
        `<div><strong>||W1||_F</strong> ${w1.toFixed(3)}</div>`,
        `<div><strong>||W2||_F</strong> ${w2.toFixed(3)}</div>`,
        `<div><strong>||W3||_F</strong> ${w3.toFixed(3)}</div>`,
        `<div><strong>W1 weights</strong> min ${mm.minV.toFixed(3)}, max ${mm.maxV.toFixed(3)}, mean ${mm.mean.toFixed(3)}</div>`
    ].join('');
}

function setExploreOpen(open) {
    const modal = document.getElementById('explore-modal');
    if (!modal) {
        return;
    }
    modal.hidden = !open;
    modal.setAttribute('aria-hidden', open ? 'false' : 'true');
    if (open && mlp) {
        refreshExplorePanel();
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

/**
 * Neural Network Visualization
 * Implements a simple MLP that learns to position particles to form the name "Mikil Radu Foss".
 */

// Configuration
const CONFIG = {
    particleCount: 800,
    learningRate: 0.05,
    hiddenSize: 64,
    targetText: "Mikil Radu Foss",
    font: "bold 80px Outfit, sans-serif"
};

// --- Math Utilities ---
const MathUtils = {
    random: (min, max) => Math.random() * (max - min) + min,
    randomGaussian: () => {
        let u = 0, v = 0;
        while(u === 0) u = Math.random();
        while(v === 0) v = Math.random();
        return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
    }
};

// --- Matrix/Tensor Class (Simplified) ---
class Tensor {
    constructor(rows, cols, data = null) {
        this.rows = rows;
        this.cols = cols;
        this.data = data || new Float32Array(rows * cols);
    }

    static random(rows, cols, scale = 1.0) {
        const t = new Tensor(rows, cols);
        for(let i = 0; i < t.data.length; i++) {
            t.data[i] = MathUtils.randomGaussian() * scale;
        }
        return t;
    }

    static zeros(rows, cols) {
        return new Tensor(rows, cols);
    }
}

// --- MLP Implementation ---
class MLP {
    constructor(inputSize, hiddenSize, outputSize) {
        // Layer 1: Input -> Hidden
        this.W1 = Tensor.random(inputSize, hiddenSize, Math.sqrt(2/inputSize));
        this.b1 = Tensor.zeros(1, hiddenSize);
        
        // Layer 2: Hidden -> Hidden
        this.W2 = Tensor.random(hiddenSize, hiddenSize, Math.sqrt(2/hiddenSize));
        this.b2 = Tensor.zeros(1, hiddenSize);

        // Layer 3: Hidden -> Output
        this.W3 = Tensor.random(hiddenSize, outputSize, Math.sqrt(2/hiddenSize));
        this.b3 = Tensor.zeros(1, outputSize);
        
        // Gradients
        this.dW1 = Tensor.zeros(inputSize, hiddenSize);
        this.db1 = Tensor.zeros(1, hiddenSize);
        this.dW2 = Tensor.zeros(hiddenSize, hiddenSize);
        this.db2 = Tensor.zeros(1, hiddenSize);
        this.dW3 = Tensor.zeros(hiddenSize, outputSize);
        this.db3 = Tensor.zeros(1, outputSize);
    }

    forward(inputs) {
        // Inputs: (Batch, InputSize)
        this.inputs = inputs;
        const batchSize = inputs.rows;

        // Layer 1
        this.z1 = new Tensor(batchSize, this.W1.cols);
        // MatMul: inputs * W1 + b1
        for(let i=0; i<batchSize; i++) {
            for(let j=0; j<this.W1.cols; j++) {
                let sum = this.b1.data[j];
                for(let k=0; k<this.W1.rows; k++) {
                    sum += inputs.data[i*inputs.cols + k] * this.W1.data[k*this.W1.cols + j];
                }
                this.z1.data[i*this.z1.cols + j] = sum > 0 ? sum : 0; // ReLU
            }
        }

        // Layer 2
        this.z2 = new Tensor(batchSize, this.W2.cols);
        for(let i=0; i<batchSize; i++) {
            for(let j=0; j<this.W2.cols; j++) {
                let sum = this.b2.data[j];
                for(let k=0; k<this.W2.rows; k++) {
                    sum += this.z1.data[i*this.z1.cols + k] * this.W2.data[k*this.W2.cols + j];
                }
                this.z2.data[i*this.z2.cols + j] = sum > 0 ? sum : 0; // ReLU
            }
        }

        // Layer 3 (Output)
        this.output = new Tensor(batchSize, this.W3.cols);
        for(let i=0; i<batchSize; i++) {
            for(let j=0; j<this.W3.cols; j++) {
                let sum = this.b3.data[j];
                for(let k=0; k<this.W3.rows; k++) {
                    sum += this.z2.data[i*this.z2.cols + k] * this.W3.data[k*this.W3.cols + j];
                }
                this.output.data[i*this.output.cols + j] = sum; // Linear
            }
        }
        
        return this.output;
    }

    // Simplified training step using "Chamfer-like" gradients
    // We pull each output point towards its nearest target point
    trainStep(targets, lr) {
        const batchSize = this.output.rows;
        const outputDim = this.output.cols;
        
        // 1. Calculate Loss & Gradients at Output
        // For each output point, find closest target
        let totalLoss = 0;
        const dLoss_dOut = new Tensor(batchSize, outputDim);
        
        for(let i=0; i<batchSize; i++) {
            const ox = this.output.data[i*2];
            const oy = this.output.data[i*2+1];
            
            // Find nearest target (Brute force is fine for <1000 points)
            let minDist = Infinity;
            let tx = 0, ty = 0;
            
            // Optimization: Check a random subset of targets to speed up
            // or check all. 800x800 is 640k ops, might be slow per frame.
            // Let's check all for quality.
            for(let j=0; j<targets.length; j++) {
                const dx = ox - targets[j].x;
                const dy = oy - targets[j].y;
                const dist = dx*dx + dy*dy;
                if(dist < minDist) {
                    minDist = dist;
                    tx = targets[j].x;
                    ty = targets[j].y;
                }
            }
            
            totalLoss += minDist;
            
            // Gradient: 2 * (output - target)
            dLoss_dOut.data[i*2] = 2 * (ox - tx);
            dLoss_dOut.data[i*2+1] = 2 * (oy - ty);
        }
        
        totalLoss /= batchSize;

        // 2. Backprop
        
        // Layer 3 Gradients
        // dLoss/dW3 = z2.T * dLoss_dOut
        for(let i=0; i<this.W3.rows; i++) {
            for(let j=0; j<this.W3.cols; j++) {
                let sum = 0;
                for(let k=0; k<batchSize; k++) {
                    sum += this.z2.data[k*this.z2.cols + i] * dLoss_dOut.data[k*dLoss_dOut.cols + j];
                }
                this.W3.data[i*this.W3.cols + j] -= lr * (sum / batchSize);
            }
        }
        // Bias 3
        for(let j=0; j<this.b3.cols; j++) {
            let sum = 0;
            for(let k=0; k<batchSize; k++) {
                sum += dLoss_dOut.data[k*dLoss_dOut.cols + j];
            }
            this.b3.data[j] -= lr * (sum / batchSize);
        }

        // Backprop to Hidden 2
        const dLoss_dZ2 = new Tensor(batchSize, this.hiddenSize);
        // dLoss_dZ2 = dLoss_dOut * W3.T * ReLU_deriv
        for(let i=0; i<batchSize; i++) {
            for(let j=0; j<this.W3.rows; j++) {
                let sum = 0;
                for(let k=0; k<this.W3.cols; k++) {
                    sum += dLoss_dOut.data[i*dLoss_dOut.cols + k] * this.W3.data[j*this.W3.cols + k];
                }
                // ReLU derivative
                const val = this.z2.data[i*this.z2.cols + j];
                this.W2.data // Wait, we need to update W2 based on this.
                // Actually, let's simplify backprop for code brevity/speed.
                // We are just doing SGD.
                
                // Correct backprop logic:
                const grad = (val > 0) ? sum : 0;
                
                // Update W2 immediately? No, accumulate.
                // Let's just do the W2 update loop here to save memory/loops
                // dW2 = z1.T * grad
                // This is getting complex for a single file.
                // Let's use a simpler heuristic:
                // Just update weights directly with a simplified rule or use a library?
                // No, user wants "custom ML library" style.
                
                // Let's stick to the plan but maybe simplify the network to 2 layers if 3 is too slow.
                // Or just implement standard backprop correctly.
            }
        }
        
        // RE-IMPLEMENTING BACKPROP SIMPLY
        // We need to store gradients or update in place.
        // Let's update in place (SGD).
        
        // Propagate error to Z2
        const dZ2 = new Tensor(batchSize, this.W2.cols);
        for(let i=0; i<batchSize; i++) {
            for(let j=0; j<this.W2.cols; j++) { // Hidden size
                let sum = 0;
                for(let k=0; k<outputDim; k++) {
                    sum += dLoss_dOut.data[i*outputDim + k] * this.W3.data[j*outputDim + k];
                }
                dZ2.data[i*this.W2.cols + j] = (this.z2.data[i*this.W2.cols + j] > 0) ? sum : 0;
            }
        }
        
        // Update W3, b3
        for(let i=0; i<batchSize; i++) {
            for(let j=0; j<this.W3.rows; j++) {
                for(let k=0; k<this.W3.cols; k++) {
                    this.W3.data[j*this.W3.cols + k] -= lr * this.z2.data[i*this.z2.cols + j] * dLoss_dOut.data[i*outputDim + k] / batchSize;
                }
            }
            for(let k=0; k<this.W3.cols; k++) {
                this.b3.data[k] -= lr * dLoss_dOut.data[i*outputDim + k] / batchSize;
            }
        }
        
        // Propagate error to Z1
        const dZ1 = new Tensor(batchSize, this.W1.cols);
        for(let i=0; i<batchSize; i++) {
            for(let j=0; j<this.W1.cols; j++) {
                let sum = 0;
                for(let k=0; k<this.W2.cols; k++) {
                    sum += dZ2.data[i*this.W2.cols + k] * this.W2.data[j*this.W2.cols + k];
                }
                dZ1.data[i*this.W1.cols + j] = (this.z1.data[i*this.W1.cols + j] > 0) ? sum : 0;
            }
        }

        // Update W2, b2
        for(let i=0; i<batchSize; i++) {
            for(let j=0; j<this.W2.rows; j++) {
                for(let k=0; k<this.W2.cols; k++) {
                    this.W2.data[j*this.W2.cols + k] -= lr * this.z1.data[i*this.z1.cols + j] * dZ2.data[i*this.W2.cols + k] / batchSize;
                }
            }
            for(let k=0; k<this.W2.cols; k++) {
                this.b2.data[k] -= lr * dZ2.data[i*this.W2.cols + k] / batchSize;
            }
        }

        // Update W1, b1
        for(let i=0; i<batchSize; i++) {
            for(let j=0; j<this.W1.rows; j++) {
                for(let k=0; k<this.W1.cols; k++) {
                    this.W1.data[j*this.W1.cols + k] -= lr * this.inputs.data[i*this.inputs.cols + j] * dZ1.data[i*this.W1.cols + k] / batchSize;
                }
            }
            for(let k=0; k<this.W1.cols; k++) {
                this.b1.data[k] -= lr * dZ1.data[i*this.W1.cols + k] / batchSize;
            }
        }

        return totalLoss;
    }
}

// --- Application Logic ---

const canvas = document.getElementById('nn-canvas');
const ctx = canvas.getContext('2d');
const lossCanvas = document.getElementById('loss-canvas');
const lossCtx = lossCanvas.getContext('2d');

let mlp;
let fixedInput;
let targetPoints = [];
let lossHistory = [];
let epoch = 0;
let animationId;

// Resize handling
function resize() {
    const container = canvas.parentElement;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    
    lossCanvas.width = lossCanvas.parentElement.clientWidth;
    lossCanvas.height = lossCanvas.parentElement.clientHeight - 40; // minus header
    
    initTargets();
}

window.addEventListener('resize', resize);

// Generate Target Points from Text
function initTargets() {
    // Create offscreen canvas
    const osc = document.createElement('canvas');
    osc.width = canvas.width;
    osc.height = canvas.height;
    const osctx = osc.getContext('2d');
    
    osctx.font = CONFIG.font;
    osctx.fillStyle = 'black';
    osctx.textAlign = 'center';
    osctx.textBaseline = 'middle';
    osctx.fillText(CONFIG.targetText, osc.width/2, osc.height/2);
    
    const imgData = osctx.getImageData(0, 0, osc.width, osc.height);
    const data = imgData.data;
    
    targetPoints = [];
    // Sample points
    for(let y=0; y<osc.height; y+=2) {
        for(let x=0; x<osc.width; x+=2) {
            if(data[(y*osc.width + x)*4 + 3] > 128) {
                // Normalize to [-1, 1]
                targetPoints.push({
                    x: (x / osc.width) * 2 - 1,
                    y: (y / osc.height) * 2 - 1
                });
            }
        }
    }
    
    // Downsample if too many
    if(targetPoints.length > CONFIG.particleCount) {
        targetPoints = targetPoints.sort(() => 0.5 - Math.random()).slice(0, CONFIG.particleCount);
    }
    
    // Restart training
    initNetwork();
}

function initNetwork() {
    // Input: Random noise fixed for each particle
    // We want the network to map FixedNoise[i] -> Target[i] (conceptually)
    // But since we use Chamfer loss, it maps FixedNoise[i] -> Any Target Point
    
    const inputSize = 4; // x, y, r, theta (polar coords of initial random pos)
    fixedInput = new Tensor(CONFIG.particleCount, inputSize);
    
    for(let i=0; i<CONFIG.particleCount; i++) {
        const x = MathUtils.random(-1, 1);
        const y = MathUtils.random(-1, 1);
        fixedInput.data[i*4] = x;
        fixedInput.data[i*4+1] = y;
        fixedInput.data[i*4+2] = Math.sqrt(x*x + y*y);
        fixedInput.data[i*4+3] = Math.atan2(y, x);
    }
    
    mlp = new MLP(inputSize, CONFIG.hiddenSize, 2);
    lossHistory = [];
    epoch = 0;
}

function drawLoss() {
    const w = lossCanvas.width;
    const h = lossCanvas.height;
    lossCtx.clearRect(0, 0, w, h);
    
    if(lossHistory.length < 2) return;
    
    lossCtx.beginPath();
    lossCtx.strokeStyle = '#D2691E'; // Primary color
    lossCtx.lineWidth = 2;
    
    const maxLoss = Math.max(...lossHistory);
    const minLoss = 0;
    
    for(let i=0; i<lossHistory.length; i++) {
        const x = (i / (lossHistory.length - 1)) * w;
        const y = h - ((lossHistory[i] - minLoss) / (maxLoss - minLoss || 1)) * (h * 0.8) - 10;
        if(i===0) lossCtx.moveTo(x, y);
        else lossCtx.lineTo(x, y);
    }
    lossCtx.stroke();
}

function animate() {
    // Train Step
    const loss = mlp.trainStep(targetPoints, CONFIG.learningRate);
    lossHistory.push(loss);
    if(lossHistory.length > 200) lossHistory.shift();
    
    epoch++;
    document.getElementById('epoch-counter').innerText = epoch;
    document.getElementById('loss-value').innerText = loss.toFixed(4);
    
    // Render Network Output
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const output = mlp.output;
    ctx.fillStyle = '#D2691E';
    
    for(let i=0; i<output.rows; i++) {
        // Denormalize
        const x = (output.data[i*2] + 1) / 2 * canvas.width;
        const y = (output.data[i*2+1] + 1) / 2 * canvas.height;
        
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI*2);
        ctx.fill();
    }
    
    drawLoss();
    
    // Decay learning rate slightly or keep constant?
    // CONFIG.learningRate *= 0.999;
    
    animationId = requestAnimationFrame(animate);
}

// Init
resize();
document.getElementById('retrain-btn').addEventListener('click', initNetwork);
animate();

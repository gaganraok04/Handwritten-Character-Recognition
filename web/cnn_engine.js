/**
 * Client-Side Neural Network Inference Engine
 * Implements Conv2D, MaxPool2D, ReLU, Linear, and Softmax in pure JavaScript.
 */

// 62-class EMNIST ByClass labels
const EMNIST_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
];

const MNIST_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

class NeuralInferenceEngine {
    constructor() {
        this.emnistWeights = null;
        this.emnistReady = false;
    }

    async loadEmnistBinary(url = 'web/weights/emnist_deep_cnn.bin') {
        try {
            const resp = await fetch(url);
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            const buf = await resp.arrayBuffer();
            const f32 = new Float32Array(buf);

            let offset = 0;
            const w1 = f32.subarray(offset, offset + 32 * 1 * 3 * 3); offset += 32 * 1 * 3 * 3;
            const b1 = f32.subarray(offset, offset + 32); offset += 32;

            const w2 = f32.subarray(offset, offset + 64 * 32 * 3 * 3); offset += 64 * 32 * 3 * 3;
            const b2 = f32.subarray(offset, offset + 64); offset += 64;

            const fc1_w = f32.subarray(offset, offset + 128 * 3136); offset += 128 * 3136;
            const fc1_b = f32.subarray(offset, offset + 128); offset += 128;

            const fc2_w = f32.subarray(offset, offset + 62 * 128); offset += 62 * 128;
            const fc2_b = f32.subarray(offset, offset + 62); offset += 62;

            this.emnistWeights = { w1, b1, w2, b2, fc1_w, fc1_b, fc2_w, fc2_b };
            this.emnistReady = true;
            return true;
        } catch (e) {
            console.warn('Failed to load EMNIST binary weights:', e);
            return false;
        }
    }

    /**
     * Run SimpleCNN (MNIST Digits 0-9)
     * Input: 28x28 normalized Float32Array
     */
    predictSimpleCNN(input28x28, weights = window.MNIST_WEIGHTS) {
        if (!weights) throw new Error("MNIST_WEIGHTS not loaded.");
        const t0 = performance.now();

        // 1. Conv2D: 1x28x28 -> 16x26x26 (kernel 3x3, pad 0, stride 1)
        const c1_w = weights.conv1_w;
        const c1_b = weights.conv1_b;
        const out_c1 = new Float32Array(16 * 26 * 26);

        for (let oc = 0; oc < 16; oc++) {
            const b = c1_b[oc];
            const w_base = oc * 9;
            for (let y = 0; y < 26; y++) {
                for (let x = 0; x < 26; x++) {
                    let sum = b;
                    for (let ky = 0; ky < 3; ky++) {
                        const in_row = (y + ky) * 28;
                        const k_row = w_base + ky * 3;
                        for (let kx = 0; kx < 3; kx++) {
                            sum += input28x28[in_row + (x + kx)] * c1_w[k_row + kx];
                        }
                    }
                    // ReLU
                    out_c1[oc * 676 + y * 26 + x] = sum > 0 ? sum : 0;
                }
            }
        }

        // 2. MaxPool2D(2, 2): 16x26x26 -> 16x13x13 (flattened length: 2704)
        const pooled = new Float32Array(16 * 13 * 13);
        for (let oc = 0; oc < 16; oc++) {
            const in_ch_offset = oc * 676;
            const out_ch_offset = oc * 169;
            for (let py = 0; py < 13; py++) {
                const in_y = py * 2;
                for (let px = 0; px < 13; px++) {
                    const in_x = px * 2;
                    const idx = in_ch_offset + in_y * 26 + in_x;
                    const m = Math.max(
                        out_c1[idx],
                        out_c1[idx + 1],
                        out_c1[idx + 26],
                        out_c1[idx + 27]
                    );
                    pooled[out_ch_offset + py * 13 + px] = m;
                }
            }
        }

        // 3. Dense (FC): 2704 -> 10
        const fc_w = weights.fc_w;
        const fc_b = weights.fc_b;
        const logits = new Float32Array(10);
        for (let c = 0; c < 10; c++) {
            let sum = fc_b[c];
            const row_offset = c * 2704;
            for (let i = 0; i < 2704; i++) {
                sum += pooled[i] * fc_w[row_offset + i];
            }
            logits[c] = sum;
        }

        const probs = this.softmax(logits);
        const t1 = performance.now();
        return this.formatResults(probs, MNIST_CLASSES, t1 - t0);
    }

    /**
     * Run DeepCNN (EMNIST 62-Classes) with Folded BatchNorm
     */
    predictDeepCNN(input28x28) {
        if (!this.emnistReady) throw new Error("EMNIST model weights not loaded yet.");
        const t0 = performance.now();
        const { w1, b1, w2, b2, fc1_w, fc1_b, fc2_w, fc2_b } = this.emnistWeights;

        // Block 1: Conv2D (1->32, 3x3, pad 1) + ReLU -> 32x28x28
        const out_c1 = new Float32Array(32 * 28 * 28);
        for (let oc = 0; oc < 32; oc++) {
            const b = b1[oc];
            const w_base = oc * 9;
            const oc_offset = oc * 784;
            for (let y = 0; y < 28; y++) {
                for (let x = 0; x < 28; x++) {
                    let sum = b;
                    for (let ky = 0; ky < 3; ky++) {
                        const iy = y + ky - 1;
                        if (iy < 0 || iy >= 28) continue;
                        const in_row = iy * 28;
                        const k_row = w_base + ky * 3;
                        for (let kx = 0; kx < 3; kx++) {
                            const ix = x + kx - 1;
                            if (ix < 0 || ix >= 28) continue;
                            sum += input28x28[in_row + ix] * w1[k_row + kx];
                        }
                    }
                    out_c1[oc_offset + y * 28 + x] = sum > 0 ? sum : 0;
                }
            }
        }

        // MaxPool2D(2, 2): 32x28x28 -> 32x14x14
        const p1 = new Float32Array(32 * 14 * 14);
        for (let oc = 0; oc < 32; oc++) {
            const in_ch = oc * 784;
            const out_ch = oc * 196;
            for (let py = 0; py < 14; py++) {
                const in_y = py * 2;
                for (let px = 0; px < 14; px++) {
                    const in_x = px * 2;
                    const idx = in_ch + in_y * 28 + in_x;
                    p1[out_ch + py * 14 + px] = Math.max(
                        out_c1[idx],
                        out_c1[idx + 1],
                        out_c1[idx + 28],
                        out_c1[idx + 29]
                    );
                }
            }
        }

        // Block 2: Conv2D (32->64, 3x3, pad 1) + ReLU -> 64x14x14
        const out_c2 = new Float32Array(64 * 14 * 14);
        for (let oc = 0; oc < 64; oc++) {
            const b = b2[oc];
            const oc_offset = oc * 196;
            const oc_w_base = oc * 32 * 9;
            for (let y = 0; y < 14; y++) {
                for (let x = 0; x < 14; x++) {
                    let sum = b;
                    for (let ic = 0; ic < 32; ic++) {
                        const in_ch_offset = ic * 196;
                        const k_ch_offset = oc_w_base + ic * 9;
                        for (let ky = 0; ky < 3; ky++) {
                            const iy = y + ky - 1;
                            if (iy < 0 || iy >= 14) continue;
                            const in_row = in_ch_offset + iy * 14;
                            const k_row = k_ch_offset + ky * 3;
                            for (let kx = 0; kx < 3; kx++) {
                                const ix = x + kx - 1;
                                if (ix < 0 || ix >= 14) continue;
                                sum += p1[in_row + ix] * w2[k_row + kx];
                            }
                        }
                    }
                    out_c2[oc_offset + y * 14 + x] = sum > 0 ? sum : 0;
                }
            }
        }

        // MaxPool2D(2, 2): 64x14x14 -> 64x7x7 (length 3136)
        const p2 = new Float32Array(64 * 49);
        for (let oc = 0; oc < 64; oc++) {
            const in_ch = oc * 196;
            const out_ch = oc * 49;
            for (let py = 0; py < 7; py++) {
                const in_y = py * 2;
                for (let px = 0; px < 7; px++) {
                    const in_x = px * 2;
                    const idx = in_ch + in_y * 14 + in_x;
                    p2[out_ch + py * 7 + px] = Math.max(
                        out_c2[idx],
                        out_c2[idx + 1],
                        out_c2[idx + 14],
                        out_c2[idx + 15]
                    );
                }
            }
        }

        // FC1: 3136 -> 128 + ReLU
        const fc1_out = new Float32Array(128);
        for (let i = 0; i < 128; i++) {
            let sum = fc1_b[i];
            const w_row = i * 3136;
            for (let j = 0; j < 3136; j++) {
                sum += p2[j] * fc1_w[w_row + j];
            }
            fc1_out[i] = sum > 0 ? sum : 0;
        }

        // FC2: 128 -> 62
        const logits = new Float32Array(62);
        for (let c = 0; c < 62; c++) {
            let sum = fc2_b[c];
            const w_row = c * 128;
            for (let j = 0; j < 128; j++) {
                sum += fc1_out[j] * fc2_w[w_row + j];
            }
            logits[c] = sum;
        }

        const probs = this.softmax(logits);
        const t1 = performance.now();
        return this.formatResults(probs, EMNIST_CLASSES, t1 - t0);
    }

    softmax(arr) {
        let maxVal = -Infinity;
        for (let i = 0; i < arr.length; i++) {
            if (arr[i] > maxVal) maxVal = arr[i];
        }
        let sum = 0;
        const exps = new Float32Array(arr.length);
        for (let i = 0; i < arr.length; i++) {
            const exp = Math.exp(arr[i] - maxVal);
            exps[i] = exp;
            sum += exp;
        }
        for (let i = 0; i < arr.length; i++) {
            exps[i] /= sum;
        }
        return exps;
    }

    formatResults(probs, classLabels, latencyMs) {
        const indexed = [];
        for (let i = 0; i < probs.length; i++) {
            indexed.push({ label: classLabels[i], prob: probs[i], index: i });
        }
        indexed.sort((a, b) => b.prob - a.prob);

        const top1 = indexed[0];
        const top5 = indexed.slice(0, 5);

        return {
            predicted: top1.label,
            confidence: (top1.prob * 100).toFixed(2),
            latency: latencyMs.toFixed(2),
            top5: top5.map(item => ({
                label: item.label,
                percent: (item.prob * 100).toFixed(1),
                rawProb: item.prob
            }))
        };
    }
}

window.NeuralInferenceEngine = NeuralInferenceEngine;


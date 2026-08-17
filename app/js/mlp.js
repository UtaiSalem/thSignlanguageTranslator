/**
 * โครงข่ายประสาทเทียมขนาดเล็ก เขียนด้วย JavaScript ล้วน
 *
 * ทำไมไม่ใช้ TensorFlow.js: ไลบรารีนั้นมีขนาดหลายเมกะไบต์และเกินความจำเป็นมาก
 * สำหรับปัญหานี้ input มีแค่ 128 ค่าเพราะ features.js กลั่นมาให้ดีแล้ว
 * โมเดลจึงเล็กพอที่จะเขียนเองทั้งหมดและเทรนบนมือถือได้ในไม่กี่วินาที
 *
 * โครงสร้าง:  128 -> [ReLU 64] -> softmax(จำนวนคำ)
 * ตัวปรับพารามิเตอร์: Adam  ฟังก์ชันสูญเสีย: cross-entropy
 */

const EPS = 1e-8;

/** สุ่มตัวเลขแบบทำซ้ำได้ เพื่อให้เทรนสองครั้งได้ผลเหมือนกัน */
function makeRandom(seed) {
  let state = seed >>> 0;
  return function random() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** สุ่มแบบการแจกแจงปกติ (Box-Muller) ใช้ตั้งค่าน้ำหนักเริ่มต้น */
function randomNormal(random) {
  let u = 0;
  let v = 0;
  while (u === 0) u = random();
  while (v === 0) v = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

class Dense {
  constructor(inputDim, outputDim, random) {
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.weights = new Float32Array(inputDim * outputDim);
    this.bias = new Float32Array(outputDim);

    // He initialization — เหมาะกับ ReLU ช่วยไม่ให้สัญญาณจางหายระหว่างชั้น
    const stddev = Math.sqrt(2 / inputDim);
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = randomNormal(random) * stddev;
    }

    // สถานะของ Adam
    this.mW = new Float32Array(this.weights.length);
    this.vW = new Float32Array(this.weights.length);
    this.mB = new Float32Array(outputDim);
    this.vB = new Float32Array(outputDim);
    this.gradW = new Float32Array(this.weights.length);
    this.gradB = new Float32Array(outputDim);
  }

  forward(input, output) {
    const { inputDim, outputDim, weights, bias } = this;
    for (let j = 0; j < outputDim; j++) {
      let sum = bias[j];
      for (let i = 0; i < inputDim; i++) sum += input[i] * weights[i * outputDim + j];
      output[j] = sum;
    }
  }

  /** สะสม gradient ของหนึ่งตัวอย่าง และคืน gradient ที่ส่งย้อนไปชั้นก่อนหน้า */
  backward(input, gradOutput, gradInput) {
    const { inputDim, outputDim, weights, gradW, gradB } = this;
    if (gradInput) gradInput.fill(0);

    for (let j = 0; j < outputDim; j++) {
      const g = gradOutput[j];
      if (g === 0) continue;
      gradB[j] += g;
      for (let i = 0; i < inputDim; i++) {
        gradW[i * outputDim + j] += input[i] * g;
        if (gradInput) gradInput[i] += weights[i * outputDim + j] * g;
      }
    }
  }

  applyGradients(learningRate, batchSize, step) {
    const beta1 = 0.9;
    const beta2 = 0.999;
    const correction1 = 1 - Math.pow(beta1, step);
    const correction2 = 1 - Math.pow(beta2, step);

    const update = (params, grads, m, v) => {
      for (let i = 0; i < params.length; i++) {
        const g = grads[i] / batchSize;
        m[i] = beta1 * m[i] + (1 - beta1) * g;
        v[i] = beta2 * v[i] + (1 - beta2) * g * g;
        const mHat = m[i] / correction1;
        const vHat = v[i] / correction2;
        params[i] -= (learningRate * mHat) / (Math.sqrt(vHat) + EPS);
        grads[i] = 0;
      }
    };

    update(this.weights, this.gradW, this.mW, this.vW);
    update(this.bias, this.gradB, this.mB, this.vB);
  }

  toJSON() {
    return {
      inputDim: this.inputDim,
      outputDim: this.outputDim,
      weights: Array.from(this.weights),
      bias: Array.from(this.bias),
    };
  }

  static fromJSON(data) {
    const layer = Object.create(Dense.prototype);
    layer.inputDim = data.inputDim;
    layer.outputDim = data.outputDim;
    layer.weights = Float32Array.from(data.weights);
    layer.bias = Float32Array.from(data.bias);
    return layer;
  }
}

export class Classifier {
  constructor(labels, inputDim, hiddenDim = 64, seed = 42) {
    this.labels = labels;
    this.inputDim = inputDim;
    this.hiddenDim = hiddenDim;
    this.mean = new Float32Array(inputDim);
    this.std = new Float32Array(inputDim).fill(1);
    this.accuracy = null;
    this.trainedAt = null;

    const random = makeRandom(seed);
    this.hidden = new Dense(inputDim, hiddenDim, random);
    this.output = new Dense(hiddenDim, labels.length, random);

    this._h = new Float32Array(hiddenDim);
    this._hAct = new Float32Array(hiddenDim);
    this._logits = new Float32Array(labels.length);
    this._probs = new Float32Array(labels.length);
    this._scaled = new Float32Array(inputDim);
  }

  /** ปรับค่าให้แต่ละ feature มีค่าเฉลี่ย 0 และส่วนเบี่ยงเบน 1 (เทียบเท่า StandardScaler) */
  _scale(vector, out) {
    for (let i = 0; i < this.inputDim; i++) {
      out[i] = (vector[i] - this.mean[i]) / this.std[i];
    }
    return out;
  }

  _forward(vector) {
    const scaled = this._scale(vector, this._scaled);
    this.hidden.forward(scaled, this._h);
    for (let i = 0; i < this.hiddenDim; i++) {
      this._hAct[i] = this._h[i] > 0 ? this._h[i] : 0; // ReLU
    }
    this.output.forward(this._hAct, this._logits);

    // softmax แบบลบค่าสูงสุดก่อน เพื่อไม่ให้ exp ระเบิดเป็น Infinity
    let max = -Infinity;
    for (let i = 0; i < this._logits.length; i++) {
      if (this._logits[i] > max) max = this._logits[i];
    }
    let sum = 0;
    for (let i = 0; i < this._logits.length; i++) {
      this._probs[i] = Math.exp(this._logits[i] - max);
      sum += this._probs[i];
    }
    for (let i = 0; i < this._probs.length; i++) this._probs[i] /= sum;

    return { scaled, probs: this._probs };
  }

  /** ทำนายคำจากเวกเตอร์ 128 ค่า */
  predict(vector) {
    const { probs } = this._forward(vector);
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return {
      label: this.labels[best],
      confidence: probs[best],
      probabilities: Array.from(probs),
    };
  }

  _fitScaler(X) {
    const n = X.length;
    this.mean.fill(0);
    this.std.fill(0);

    for (const row of X) {
      for (let i = 0; i < this.inputDim; i++) this.mean[i] += row[i];
    }
    for (let i = 0; i < this.inputDim; i++) this.mean[i] /= n;

    for (const row of X) {
      for (let i = 0; i < this.inputDim; i++) {
        const d = row[i] - this.mean[i];
        this.std[i] += d * d;
      }
    }
    for (let i = 0; i < this.inputDim; i++) {
      const variance = this.std[i] / n;
      // feature ที่ไม่เปลี่ยนค่าเลยจะมีส่วนเบี่ยงเบนเป็น 0 ต้องกันหารด้วยศูนย์
      this.std[i] = Math.sqrt(variance) || 1;
    }
  }

  /**
   * เทรนโมเดล — เป็น generator เพื่อให้หน้าจอไม่ค้าง
   * เรียกใช้แบบ: for (const progress of model.trainSteps(...)) { await yieldToUI(); }
   *
   * ส่งข้อมูล **ดิบ** เข้ามา แล้วบอกวิธีเพิ่มข้อมูลผ่าน options.augment
   * (อย่าเพิ่มข้อมูลมาก่อนแล้วส่งเข้ามา — เหตุผลอยู่ที่คอมเมนต์ตรงจุดที่แบ่งชุดข้อมูล)
   */
  *trainSteps(X, y, options = {}) {
    const {
      epochs = 120,
      batchSize = 32,
      learningRate = 0.01,
      validationSplit = 0.2,
      patience = 20,
      seed = 42,
      augment = null,
    } = options;

    // ต้องมีอย่างน้อย 2 ตัวอย่าง ไม่งั้นแบ่งชุดตรวจสอบแล้วชุดเทรนจะว่าง
    // ซึ่งทำให้ตัวปรับสเกลหารด้วยศูนย์ กลายเป็น NaN ทั้งโมเดลแบบไม่มี error ฟ้อง
    if (X.length < 2) {
      throw new Error("ต้องมีอย่างน้อย 2 ตัวอย่างถึงจะเทรนได้");
    }

    const random = makeRandom(seed);
    const indices = Array.from({ length: X.length }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    // กันไว้ทั้งสองด้าน: ชุดตรวจสอบต้องมีอย่างน้อย 1 ตัวอย่าง (ไม่งั้นคำนวณ
    // ความแม่นยำเป็น 0/0) และต้องเหลือให้ชุดเทรนอย่างน้อย 1 ตัวอย่างด้วย
    const validationCount = Math.min(
      Math.max(1, Math.floor(X.length * validationSplit)),
      X.length - 1
    );
    const validationX = [];
    const validationY = [];
    for (const index of indices.slice(0, validationCount)) {
      validationX.push(X[index]);
      validationY.push(y[index]);
    }

    const trainX = [];
    const trainY = [];
    for (const index of indices.slice(validationCount)) {
      trainX.push(X[index]);
      trainY.push(y[index]);
    }

    // เพิ่มข้อมูล **หลัง** แบ่งชุดตรวจสอบออกไปแล้วเท่านั้น ไม่งั้นภาพกลับด้าน
    // ของตัวอย่างเดียวกันจะไปโผล่ทั้งในชุดเทรนและชุดตรวจสอบ ความแม่นยำที่รายงาน
    // จะสูงเกินจริง (data leakage) — ฝั่งคอมพิวเตอร์ระวังเรื่องนี้ที่ src/train.py
    if (augment) {
      const originalCount = trainX.length;
      for (let i = 0; i < originalCount; i++) {
        trainX.push(augment(trainX[i]));
        trainY.push(trainY[i]);
      }
    }

    // ปรับสเกลจากชุดเทรนเท่านั้น ค่าเฉลี่ยและส่วนเบี่ยงเบนของชุดตรวจสอบ
    // ต้องไม่รั่วเข้ามา ด้วยเหตุผลเดียวกับข้างบน
    this._fitScaler(trainX);

    const trainIndices = Array.from({ length: trainX.length }, (_, i) => i);
    const gradHidden = new Float32Array(this.hiddenDim);
    const gradLogits = new Float32Array(this.labels.length);
    let step = 0;
    let bestAccuracy = -1;
    let bestSnapshot = null;
    let sinceImprovement = 0;

    for (let epoch = 1; epoch <= epochs; epoch++) {
      for (let i = trainIndices.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [trainIndices[i], trainIndices[j]] = [trainIndices[j], trainIndices[i]];
      }

      let lossSum = 0;
      for (let start = 0; start < trainIndices.length; start += batchSize) {
        const batch = trainIndices.slice(start, start + batchSize);

        for (const index of batch) {
          const { scaled, probs } = this._forward(trainX[index]);
          const target = trainY[index];
          lossSum -= Math.log(Math.max(probs[target], EPS));

          // อนุพันธ์ของ cross-entropy รวมกับ softmax ย่อเหลือ (p - onehot)
          for (let k = 0; k < probs.length; k++) gradLogits[k] = probs[k];
          gradLogits[target] -= 1;

          this.output.backward(this._hAct, gradLogits, gradHidden);
          for (let k = 0; k < this.hiddenDim; k++) {
            if (this._h[k] <= 0) gradHidden[k] = 0; // อนุพันธ์ของ ReLU
          }
          this.hidden.backward(scaled, gradHidden, null);
        }

        step++;
        this.hidden.applyGradients(learningRate, batch.length, step);
        this.output.applyGradients(learningRate, batch.length, step);
      }

      let correct = 0;
      for (let i = 0; i < validationX.length; i++) {
        const { probs } = this._forward(validationX[i]);
        let best = 0;
        for (let k = 1; k < probs.length; k++) if (probs[k] > probs[best]) best = k;
        if (best === validationY[i]) correct++;
      }
      const accuracy = correct / validationX.length;

      if (accuracy > bestAccuracy) {
        bestAccuracy = accuracy;
        bestSnapshot = {
          hidden: this.hidden.toJSON(),
          output: this.output.toJSON(),
        };
        sinceImprovement = 0;
      } else {
        sinceImprovement++;
      }

      yield {
        epoch,
        epochs,
        loss: lossSum / trainIndices.length,
        accuracy,
        bestAccuracy,
      };

      // หยุดก่อนกำหนดถ้าไม่ดีขึ้นแล้ว ประหยัดเวลาและกัน overfitting
      if (sinceImprovement >= patience) break;
    }

    if (bestSnapshot) {
      this.hidden = Dense.fromJSON(bestSnapshot.hidden);
      this.output = Dense.fromJSON(bestSnapshot.output);
    }
    this.accuracy = bestAccuracy;
    this.trainedAt = new Date().toISOString();
  }

  toJSON() {
    return {
      version: 1,
      labels: this.labels,
      inputDim: this.inputDim,
      hiddenDim: this.hiddenDim,
      mean: Array.from(this.mean),
      std: Array.from(this.std),
      hidden: this.hidden.toJSON(),
      output: this.output.toJSON(),
      accuracy: this.accuracy,
      trainedAt: this.trainedAt,
    };
  }

  static fromJSON(data) {
    const model = new Classifier(data.labels, data.inputDim, data.hiddenDim);
    model.mean = Float32Array.from(data.mean);
    model.std = Float32Array.from(data.std);
    model.hidden = Dense.fromJSON(data.hidden);
    model.output = Dense.fromJSON(data.output);
    model.accuracy = data.accuracy;
    model.trainedAt = data.trainedAt;
    return model;
  }
}

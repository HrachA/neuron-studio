"use strict";
const NeuralLab = (() => {
  const sizes = [2, 4, 3, 1];
  function random(seed) {
    return () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296;
    };
  }
  function dataset(name) {
    const rand = random(83);
    return Array.from({length: 160}, (_, i) => {
      const label = i % 2;
      if (name === "Rings") {
        const angle = rand() * Math.PI * 2;
        const radius = (label ? .75 : .3) + (rand() - .5) * .22;
        return {point: [Math.cos(angle)*radius, Math.sin(angle)*radius], label};
      }
      if (name === "Clusters") {
        return {point: [0, 1].map(() => (label ? .46 : -.46) + (rand() + rand() - 1) * .49), label};
      }
      const x = (rand() * .82 + .08) * (i % 4 < 2 ? 1 : -1);
      const y = (rand() * .82 + .08) * Math.sign(x) * (label ? -1 : 1);
      return {point: [x,y], label};
    });
  }
  class Model {
    constructor(activation) {
      this.activation = activation;
      this.rand = random(274);
      this.layers = sizes.slice(1).map((size, layer) => Array.from({length:size}, () =>
        Array.from({length:sizes[layer]+1}, (_, k) => ({
          w: k === sizes[layer] ? 0 : (this.rand()*2-1)*Math.sqrt(6/(size+sizes[layer])),
          g:0, m:0, v:0
        }))));
      this.steps = 0;
    }
    forward(point) {
      const values = [point];
      for (const [l, layer] of this.layers.entries()) {
        const input = values[l];
        values.push(layer.map(weights => {
          let sum = weights[input.length].w;
          for (let k=0; k<input.length; k++) sum += weights[k].w * input[k];
          if (l === this.layers.length-1) return 1/(1+Math.exp(-sum));
          return this.activation === "relu" ? Math.max(0,sum) : Math.tanh(sum);
        }));
      }
      return values;
    }
    epoch(data, rate) {
      const shuffled = data.slice();
      for (let i=shuffled.length-1; i>0; i--) {
        const j = Math.floor(this.rand()*(i+1));
        [shuffled[i],shuffled[j]] = [shuffled[j],shuffled[i]];
      }
      for (let start=0; start<data.length; start+=16) {
        const batch = shuffled.slice(start,start+16);
        for (const {point,label} of batch) {
          const a = this.forward(point);
          let delta = [a.at(-1)[0]-label];
          for (let l=this.layers.length-1; l>=0; l--) {
            const prev = a[l].map(() => 0);
            this.layers[l].forEach((weights,j) => {
              a[l].forEach((value,k) => {
                weights[k].g += delta[j]*value;
                prev[k] += delta[j]*weights[k].w;
              });
              weights.at(-1).g += delta[j];
            });
            delta = prev.map((v,k) => v*(this.activation === "relu" ? Number(a[l][k]>0) : 1-a[l][k]**2));
          }
        }
        this.steps++;
        const mScale = 1-.9**this.steps, vScale = 1-.999**this.steps;
        for (const layer of this.layers) for (const neuron of layer) for (const weight of neuron) {
          const g = weight.g / batch.length;
          weight.m = .9*weight.m + .1*g;
          weight.v = .999*weight.v + .001*g*g;
          weight.w -= rate*(weight.m/mScale)/(Math.sqrt(weight.v/vScale)+1e-8);
          weight.g = 0;
        }
      }
    }
    metrics(data) {
      let correct=0, loss=0;
      for (const {point,label} of data) {
        const p = this.forward(point).at(-1)[0];
        correct += Number((p>=.5) === Boolean(label));
        loss -= Math.log(Math.max(1e-12,label ? p : 1-p));
      }
      return {accuracy:correct/data.length, loss:loss/data.length};
    }
  }
  return {sizes,dataset,Model};
})();

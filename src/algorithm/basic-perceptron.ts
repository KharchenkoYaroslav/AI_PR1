import { computeSigmoid } from './functions.js';

export class BasicPerceptron {
    weights: number[];
    offset: number;

    constructor(inputCount: number) {
        this.weights = Array.from({ length: inputCount }, () => Math.random() - 0.5);
        this.offset = Math.random() - 0.5;
    }

    fitModel(dataset: { features: number[], target: number }[], iterations: number = 1000, learningStep: number = 0.2): void {
        for (let cycle = 0; cycle < iterations; cycle++) {
            for (const { features, target } of dataset) {
                const currentPrediction = this.predict(features);
                const loss = target - currentPrediction;
                
                for (let i = 0; i < this.weights.length; i++) {
                    const currentWeight = this.weights[i];
                    const currentFeature = features[i];
                    if (currentWeight !== undefined && currentFeature !== undefined) {
                        this.weights[i] = currentWeight + learningStep * loss * currentFeature;
                    }
                }
                this.offset += learningStep * loss;
            }
        }
    }

    predict(features: number[]): number {
        let accumulation = 0;
        for (let i = 0; i < this.weights.length; i++) {
            const currentWeight = this.weights[i];
            const currentFeature = features[i];
            if (currentWeight !== undefined && currentFeature !== undefined) {
                accumulation += currentWeight * currentFeature;
            }
        }
        accumulation += this.offset;
        return computeSigmoid(accumulation);
    }
}
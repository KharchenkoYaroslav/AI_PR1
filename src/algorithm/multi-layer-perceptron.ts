import { computeSigmoid, computeSigmoidDerivative } from './functions.js';

export class MultiLayerPerceptron {
    inputNodes: number;
    hiddenNodes: number;
    learningStep: number;
    epochLimit: number;
    minError: number;

    weightsInHidden: number[][];
    weightsHiddenOut: number[];
    offsetHidden: number[];
    offsetOut: number;

    hiddenState: number[] = [];
    lastPrediction: number = 0;

    constructor(inputNodes: number, hiddenNodes: number, learningStep: number = 0.1, epochLimit: number = 10000, minError: number = 0.001) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.learningStep = learningStep;
        this.epochLimit = epochLimit;
        this.minError = minError;

        this.weightsInHidden = Array.from({ length: hiddenNodes }, () =>
            Array.from({ length: inputNodes }, () => Math.random() - 0.5)
        );
        this.weightsHiddenOut = Array.from({ length: hiddenNodes }, () => Math.random() - 0.5);
        this.offsetHidden = Array.from({ length: hiddenNodes }, () => Math.random() - 0.5);
        this.offsetOut = Math.random() - 0.5;
    }

    calculateOutput(features: number[]): number {
        this.hiddenState = [];
        for (let hIdx = 0; hIdx < this.hiddenNodes; hIdx++) {
            let hiddenSum = 0;
            const hiddenWeights = this.weightsInHidden[hIdx]!;
            for (let iIdx = 0; iIdx < this.inputNodes; iIdx++) {
                hiddenSum += features[iIdx]! * hiddenWeights[iIdx]!;
            }
            hiddenSum += this.offsetHidden[hIdx]!;
            this.hiddenState.push(computeSigmoid(hiddenSum));
        }

        let finalSum = 0;
        for (let hIdx = 0; hIdx < this.hiddenNodes; hIdx++) {
            finalSum += this.hiddenState[hIdx]! * this.weightsHiddenOut[hIdx]!;
        }
        finalSum += this.offsetOut;
        this.lastPrediction = computeSigmoid(finalSum);

        return this.lastPrediction;
    }

    fit(featuresSet: number[][], targets: number[]): void {
        for (let epoch = 0; epoch < this.epochLimit; epoch++) {
            let cumulativeLoss = 0;

            for (let sampleIdx = 0; sampleIdx < featuresSet.length; sampleIdx++) {
                const currentFeatures = featuresSet[sampleIdx]!;
                const targetValue = targets[sampleIdx]!;

                const prediction = this.calculateOutput(currentFeatures);

                const loss = targetValue - prediction;
                cumulativeLoss += Math.pow(loss, 2);

                const outputGradient = loss * computeSigmoidDerivative(this.lastPrediction);
                const hiddenGradients: number[] = [];
                for (let hIdx = 0; hIdx < this.hiddenNodes; hIdx++) {
                    hiddenGradients.push(outputGradient * this.weightsHiddenOut[hIdx]! * computeSigmoidDerivative(this.hiddenState[hIdx]!));
                }

                for (let hIdx = 0; hIdx < this.hiddenNodes; hIdx++) {
                    const nodeWeights = this.weightsInHidden[hIdx]!;
                    this.weightsHiddenOut[hIdx]! += this.learningStep * outputGradient * this.hiddenState[hIdx]!;
                    this.offsetHidden[hIdx]! += this.learningStep * hiddenGradients[hIdx]!;
                    for (let iIdx = 0; iIdx < this.inputNodes; iIdx++) {
                        nodeWeights[iIdx]! += this.learningStep * hiddenGradients[hIdx]! * currentFeatures[iIdx]!;
                    }
                }
                this.offsetOut += this.learningStep * outputGradient;
            }

            if (cumulativeLoss < this.minError) {
                break;
            }
        }
    }

    evaluate(testFeatures: number[][], expectedTargets: number[]): void {
        console.log("\nNetwork Evaluation Results");
        for (let sampleIdx = 0; sampleIdx < testFeatures.length; sampleIdx++) {
            const features = testFeatures[sampleIdx]!;
            const predictedVal = Math.round(this.calculateOutput(features));
            console.log(`Signals: [${features.join(', ')}] > Result: ${predictedVal} (Target: ${expectedTargets[sampleIdx]})`);
        }
    }
}
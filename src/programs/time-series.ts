import { scale, reverseScale } from '../algorithm/functions.js';
import { MultiLayerPerceptron } from '../algorithm/multi-layer-perceptron.js';

const fullData = [0.99, 4.72, 1.59, 5.29, 1.53, 5.58, 0.84, 5.79, 0.21, 5.94, 0.42, 5.98, 1.18, 5.55, 0.11];

const featureMatrix = [
    [0.99, 4.72, 1.59], 
    [4.72, 1.59, 5.29], 
    [1.59, 5.29, 1.53], 
    [5.29, 1.53, 5.58], 
    [1.53, 5.58, 0.84], 
    [5.58, 0.84, 5.79], 
    [0.84, 5.79, 0.21], 
    [5.79, 0.21, 5.94], 
    [0.21, 5.94, 0.42], 
    [5.94, 0.42, 5.98]  
];

const targetVector = [5.29, 1.53, 5.58, 0.84, 5.79, 0.21, 5.94, 0.42, 5.98, 1.18];

const testFeatures = [
    [0.42, 5.98, 1.18], 
    [5.98, 1.18, 5.55]  
];

const testTargets = [5.55, 0.11];

const lowerBound = Math.min(...fullData);
const upperBound = Math.max(...fullData);

const scaledInputs = featureMatrix.map(seq => scale(seq, lowerBound, upperBound));
const scaledTargets = scale(targetVector, lowerBound, upperBound);
const scaledTestInputs = testFeatures.map(seq => scale(seq, lowerBound, upperBound));

const predictorNetwork = new MultiLayerPerceptron(3, 2, 0.5, 100000);
predictorNetwork.fit(scaledInputs, scaledTargets);

const scaledForecasts = scaledTestInputs.map(seq => predictorNetwork.calculateOutput(seq));
const actualForecasts = reverseScale(scaledForecasts, lowerBound, upperBound);

console.log("\nNetwork Evaluation Results");
for (let i = 0; i < testFeatures.length; i++) {
    const inputs = testFeatures[i]!;
    const predictedValue = actualForecasts[i]!;
    const realValue = testTargets[i]!;

    console.log(`Signals: [${inputs.join(', ')}] > Result: ${predictedValue.toFixed(2)} (Target: ${realValue})`);
}
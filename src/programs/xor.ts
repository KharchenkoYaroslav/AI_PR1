import { MultiLayerPerceptron } from '../algorithm/multi-layer-perceptron.js';

const featureMatrix = [
    [0, 0], 
    [0, 1], 
    [1, 0], 
    [1, 1]
];
const targetVector = [0, 1, 1, 0];

const mlp = new MultiLayerPerceptron(2, 2, 0.5, 100000);
mlp.fit(featureMatrix, targetVector);

console.log("Evaluation of Logical XOR");
mlp.evaluate(featureMatrix, targetVector);
import { BasicPerceptron } from '../algorithm/basic-perceptron.js';

const trainingSet = [
    { features: [0, 0], target: 0 },
    { features: [0, 1], target: 0 },
    { features: [1, 0], target: 0 },
    { features: [1, 1], target: 1 }
];

const perceptron = new BasicPerceptron(2);
perceptron.fitModel(trainingSet);

console.log("\nNetwork Evaluation Results");
for (const { features, target } of trainingSet) {
    const prediction = perceptron.predict(features);
    console.log(`Signals: [${features.join(', ')}] > Result: ${prediction.toFixed(4)} (Target: ${target})`);
}
import { BasicPerceptron } from '../algorithm/basic-perceptron.js';

const trainData = [
    { features: [0], target: 1 },
    { features: [1], target: 0 }
];

const neuron = new BasicPerceptron(1);
neuron.fitModel(trainData);

console.log("\nNetwork Evaluation Results");
for (const { features, target } of trainData) {
    const output = neuron.predict(features);
    console.log(`Signals: [${features.join(', ')}] > Result: ${output.toFixed(4)} (Target: ${target})`);
}

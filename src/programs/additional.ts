import { BasicPerceptron } from '../algorithm/basic-perceptron.js';

const trainData = [
    { features: [0, 0, 0], target: 1 },
    { features: [0, 1, 0], target: 1 },
    { features: [1, 0, 0], target: 0 },
    { features: [1, 1, 1], target: 1 }
];

const neuron = new BasicPerceptron(3);
neuron.fit(trainData);

console.log("\nNetwork Evaluation Results");
for (const { features, target } of trainData) {
    const output = neuron.calculateOutput(features);
    console.log(`Signals: [${features.join(', ')}] > Result: ${output.toFixed(4)} (Target: ${target})`);
}

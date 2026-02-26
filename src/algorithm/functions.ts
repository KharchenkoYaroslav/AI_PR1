export function computeSigmoid(value: number): number {
    return 1 / (1 + Math.exp(-value));
}

export function computeSigmoidDerivative(activatedValue: number): number {
    return activatedValue * (1 - activatedValue);
}

export function scale(dataset: number[], min: number, max: number): number[] {
    return dataset.map(val => (val - min) / (max - min));
}

export function reverseScale(scaledDataset: number[], min: number, max: number): number[] {
    return scaledDataset.map(val => val * (max - min) + min);
}
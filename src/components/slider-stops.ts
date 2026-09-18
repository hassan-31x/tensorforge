const range = (start: number, end: number, step: number) =>
  Array.from({ length: Math.floor((end - start) / step) + 1 }, (_, i) =>
    Number((start + i * step).toPrecision(12)),
  );

const powersOfTwo = (min: number, max: number) => {
  const values: number[] = [];
  for (let value = 1; value <= max; value *= 2) {
    if (value >= min) values.push(value);
  }
  return values;
};

const headCounts = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 128];
const tokenCounts = [
  ...range(1e9, 10e9, 1e9),
  ...range(20e9, 100e9, 10e9),
  ...range(200e9, 1e12, 100e9),
  ...range(2e12, 20e12, 1e12),
];
const vocabularySizes = [
  ...range(5000, 100000, 5000),
  ...range(110000, 300000, 10000),
];
const batchSizes = [1, 2, 4, 8, 16, 24, 32, 48, 64];
const parallelSizes = [1, 2, 4, 8, 16, 32, 64, 128];
const hourlyCosts = [
  ...range(0, 2, 0.05),
  ...range(2.25, 10, 0.25),
  ...range(11, 20, 1),
];

const stopsByLabel: Record<string, number[]> = {
  "Training tokens": tokenCounts,
  "Available tokens": tokenCounts,
  "Vocabulary size": vocabularySizes,
  Vocabulary: vocabularySizes,
  "Hidden size": [
    ...range(128, 2048, 128),
    ...range(2304, 8192, 256),
    ...range(8704, 16384, 512),
  ],
  Layers: [...range(1, 16, 1), ...range(18, 64, 2), ...range(68, 128, 4)],
  "Attention heads": headCounts,
  "KV heads": headCounts,
  "FFN size": [
    ...range(256, 4096, 256),
    ...range(4608, 16384, 512),
    ...range(17408, 65536, 1024),
  ],
  Experts: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
  "Active experts": [1, 2, 4, 8, 16, 32, 64],
  "Sequence length": powersOfTwo(128, 131072),
  "Special tokens": [
    ...range(0, 32, 1),
    ...range(40, 128, 8),
    ...range(160, 512, 32),
  ],
  "Characters / token": range(1, 8, 0.1),
  "Bytes / token": range(1, 8, 0.1),
  "Average document": [...range(128, 1024, 128), ...range(1280, 4096, 256)],
  Epochs: range(1, 10, 1),
  "Micro batch / GPU": batchSizes,
  "Global batch": powersOfTwo(1, 8192),
  Accumulation: batchSizes,
  "Learning rate": [
    ...range(0.00001, 0.0001, 0.00001),
    ...range(0.00012, 0.001, 0.00002),
    ...range(0.0011, 0.003, 0.0001),
  ],
  "Warmup steps": [
    ...range(0, 1000, 100),
    ...range(1250, 5000, 250),
    ...range(6000, 20000, 1000),
  ],
  "Weight decay": range(0, 0.5, 0.01),
  "Gradient clip": [...range(0, 2, 0.1), ...range(2.5, 10, 0.5)],
  "Checkpoint every": [
    ...range(100, 1000, 100),
    ...range(1250, 5000, 250),
    ...range(6000, 10000, 1000),
  ],
  Dropout: range(0, 0.5, 0.01),
  VRAM: [4, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 120, 128, 144, 160, 192],
  "HBM bandwidth": [
    ...range(100, 1000, 100),
    ...range(1200, 5000, 200),
    ...range(5500, 10000, 500),
  ],
  "BF16 compute": [
    ...range(0, 100, 10),
    ...range(125, 1000, 25),
    ...range(1100, 5000, 100),
  ],
  "FP8 compute": [
    ...range(0, 100, 10),
    ...range(125, 1000, 25),
    ...range(1100, 5000, 100),
  ],
  Power: [...range(50, 500, 25), ...range(550, 1500, 50)],
  "GPUs / node": [1, 2, 4, 8, 16],
  Nodes: parallelSizes,
  Interconnect: [
    ...range(10, 100, 10),
    ...range(125, 500, 25),
    ...range(550, 2000, 50),
  ],
  Network: [
    ...range(10, 100, 10),
    ...range(125, 500, 25),
    ...range(550, 1600, 50),
  ],
  "GPU cost / hour": hourlyCosts,
  "Network / hour": hourlyCosts,
  "Storage / hour": hourlyCosts,
  "CPU / hour": hourlyCosts,
  "Data parallel": parallelSizes,
  "Tensor parallel": parallelSizes,
  "Pipeline parallel": parallelSizes,
  "Sequence parallel": parallelSizes,
  "Expert parallel": parallelSizes,
};

export function sliderStops(
  label: string,
  value: number,
  min?: number,
  max?: number,
  step = 1,
) {
  const configured = stopsByLabel[label];
  const upper = max ?? Math.max(value * 2, step * 10, 1);
  const base = configured ?? range(min ?? 0, upper, step);
  const bounded = base.filter(
    (stop) =>
      (min === undefined || stop >= min) && (max === undefined || stop <= max),
  );
  // Model presets can contain precise values between the usual steps.
  return [...new Set([...bounded, value])].sort((a, b) => a - b);
}

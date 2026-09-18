export type Precision = "FP32" | "FP16" | "BF16" | "FP8" | "Mixed";
export type Attention =
  | "Standard"
  | "FlashAttention"
  | "Memory-efficient"
  | "Grouped query"
  | "Multi-query"
  | "MLA"
  | "Sliding window"
  | "Sparse";
export type Optimizer = "SGD" | "Adam" | "AdamW" | "Adafactor" | "8-bit Adam";
export type Config = {
  model: {
    name: string;
    vocab: number;
    hidden: number;
    layers: number;
    heads: number;
    kvHeads: number;
    ffn: number;
    experts?: number;
    activeExperts?: number;
    sequence: number;
    attention: Attention;
    precision: Precision;
    activation: "SwiGLU" | "GELU" | "GeGLU" | "ReLU";
    norm: "RMSNorm" | "LayerNorm";
    position: "RoPE" | "Learned" | "Sinusoidal" | "ALiBi";
    tiedEmbeddings: boolean;
    dropout: number;
  };
  tokenizer: {
    type: string;
    specialTokens: number;
    charsPerToken: number;
    bytesPerToken: number;
  };
  dataset: {
    name: string;
    tokens: number;
    epochs: number;
    avgDocument: number;
  };
  training: {
    microBatch: number;
    globalBatch: number;
    accumulation: number;
    learningRate: number;
    optimizer: Optimizer;
    weightDecay: number;
    warmup: number;
    scheduler: string;
    gradClip: number;
    checkpointInterval: number;
  };
  hardware: {
    gpu: string;
    gpusPerNode: number;
    nodes: number;
    vram: number;
    bandwidth: number;
    compute: number;
    fp8Compute: number;
    interconnect: number;
    network: number;
    power: number;
    hourlyCost: number;
    networkCost: number;
    storageCost: number;
    cpuCost: number;
    link: string;
  };
  distributed: { dp: number; tp: number; pp: number; sp: number; ep: number };
  seed: number;
};
export type Metrics = ReturnType<typeof simulate>;
const gb = 1024 ** 3;
export const format = (n: number, d = 1) =>
  !Number.isFinite(n)
    ? "—"
    : Math.abs(n) >= 1e12
      ? (n / 1e12).toFixed(d) + "T"
      : Math.abs(n) >= 1e9
        ? (n / 1e9).toFixed(d) + "B"
        : Math.abs(n) >= 1e6
          ? (n / 1e6).toFixed(d) + "M"
          : Math.abs(n) >= 1e3
            ? (n / 1e3).toFixed(d) + "K"
            : n.toFixed(n < 10 ? d : 0);
export const money = (n: number) => "$" + Math.round(n).toLocaleString();
export const duration = (hours: number) =>
  hours < 1
    ? `${Math.round(hours * 60)} min`
    : hours < 48
      ? `${hours.toFixed(1)} hr`
      : `${(hours / 24).toFixed(1)} days`;
export type Architecture = {
  transformer: number;
  attention: number;
  ffn: number;
  embedding: number;
  head: number;
};
export type GpuDevice = Pick<
  Config["hardware"],
  | "gpu"
  | "vram"
  | "bandwidth"
  | "compute"
  | "fp8Compute"
  | "interconnect"
  | "power"
  | "hourlyCost"
> & { id: string };
export function simulate(
  c: Config,
  architecture: Architecture = {
    transformer: 1,
    attention: 0,
    ffn: 0,
    embedding: 1,
    head: 1,
  },
  devices: GpuDevice[] = [],
) {
  const m = c.model,
    t = c.training,
    h = c.hardware,
    d = c.distributed,
    gpus = devices.length;
  const configuredDevices = d.dp * d.tp * d.pp * d.ep;
  const parallel =
    configuredDevices === gpus
      ? d
      : { ...d, dp: Math.max(1, gpus), tp: 1, pp: 1, ep: 1 };
  const capacity = devices.reduce((sum, device) => sum + device.vram, 0);
  const bytes = m.precision === "FP32" ? 4 : m.precision === "FP8" ? 1 : 2;
  const layers = m.layers * architecture.transformer;
  const attentionLayers = layers + architecture.attention,
    ffnLayers = layers + architecture.ffn;
  const q = m.hidden * m.hidden,
    k = (m.hidden * m.hidden * m.kvHeads) / Math.max(1, m.heads),
    v = k,
    o = q;
  const attention = (q + k + v + o) * attentionLayers;
  const ffnPer =
    m.activation === "SwiGLU" || m.activation === "GeGLU"
      ? 3 * m.hidden * m.ffn
      : 2 * m.hidden * m.ffn;
  const expertCount = Math.max(1, Math.round(m.experts ?? 1));
  const activeExpertCount = Math.max(
    1,
    Math.min(expertCount, Math.round(m.activeExperts ?? 1)),
  );
  const ffn = ffnPer * ffnLayers * expertCount,
    norm = layers * m.hidden * 4;
  const embeddings =
    (m.vocab * m.hidden +
      (m.position === "Learned" ? m.sequence * m.hidden : 0)) *
    architecture.embedding;
  const lmHead =
    m.tiedEmbeddings && architecture.embedding > 0
      ? 0
      : m.vocab * m.hidden * architecture.head;
  const parameters = embeddings + attention + ffn + norm + lmHead;
  const localParams = parameters / Math.max(1, gpus);
  const paramGB = (localParams * bytes) / gb,
    gradientGB = (localParams * bytes) / gb;
  const optBytes =
    t.optimizer === "SGD"
      ? 4
      : t.optimizer === "Adafactor"
        ? 4
        : t.optimizer === "8-bit Adam"
          ? 2
          : 8;
  const optimizerGB = (localParams * optBytes) / gb;
  const attentionFactor =
    m.attention === "Standard"
      ? 1
      : m.attention === "FlashAttention"
        ? 0.19
        : m.attention === "Memory-efficient"
          ? 0.28
          : m.attention === "MLA"
            ? 0.2
            : m.attention === "Sliding window"
              ? 0.3
              : m.attention === "Sparse"
                ? 0.22
                : 0.65;
  const attentionGB =
    (t.microBatch *
      m.heads *
      m.sequence *
      m.sequence *
      bytes *
      attentionFactor) /
    gb /
    Math.max(1, gpus * d.sp);
  const activationGB =
    (t.microBatch *
      m.sequence *
      m.hidden *
      Math.max(1, layers + architecture.attention + architecture.ffn) *
      bytes *
      13) /
      gb /
      Math.max(1, gpus * d.sp) +
    attentionGB;
  const temporaryGB = Math.max(0.5, paramGB * 0.05 + activationGB * 0.08),
    runtimeGB = 1.2;
  const usedGB =
    paramGB + gradientGB + optimizerGB + activationGB + temporaryGB + runtimeGB;
  const gpuDevices = devices.map((device) => ({
    ...device,
    usedGB,
    freeGB: device.vram - usedGB,
    usage: device.vram > 0 ? (usedGB / device.vram) * 100 : 0,
    oom: usedGB > device.vram,
  }));
  const oom = gpuDevices.some((device) => device.oom);
  const tokensPerStep = t.globalBatch * m.sequence;
  const activeFfn = ffnPer * ffnLayers * activeExpertCount;
  const flopsPerToken =
    6 * (embeddings + attention + activeFfn + norm + lmHead) +
    12 * attentionLayers * m.sequence * m.hidden;
  const precisionCompute = devices.reduce(
    (sum, device) =>
      sum +
      (m.precision === "FP8" && device.fp8Compute > 0
        ? device.fp8Compute
        : m.precision === "FP32"
          ? device.compute * 0.25
          : m.precision === "Mixed"
            ? device.compute * 0.88
            : device.compute),
    0,
  );
  const peak = precisionCompute * 1e12;
  const parallelPenalty = Math.min(
    0.52,
    0.025 * Math.log2(gpus) +
      0.038 * (parallel.tp - 1) +
      0.025 * (parallel.pp - 1) +
      0.015 * (parallel.dp - 1),
  );
  const effectiveLink = Math.max(
    1,
    gpus > 1
      ? Math.min(...devices.map((device) => device.interconnect))
      : (devices[0]?.interconnect ?? 1),
  );
  const topologyPenalty =
    gpus > 1
      ? Math.min(0.3, (gpus - 1) * 0.008 * (400 / Math.max(10, h.network)))
      : 0;
  const linkPenalty = Math.min(
    0.18,
    (Math.max(0, 900 - effectiveLink) / 900) * 0.09,
  );
  const memoryPenalty = Math.max(
    0,
    (usedGB / Math.max(1, capacity / Math.max(1, gpus)) - 0.72) * 0.14,
  );
  const utilization =
    gpus && !oom
      ? Math.max(
          0.15,
          Math.min(
            0.72,
            0.52 +
              Math.log2(Math.max(1, t.microBatch)) * 0.045 -
              parallelPenalty * 0.3 -
              memoryPenalty,
          ),
        )
      : 0;
  const efficiency = gpus
    ? Math.max(0.18, 1 - parallelPenalty - topologyPenalty - linkPenalty)
    : 0;
  const computeTokens = (peak * utilization * efficiency) / flopsPerToken;
  const bandwidthTokens =
    (devices.reduce((sum, device) => sum + device.bandwidth, 0) *
      1e9 *
      Math.max(0.25, efficiency)) /
    (Math.max(1, (parameters * bytes) / tokensPerStep) +
      m.hidden * m.layers * bytes * 2);
  const tokensPerSecond =
    gpus && !oom ? Math.max(0, Math.min(computeTokens, bandwidthTokens)) : 0;
  const stepsPerSecond = tokensPerSecond / tokensPerStep;
  const stepSeconds = stepsPerSecond ? 1 / stepsPerSecond : 0;
  const commShare = Math.min(0.38, (1 - efficiency) * 0.62);
  const activeShare = 1 - commShare;
  const forwardSeconds = stepSeconds * 0.3 * activeShare,
    backwardSeconds = stepSeconds * 0.55 * activeShare,
    optimizerSeconds = stepSeconds * 0.15 * activeShare,
    communicationSeconds = stepSeconds * commShare;
  const totalTokens = c.dataset.tokens * c.dataset.epochs;
  const steps = Math.ceil(totalTokens / tokensPerStep);
  const rawHours = tokensPerSecond ? totalTokens / tokensPerSecond / 3600 : 0;
  const overhead =
    1.12 +
    Math.min(0.13, (1000 / Math.max(1000, t.checkpointInterval)) * 0.04) +
    topologyPenalty * 0.4;
  const hours = rawHours * overhead;
  const hourlyCost = gpus
    ? devices.reduce((sum, device) => sum + device.hourlyCost, 0) +
      h.networkCost +
      h.storageCost +
      h.cpuCost
    : 0;
  const cost = hours * hourlyCost;
  const energyKWh =
    (hours * devices.reduce((sum, device) => sum + device.power, 0)) / 1000;
  const mfu = gpus
    ? Math.min(
        0.75,
        utilization * efficiency * ((6 * parameters) / flopsPerToken),
      )
    : 0;
  const bandwidthUtil = gpus
    ? Math.min(
        0.98,
        Math.max(
          0.22,
          bandwidthTokens < computeTokens
            ? 0.81
            : 0.35 +
                (usedGB / Math.max(1, capacity / Math.max(1, gpus))) * 0.35,
        ),
      )
    : 0;
  const bottleneck = oom
    ? "GPU memory"
    : !gpus
      ? "No GPU connected"
      : bandwidthTokens < computeTokens
        ? "Memory bandwidth"
        : efficiency < 0.7
          ? "Interconnect"
          : "Compute";
  const attentionMatrixGB =
    (t.microBatch * m.heads * m.sequence * m.sequence * bytes) / gb;
  const checkpointGB = (parameters * bytes) / gb + (parameters * optBytes) / gb;
  return {
    parameters,
    breakdown: {
      embeddings,
      attention,
      ffn,
      norm,
      lmHead,
      q: q * attentionLayers,
      k: k * attentionLayers,
      v: v * attentionLayers,
      o: o * attentionLayers,
    },
    memory: {
      parameters: gpus ? paramGB : 0,
      gradients: gpus ? gradientGB : 0,
      optimizer: gpus ? optimizerGB : 0,
      activations: gpus ? activationGB : 0,
      temporary: gpus ? temporaryGB : 0,
      runtime: gpus ? runtimeGB : 0,
      total: gpus ? usedGB : 0,
      free: gpus ? Math.min(...gpuDevices.map((device) => device.freeGB)) : 0,
    },
    gpuDevices,
    capacity,
    oom,
    gpus,
    efficiency,
    utilization,
    mfu,
    bandwidthUtil,
    tokensPerSecond,
    tokensPerGpu: tokensPerSecond / gpus,
    samplesPerSecond: tokensPerSecond / m.sequence,
    stepsPerSecond,
    stepSeconds,
    forwardSeconds,
    backwardSeconds,
    optimizerSeconds,
    communicationSeconds,
    totalTokens,
    steps,
    hours,
    hourlyCost,
    cost,
    costPerB: cost / (totalTokens / 1e9),
    costPerStep: cost / steps,
    energyKWh,
    bottleneck,
    attentionMatrixGB,
    checkpointGB,
    flopsPerToken,
    peakTFlops: precisionCompute,
    overhead,
  };
}
export function trainingAtProgress(
  c: Config,
  metrics: Metrics,
  fraction: number,
) {
  const p = Math.max(0, Math.min(1, fraction));
  const startLoss = Math.log(Math.max(2, c.model.vocab));
  const floor = 1.7 + Math.log10(Math.max(1, metrics.parameters)) / 10;
  const variation = Math.sin((p * 997 + c.seed) * 0.73) * 0.025 * (1 - p);
  const loss = floor + (startLoss - floor) * Math.exp(-4.6 * p) + variation;
  const completedSteps = Math.floor(metrics.steps * p);
  const processedTokens = Math.min(
    metrics.totalTokens,
    completedSteps * c.training.globalBatch * c.model.sequence,
  );
  const elapsedHours = metrics.hours * p;
  return {
    loss,
    completedSteps,
    processedTokens,
    elapsedHours,
    spent: metrics.hourlyCost * elapsedHours,
    energyKWh: metrics.energyKWh * p,
    checkpoints: Math.floor(
      completedSteps / Math.max(1, c.training.checkpointInterval),
    ),
  };
}

import type { Config } from '@/sim/engine';
export const gpuPresets: Record<string, Pick<Config['hardware'], 'vram' | 'bandwidth' | 'compute' | 'fp8Compute' | 'interconnect' | 'power' | 'hourlyCost'>> = {
  'NVIDIA T4': { vram: 16, bandwidth: 320, compute: 65, fp8Compute: 0, interconnect: 32, power: 70, hourlyCost: .45 },
  'NVIDIA A10': { vram: 24, bandwidth: 600, compute: 125, fp8Compute: 0, interconnect: 64, power: 150, hourlyCost: 1.1 },
  'NVIDIA L4': { vram: 24, bandwidth: 300, compute: 121, fp8Compute: 242, interconnect: 64, power: 72, hourlyCost: .9 },
  'A100 40GB': { vram: 40, bandwidth: 1555, compute: 312, fp8Compute: 0, interconnect: 600, power: 400, hourlyCost: 2.2 },
  'A100 80GB': { vram: 80, bandwidth: 2039, compute: 312, fp8Compute: 0, interconnect: 600, power: 400, hourlyCost: 3.1 },
  'H100 80GB': { vram: 80, bandwidth: 3350, compute: 989, fp8Compute: 1979, interconnect: 900, power: 700, hourlyCost: 4.25 },
  'H200': { vram: 141, bandwidth: 4800, compute: 989, fp8Compute: 1979, interconnect: 900, power: 700, hourlyCost: 5.2 },
  'B200 class': { vram: 180, bandwidth: 8000, compute: 2250, fp8Compute: 4500, interconnect: 1800, power: 1000, hourlyCost: 8.5 },
  'AMD MI250': { vram: 128, bandwidth: 3200, compute: 383, fp8Compute: 0, interconnect: 400, power: 560, hourlyCost: 2.8 },
  'AMD MI300X': { vram: 192, bandwidth: 5300, compute: 1307, fp8Compute: 0, interconnect: 896, power: 750, hourlyCost: 4.8 },
  'Custom Accelerator': { vram: 80, bandwidth: 2000, compute: 500, fp8Compute: 0, interconnect: 600, power: 500, hourlyCost: 3 }
};
export const modelPresets: Record<string, Partial<Config['model']>> = {
  'Tiny Transformer': { hidden: 256, layers: 6, heads: 8, kvHeads: 8, ffn: 1024, vocab: 32000 },
  'GPT-2 style': { hidden: 768, layers: 12, heads: 12, kvHeads: 12, ffn: 3072, vocab: 50257, activation: 'GELU' },
  '125M': { hidden: 768, layers: 12, heads: 12, kvHeads: 12, ffn: 3072, vocab: 32000 },
  '350M': { hidden: 1024, layers: 24, heads: 16, kvHeads: 16, ffn: 4096, vocab: 32000 },
  '1B': { hidden: 2048, layers: 24, heads: 16, kvHeads: 8, ffn: 5632, vocab: 32000 },
  '3B': { hidden: 3072, layers: 28, heads: 24, kvHeads: 8, ffn: 8192, vocab: 32000 },
  '7B': { hidden: 4096, layers: 32, heads: 32, kvHeads: 8, ffn: 14336, vocab: 32000 },
  '13B': { hidden: 5120, layers: 40, heads: 40, kvHeads: 8, ffn: 13824, vocab: 32000 },
  '30B': { hidden: 6656, layers: 60, heads: 52, kvHeads: 8, ffn: 17920, vocab: 32000 },
  '70B': { hidden: 8192, layers: 80, heads: 64, kvHeads: 8, ffn: 28672, vocab: 32000 },
  'LLaMA style': { hidden: 4096, layers: 32, heads: 32, kvHeads: 32, ffn: 11008, activation: 'SwiGLU', norm: 'RMSNorm', position: 'RoPE' },
  'Mistral style': { hidden: 4096, layers: 32, heads: 32, kvHeads: 8, ffn: 14336, attention: 'Sliding window' },
  'MoE approximation': { hidden: 4096, layers: 32, heads: 32, kvHeads: 8, ffn: 14336 },
  'Encoder only': { hidden: 768, layers: 12, heads: 12, kvHeads: 12, ffn: 3072, attention: 'Standard' },
  'Encoder decoder': { hidden: 1024, layers: 24, heads: 16, kvHeads: 16, ffn: 4096 }
};
export const initialConfig: Config = { model: { name: '7B decoder', vocab: 32000, hidden: 4096, layers: 32, heads: 32, kvHeads: 8, ffn: 14336, sequence: 4096, attention: 'FlashAttention', precision: 'BF16', activation: 'SwiGLU', norm: 'RMSNorm', position: 'RoPE', tiedEmbeddings: true, dropout: 0 }, tokenizer: { type: 'BPE', specialTokens: 4, charsPerToken: 4, bytesPerToken: 4 }, dataset: { name: 'Web + code corpus', tokens: 30e9, epochs: 1, avgDocument: 1800 }, training: { microBatch: 1, globalBatch: 256, accumulation: 4, learningRate: .0003, optimizer: 'AdamW', weightDecay: .1, warmup: 2000, scheduler: 'Cosine', gradClip: 1, checkpointInterval: 1000 }, hardware: { gpu: 'H100 80GB', gpusPerNode: 8, nodes: 1, network: 400, link: 'NVLink + InfiniBand', networkCost: 2, storageCost: 1.5, cpuCost: 1, vram: 80, bandwidth: 3350, compute: 989, fp8Compute: 1979, interconnect: 900, power: 700, hourlyCost: 4.25 }, distributed: { dp: 4, tp: 2, pp: 1, sp: 1, ep: 1 }, seed: 42 };

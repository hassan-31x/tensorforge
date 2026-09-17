# Tensorforge

An interactive Next.js studio for sketching LLM training architecture and estimating its hardware footprint, throughput, duration, and cost.

## Run

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## Workflow

Drag model and hardware blocks from the left palette onto the canvas. Connect their handles to document tensor and hardware flow. Select a block to inspect it, or use the right panel to configure model, tokenizer, dataset, training, cluster, and parallelism. The top bar updates as configuration and model blocks change. Run the animation and open the profiler for an iteration timeline, memory and parameter breakdown, and logs.

Use the bottom bar to save to local storage, import or export a project JSON file, export a PNG or report JSON, and copy a shareable URL.

## Estimation model

The deterministic engine in `src/sim/engine.ts` estimates dense transformer parameters, per-GPU parameter and optimizer memory, attention activations, training FLOPs, throughput, topology penalties, duration, energy, and cost. GPU specifications and prices in `src/data/presets.ts` are editable educational presets, not live cloud quotes. The profiler and training animation visualize these estimates; they do not execute training kernels or predict actual model quality.

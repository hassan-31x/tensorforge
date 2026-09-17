# Tensorforge

Tensorforge is an interactive LLM training design studio. It lets you sketch a decoder-style language model and its GPU topology, then explore how architecture, precision, batching, parallelism, and hardware change memory, throughput, training time, energy, and cost.

The interface is a compact canvas for technical exploration rather than a static neural-network diagram. Components can be added, moved, connected, inspected, and reconfigured while the simulation updates in place.

## What it includes

- A drag-and-drop model graph for raw text, tokenizers, embeddings, positional encoding, transformer blocks, attention, feed-forward layers, normalization, LM heads, loss, datasets, optimizers, GPU nodes, and clusters.
- Connectable node handles for documenting tensor flow and hardware flow.
- Tensor shape mode with representative shapes such as `[B, T]`, `[B, T, D]`, and `[B, H, T, T]`.
- Configuration panels for model dimensions, tokenizer settings, dataset size, training schedule, optimizer, precision, GPU presets, cluster pricing, and distributed parallelism.
- Model presets from tiny transformers through approximate 70B architectures, plus GPT-style, LLaMA-style, Mistral-style, MoE, encoder-only, and encoder-decoder starting points.
- Hardware presets for T4, A10, L4, A100, H100, H200, B200-class, and AMD accelerators, with editable specifications and prices.
- Live estimates for parameter count, parameter groups, per-GPU memory, attention activation memory, GPU utilization, MFU, bandwidth utilization, tokens per second, samples per second, step time, total training time, energy, cost, checkpoint size, and cost per billion tokens.
- Explicit warnings for VRAM overflow and invalid DP × TP × PP × EP device mappings.
- A profiler drawer with forward, backward, all-reduce, and optimizer segments, plus memory, parameter, comparison, and log views.
- A training animation with 1×, 10×, 100×, 1000×, and instant speed controls. The animation exposes deterministic progress, loss, processed tokens, spend, and active graph phases.
- Local save/load, JSON import/export, report export, PNG architecture export, and shareable URL state.

## Requirements

- Node.js 20.9 or newer
- npm

Tensorforge is a client-side Next.js app. It does not require a database, API key, GPU, or hosted backend to run locally.

## Run locally

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

To create and run a production build:

```bash
npm run build
npm start
```

## Using the studio

1. Start with the model graph in the center. The default project is an approximately 7B decoder setup using BF16 and an 8-GPU H100 node.
2. Drag a component from the left palette onto the canvas, or click a palette item to add it.
3. Connect a node's right handle to another node's left handle. Select a node or edge to inspect it.
4. Use the right-side tabs to change model, dataset, training, hardware, or parallelism settings. Numeric changes recalculate the simulation immediately.
5. Press **Simulate** to animate execution through the graph. The forward pass, backward pass, all-reduce, and optimizer phases use different colors.
6. Open **Show profiler** at the bottom for iteration timing, memory and parameter breakdowns, comparisons, and logs.
7. Use **Compare** in the bottom utility bar to capture a baseline, then change settings to see the current setup against it.

### Useful experiments

- Double sequence length and watch the attention score matrix grow quadratically.
- Switch from standard attention to FlashAttention to compare activation memory and throughput.
- Change BF16 to FP32 or FP8 and observe the memory and compute consequences.
- Increase micro-batch size until the GPU memory panel reports an OOM and explains the overage.
- Change node count, network bandwidth, or parallelism to see topology and communication penalties reduce scaling efficiency.
- Edit GPU and cloud prices to model a local cluster or a different provider rather than relying on a fixed quote.

## Save, import, and share

- **Save** stores the current configuration, nodes, and edges in browser local storage under `tensorforge-project`.
- **Import** accepts a project JSON file exported by Tensorforge.
- **JSON** exports the complete project configuration and graph.
- **Report** exports the current configuration, calculated metrics, and timestamp as JSON.
- **PNG** exports the visible architecture canvas as a PNG image.
- **Share** encodes the project state into the page URL hash and copies the link to the clipboard.

A project file contains the model, tokenizer, dataset, training, hardware, distributed-training, and graph state needed to reproduce the same estimate with the same seed.

## Simulation model

The simulator is deterministic for a given configuration and seed. It is separated from the React UI in [`src/sim/engine.ts`](src/sim/engine.ts) and currently estimates:

- Dense transformer parameter groups, including embeddings, attention projections, feed-forward layers, normalization, and the LM head.
- Precision-dependent parameter, gradient, optimizer, activation, temporary, and runtime memory.
- Attention activation memory, including the quadratic `T × T` score matrix and memory-saving attention modes.
- FLOPs per token, compute and bandwidth limits, GPU utilization, MFU, and tokens per second.
- Data, tensor, pipeline, sequence, and expert parallelism penalties, including multi-node network overhead.
- Step count, execution time, checkpoint overhead, energy, hourly spend, complete training cost, and cost per billion tokens.

GPU properties and prices live in [`src/data/presets.ts`](src/data/presets.ts). They are editable educational presets, not live cloud-provider pricing or hardware guarantees.

The training animation visualizes these estimates. It does not execute CUDA kernels, train a model, benchmark a real cluster, or predict convergence quality. The displayed loss curve is a deterministic teaching signal intended to make progress and cost easier to understand.

## Project structure

```text
src/
  app/
    page.tsx              # App entry route
    layout.tsx            # Metadata and root layout
    globals.css           # Canvas, panels, controls, and responsive styling
  components/
    Studio.tsx            # Canvas, inspector, profiler, and interactions
  data/
    presets.ts            # Model, GPU, and default configuration presets
  sim/
    engine.ts             # Deterministic training and cost calculations
```

The UI uses [React Flow](https://reactflow.dev/) for the graph canvas and `lucide-react` for interface icons. PNG export uses `html-to-image`.

## Extending Tensorforge

To add a model or accelerator preset, update the corresponding map in `src/data/presets.ts`. To add a new metric or change an estimate, update `simulate` in `src/sim/engine.ts` and expose the result in the top bar, inspector, profiler, or report export. Keep the engine free of React state so it remains deterministic and testable.

Before opening a pull request, run:

```bash
npm run build
```

## License

No license has been selected for this project yet.

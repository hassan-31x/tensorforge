import test from "node:test";
import assert from "node:assert/strict";
import { resolveNetwork } from "../src/sim/topology.ts";
import { simulate } from "../src/sim/engine.ts";
import { initialConfig, gpuPresets } from "../src/data/presets.ts";

const architecture = {
  transformer: 1,
  attention: 0,
  ffn: 0,
  embedding: 1,
  head: 1,
};
const hardware = (gpu) => ({
  ...initialConfig.hardware,
  gpu,
  ...gpuPresets[gpu],
});
const model = [
  {
    id: "transformer",
    position: { x: 100, y: 100 },
    width: 218,
    height: 150,
    data: { kind: "transformer" },
  },
  {
    id: "head",
    position: { x: 400, y: 100 },
    width: 218,
    height: 150,
    data: { kind: "head" },
  },
];
const group = {
  id: "group",
  position: { x: 50, y: 50 },
  width: 650,
  height: 300,
  data: { kind: "group" },
};
const h100 = {
  id: "h100",
  position: { x: 100, y: 450 },
  data: { kind: "gpu", hardware: hardware("H100 80GB") },
};
const t4 = {
  id: "t4",
  position: { x: 400, y: 450 },
  data: { kind: "gpu", hardware: hardware("NVIDIA T4") },
};
const modelEdge = { source: "transformer", target: "head" };

test("hardware estimates require a complete framed model and a GPU connection", () => {
  const withoutGroup = resolveNetwork(
    [...model, h100],
    [modelEdge],
    initialConfig.hardware,
  );
  assert.equal(
    simulate(initialConfig, architecture, withoutGroup.devices).cost,
    0,
  );

  const disconnected = resolveNetwork(
    [...model, group, h100],
    [modelEdge],
    initialConfig.hardware,
  );
  assert.equal(disconnected.devices.length, 0);
  assert.equal(
    simulate(initialConfig, architecture, disconnected.devices).hours,
    0,
  );

  const partialGroup = { ...group, width: 300 };
  const partial = resolveNetwork(
    [...model, partialGroup, h100],
    [modelEdge, { source: "h100", target: "group" }],
    initialConfig.hardware,
  );
  assert.equal(partial.devices.length, 0);

  const connected = resolveNetwork(
    [...model, group, h100],
    [modelEdge, { source: "h100", target: "group" }],
    initialConfig.hardware,
  );
  assert.equal(connected.devices.length, 1);
  assert.ok(
    simulate(initialConfig, architecture, connected.devices).hourlyCost > 0,
  );
});

test("only connected GPUs are billed and mixed VRAM changes each device's usage", () => {
  const one = resolveNetwork(
    [...model, group, h100, t4],
    [modelEdge, { source: "h100", target: "group" }],
    initialConfig.hardware,
  );
  const two = resolveNetwork(
    [...model, group, h100, t4],
    [
      modelEdge,
      { source: "h100", target: "group" },
      { source: "t4", target: "group" },
    ],
    initialConfig.hardware,
  );
  const oneMetrics = simulate(initialConfig, architecture, one.devices);
  const twoMetrics = simulate(initialConfig, architecture, two.devices);

  assert.equal(oneMetrics.gpus, 1);
  assert.equal(twoMetrics.gpus, 2);
  assert.ok(twoMetrics.hourlyCost > oneMetrics.hourlyCost);
  assert.ok(twoMetrics.memory.total < oneMetrics.memory.total);
  assert.ok(twoMetrics.gpuDevices[0].usage < oneMetrics.gpuDevices[0].usage);
  assert.ok(twoMetrics.gpuDevices[1].usage > twoMetrics.gpuDevices[0].usage);
  assert.ok(
    (twoMetrics.memory.total * twoMetrics.gpus) / twoMetrics.capacity <
      (oneMetrics.memory.total * oneMetrics.gpus) / oneMetrics.capacity,
  );
  assert.equal(twoMetrics.gpuDevices[1].oom, true);
  assert.equal(twoMetrics.tokensPerSecond, 0);
  assert.equal(twoMetrics.hours, 0);
  assert.equal(twoMetrics.cost, 0);
});

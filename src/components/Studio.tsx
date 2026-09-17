"use client";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Handle,
  Position,
  addEdge,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type Connection,
  type NodeProps,
  ReactFlowProvider,
  useReactFlow,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  ArrowDownToLine,
  Braces,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Download,
  FileJson,
  Maximize2,
  MemoryStick,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Save,
  Search,
  Settings2,
  Share2,
  ShieldAlert,
  Sparkles,
  Upload,
  Workflow,
} from "lucide-react";
import {
  SiDeepseek,
  SiGoogle,
  SiHuggingface,
  SiMeta,
  SiMistralai,
  SiNvidia,
  SiPytorch,
  SiQwen,
} from "react-icons/si";
import type { IconType } from "react-icons";
import {
  TbAdjustmentsHorizontal,
  TbArrowsShuffle,
  TbBinaryTree,
  TbBrain,
  TbBraces,
  TbChartDots3,
  TbCircleDot,
  TbCube,
  TbMathFunction,
  TbSparkles,
  TbTextRecognition,
} from "react-icons/tb";
import {
  initialConfig,
  modelLibrary,
  modelPresets,
  gpuPresets,
} from "@/data/presets";
import {
  simulate,
  trainingAtProgress,
  format,
  money,
  duration,
  type Config,
} from "@/sim/engine";
import Image from "next/image";

type Kind =
  | "input"
  | "tokenizer"
  | "embedding"
  | "position"
  | "transformer"
  | "attention"
  | "ffn"
  | "norm"
  | "head"
  | "loss"
  | "dataset"
  | "gpu"
  | "cluster"
  | "optimizer"
  | "custom";
type StudioNode = Node<
  {
    kind: Kind;
    title: string;
    subtitle?: string;
    metric?: string;
    detail?: string;
    warning?: boolean;
    selected?: boolean;
    tensor?: string;
    executing?: boolean;
  },
  "studio"
>;
const catalog: {
  group: string;
  items: { kind: Kind; title: string; desc: string }[];
}[] = [
  {
    group: "INPUT & DATA",
    items: [
      { kind: "input", title: "Raw text", desc: "Prompt stream" },
      { kind: "tokenizer", title: "Tokenizer", desc: "Text to tokens" },
      { kind: "dataset", title: "Dataset", desc: "Training corpus" },
    ],
  },
  {
    group: "MODEL",
    items: [
      { kind: "embedding", title: "Token embedding", desc: "Lookup table" },
      { kind: "position", title: "Position encoding", desc: "RoPE or learned" },
      {
        kind: "transformer",
        title: "Transformer block",
        desc: "Repeatable layers",
      },
      { kind: "attention", title: "Attention", desc: "Q · K · V" },
      { kind: "ffn", title: "Feed forward", desc: "MLP / SwiGLU" },
      { kind: "norm", title: "Normalization", desc: "RMS / LayerNorm" },
      { kind: "head", title: "LM head", desc: "Vocabulary logits" },
      { kind: "loss", title: "Loss", desc: "Cross entropy" },
    ],
  },
  {
    group: "TRAINING & COMPUTE",
    items: [
      { kind: "gpu", title: "GPU node", desc: "Accelerator memory" },
      { kind: "cluster", title: "GPU cluster", desc: "Distributed system" },
      { kind: "optimizer", title: "Optimizer", desc: "Parameter updates" },
    ],
  },
];
const kindIcons: Record<Kind, IconType> = {
  input: TbTextRecognition,
  tokenizer: TbBraces,
  embedding: TbCube,
  position: TbAdjustmentsHorizontal,
  transformer: TbBinaryTree,
  attention: TbBrain,
  ffn: TbArrowsShuffle,
  norm: TbAdjustmentsHorizontal,
  head: TbMathFunction,
  loss: TbChartDots3,
  dataset: SiHuggingface,
  gpu: SiNvidia,
  cluster: TbBinaryTree,
  optimizer: SiPytorch,
  custom: TbSparkles,
};
const providerIcons: Record<string, IconType> = {
  OpenAI: TbBrain,
  Meta: SiMeta,
  "Mistral AI": SiMistralai,
  Google: SiGoogle,
  Qwen: SiQwen,
  DeepSeek: SiDeepseek,
};
const KindIcon = ({ kind, size = 17 }: { kind: Kind; size?: number }) => {
  const Icon = kindIcons[kind] || TbCircleDot;
  return <Icon size={size} strokeWidth={1.9} aria-hidden="true" />;
};
const initialNodes: StudioNode[] = [
  {
    id: "input",
    type: "studio",
    position: { x: 90, y: 120 },
    data: { kind: "input", title: "Raw text", subtitle: "Corpus input" },
  },
  {
    id: "tokenizer",
    type: "studio",
    position: { x: 335, y: 120 },
    data: {
      kind: "tokenizer",
      title: "Tokenizer",
      subtitle: "BPE · 32K vocabulary",
    },
  },
  {
    id: "embedding",
    type: "studio",
    position: { x: 580, y: 120 },
    data: {
      kind: "embedding",
      title: "Token embedding",
      subtitle: "[B, T] → [B, T, D]",
    },
  },
  {
    id: "transformer",
    type: "studio",
    position: { x: 830, y: 105 },
    data: {
      kind: "transformer",
      title: "Transformer stack",
      subtitle: "32 layers · hidden 4096",
    },
  },
  {
    id: "head",
    type: "studio",
    position: { x: 1110, y: 120 },
    data: { kind: "head", title: "LM head", subtitle: "Vocabulary projection" },
  },
  {
    id: "loss",
    type: "studio",
    position: { x: 1355, y: 120 },
    data: { kind: "loss", title: "Loss", subtitle: "Cross entropy" },
  },
  {
    id: "dataset",
    type: "studio",
    position: { x: 90, y: 385 },
    data: {
      kind: "dataset",
      title: "Training dataset",
      subtitle: "30B tokens · 1 epoch",
    },
  },
  {
    id: "gpu",
    type: "studio",
    position: { x: 580, y: 370 },
    data: { kind: "gpu", title: "H100 GPU node", subtitle: "8 GPUs · NVLink" },
  },
  {
    id: "optimizer",
    type: "studio",
    position: { x: 1110, y: 385 },
    data: {
      kind: "optimizer",
      title: "AdamW optimizer",
      subtitle: "Gradient update",
    },
  },
];
const edge = (
  source: string,
  target: string,
  id = source + "-" + target,
): Edge => ({
  id,
  source,
  target,
  type: "smoothstep",
  animated: true,
  markerEnd: { type: MarkerType.ArrowClosed, color: "#8ddac3" },
  style: { stroke: "#8ddac3", strokeWidth: 2 },
});
const initialEdges = [
  edge("input", "tokenizer"),
  edge("tokenizer", "embedding"),
  edge("embedding", "transformer"),
  edge("transformer", "head"),
  edge("head", "loss"),
  edge("dataset", "input"),
  edge("gpu", "transformer"),
  edge("loss", "optimizer"),
];
const IconButton = ({
  children,
  title,
  onClick,
  className = "",
}: {
  children: React.ReactNode;
  title: string;
  onClick?: () => void;
  className?: string;
}) => (
  <button
    className={"icon-btn " + className}
    title={title}
    aria-label={title}
    onClick={onClick}
  >
    {children}
  </button>
);
function NodeCard({ data, selected }: NodeProps<StudioNode>) {
  const k = data.kind;
  return (
    <div
      className={`node-card ${k} ${selected ? "is-selected" : ""} ${data.warning ? "warn" : ""} ${data.executing ? "executing" : ""}`}
    >
      <Handle type="target" position={Position.Left} className="node-handle" />
      <div className="node-top">
        <span className="node-glyph">
          <KindIcon kind={k} size={16} />
        </span>
        <span className="node-title">{data.title}</span>
        <span className="node-menu">•••</span>
      </div>
      <div className="node-sub">{data.subtitle}</div>
      {k === "transformer" ? (
        <div className="neuron-art">
          <i />
          <i />
          <i />
          <i />
          <i />
          <span />
          <i />
          <i />
          <i />
          <i />
          <i />
          <span />
          <i />
          <i />
          <i />
          <i />
          <i />
        </div>
      ) : null}
      {k === "gpu" || k === "cluster" ? (
        <>
          <div className="gpu-chips">
            {Array.from(
              {
                length: Math.min(
                  8,
                  Math.max(1, parseInt(data.subtitle || "1") || 1),
                ),
              },
              (_, i) => (
                <span key={i} title={`GPU ${i}`}>
                  <SiNvidia size={10} aria-hidden="true" />
                </span>
              ),
            )}
          </div>
          <div className="node-meter">
            <span
              style={{
                width: `${Math.min(100, Math.max(0, Number(data.metric?.replace("%", "")) || 0))}%`,
              }}
            />
          </div>
          <div className="node-metric">
            <strong>{data.metric}</strong>
            <span>{data.detail || "VRAM used"}</span>
          </div>
        </>
      ) : (
        <div className="node-footer">
          <span>{data.metric || "Ready"}</span>
          <span>{data.detail || "→"}</span>
        </div>
      )}
      <Handle type="source" position={Position.Right} className="node-handle" />
    </div>
  );
}
const nodeTypes = { studio: NodeCard };

const sliderBounds: Record<string, { min: number; max: number; step: number }> =
  {
    "Training tokens": { min: 1e9, max: 20e12, step: 1e9 },
    "Available tokens": { min: 1e9, max: 20e12, step: 1e9 },
    "Vocabulary size": { min: 8000, max: 300000, step: 1000 },
    Vocabulary: { min: 8000, max: 300000, step: 1000 },
    "Hidden size": { min: 128, max: 16384, step: 128 },
    Layers: { min: 1, max: 128, step: 1 },
    "Attention heads": { min: 1, max: 128, step: 1 },
    "KV heads": { min: 1, max: 128, step: 1 },
    "FFN size": { min: 256, max: 65536, step: 256 },
    Experts: { min: 1, max: 512, step: 1 },
    "Active experts": { min: 1, max: 64, step: 1 },
    "Sequence length": { min: 128, max: 131072, step: 128 },
    "Special tokens": { min: 0, max: 512, step: 1 },
    "Characters / token": { min: 1, max: 8, step: 0.1 },
    "Bytes / token": { min: 1, max: 8, step: 0.1 },
    "Average document": { min: 128, max: 4096, step: 64 },
    Epochs: { min: 1, max: 10, step: 1 },
    "Micro batch / GPU": { min: 1, max: 64, step: 1 },
    "Global batch": { min: 1, max: 8192, step: 1 },
    Accumulation: { min: 1, max: 64, step: 1 },
    "Learning rate": { min: 0.00001, max: 0.003, step: 0.00001 },
    "Warmup steps": { min: 0, max: 20000, step: 100 },
    "Weight decay": { min: 0, max: 0.5, step: 0.01 },
    "Gradient clip": { min: 0, max: 10, step: 0.1 },
    "Checkpoint every": { min: 100, max: 10000, step: 100 },
    VRAM: { min: 4, max: 192, step: 4 },
    "HBM bandwidth": { min: 100, max: 10000, step: 50 },
    "BF16 compute": { min: 0, max: 5000, step: 10 },
    "FP8 compute": { min: 0, max: 5000, step: 10 },
    Power: { min: 50, max: 1500, step: 10 },
    "GPUs / node": { min: 1, max: 16, step: 1 },
    Nodes: { min: 1, max: 128, step: 1 },
    Interconnect: { min: 10, max: 2000, step: 10 },
    Network: { min: 10, max: 1600, step: 10 },
    "GPU cost / hour": { min: 0, max: 20, step: 0.05 },
    "Network / hour": { min: 0, max: 20, step: 0.05 },
    "Storage / hour": { min: 0, max: 20, step: 0.05 },
    "CPU / hour": { min: 0, max: 20, step: 0.05 },
    "Data parallel": { min: 1, max: 128, step: 1 },
    "Tensor parallel": { min: 1, max: 128, step: 1 },
    "Pipeline parallel": { min: 1, max: 128, step: 1 },
    "Sequence parallel": { min: 1, max: 128, step: 1 },
    "Expert parallel": { min: 1, max: 128, step: 1 },
  };
function Field({
  label,
  value,
  onChange,
  unit,
  min,
  max,
  step = 1,
  hint,
}: {
  label: string;
  value: number | string;
  onChange: (v: any) => void;
  unit?: string;
  min?: number;
  max?: number;
  step?: number;
  hint?: string;
}) {
  const numeric = typeof value === "number";
  const preset = sliderBounds[label];
  const sliderStep = preset?.step ?? step;
  const sliderMin = min ?? preset?.min ?? 0;
  const sliderMax =
    max ?? preset?.max ?? Math.max(Number(value) * 2, sliderStep * 10, 1);
  return (
    <label className="field">
      <span className="field-label">
        {label}
        {hint && <small title={hint}>ⓘ</small>}
        <output className="field-value">
          {numeric
            ? Number(value).toLocaleString(undefined, {
                maximumFractionDigits: sliderStep < 1 ? 5 : 0,
              })
            : value}
          {unit ? ` ${unit}` : ""}
        </output>
      </span>
      {numeric ? (
        <input
          className="range-input"
          type="range"
          value={value}
          min={sliderMin}
          max={sliderMax}
          step={sliderStep}
          aria-label={label}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      ) : (
        <div className="input-wrap">
          <input
            type="text"
            value={value}
            aria-label={label}
            onChange={(e) => onChange(e.target.value)}
          />
        </div>
      )}
    </label>
  );
}
function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <div className="select-wrap">
        <select value={value} onChange={(e) => onChange(e.target.value)}>
          {options.map((x) => (
            <option key={x}>{x}</option>
          ))}
        </select>
        <ChevronDown size={14} />
      </div>
    </label>
  );
}
function Section({
  title,
  children,
  defaultOpen = true,
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="inspector-section">
      <button className="section-heading" onClick={() => setOpen(!open)}>
        <span>{title}</span>
        {open ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
      </button>
      {open && <div className="section-body">{children}</div>}
    </section>
  );
}
function Bar({
  label,
  value,
  total,
  color,
}: {
  label: string;
  value: number;
  total: number;
  color: string;
}) {
  return (
    <div className="bar-row">
      <span>{label}</span>
      <div className="bar-track">
        <span
          style={{
            width: `${Math.max(0, Math.min(100, (value / Math.max(1, total)) * 100))}%`,
            background: color,
          }}
        />
      </div>
      <b>{value.toFixed(1)} GB</b>
    </div>
  );
}
function StudioInner() {
  const [config, setConfig] = useState<Config>(initialConfig);
  const [nodes, setNodes, onNodesChange] =
    useNodesState<StudioNode>(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selected, setSelected] = useState<string | null>("transformer");
  const [activePanel, setActivePanel] = useState<
    "inspect" | "model" | "dataset" | "training" | "hardware" | "distributed"
  >("inspect");
  const [search, setSearch] = useState("");
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState("10×");
  const [progress, setProgress] = useState(0);
  const [liveTick, setLiveTick] = useState(0);
  const [bottomOpen, setBottomOpen] = useState(false);
  const [bottomTab, setBottomTab] = useState<
    "timeline" | "memory" | "parameters" | "compare" | "logs"
  >("timeline");
  const [tensorMode, setTensorMode] = useState(false);
  const [showPresets, setShowPresets] = useState(false);
  const [notice, setNotice] = useState("");
  const [mobileInspectorOpen, setMobileInspectorOpen] = useState(false);
  const [baseline, setBaseline] = useState<{
    name: string;
    parameters: number;
    memory: number;
    throughput: number;
    hours: number;
    cost: number;
  } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  const rf = useReactFlow();
  const architecture = useMemo(() => {
    const connected = new Set(edges.flatMap((e) => [e.source, e.target]));
    const count = (kind: Kind) =>
      nodes.filter((n) => n.data.kind === kind && connected.has(n.id)).length;
    return {
      transformer: count("transformer"),
      attention: count("attention"),
      ffn: count("ffn"),
      embedding: count("embedding"),
      head: count("head"),
    };
  }, [nodes, edges]);
  const metrics = useMemo(
    () => simulate(config, architecture),
    [config, architecture],
  );
  const live = useCallback(
    (value: number, amplitude = 1, phase = 0) =>
      value + Math.sin(liveTick * 0.78 + phase) * amplitude,
    [liveTick],
  );
  const animatedMetrics = useMemo(() => {
    const memory = {
      ...metrics.memory,
      parameters: Math.max(0, live(metrics.memory.parameters, 0.16, 0.4)),
      gradients: Math.max(0, live(metrics.memory.gradients, 0.14, 1.2)),
      optimizer: Math.max(0, live(metrics.memory.optimizer, 0.12, 2.1)),
      activations: Math.max(0, live(metrics.memory.activations, 0.2, 2.8)),
      temporary: Math.max(0, live(metrics.memory.temporary, 0.08, 3.4)),
      runtime: Math.max(0, live(metrics.memory.runtime, 0.04, 4.1)),
    };
    memory.total =
      memory.parameters +
      memory.gradients +
      memory.optimizer +
      memory.activations +
      memory.temporary +
      memory.runtime;
    memory.free = config.hardware.vram - memory.total;
    return {
      ...metrics,
      parameters: Math.max(
        0,
        live(
          metrics.parameters,
          Math.max(25_000_000, metrics.parameters * 0.003),
          0.7,
        ),
      ),
      memory,
      utilization: Math.max(
        0,
        Math.min(1, live(metrics.utilization, 0.008, 1.3)),
      ),
      mfu: Math.max(0, Math.min(1, live(metrics.mfu, 0.006, 2.4))),
      bandwidthUtil: Math.max(
        0,
        Math.min(1, live(metrics.bandwidthUtil, 0.01, 3.1)),
      ),
      tokensPerSecond: Math.max(
        1,
        live(
          metrics.tokensPerSecond,
          Math.max(1, metrics.tokensPerSecond * 0.012),
          1.7,
        ),
      ),
      cost: Math.max(
        0,
        live(metrics.cost, Math.max(0.15, metrics.cost * 0.004), 2.9),
      ),
      energyKWh: Math.max(
        0,
        live(metrics.energyKWh, Math.max(0.05, metrics.energyKWh * 0.005), 3.6),
      ),
      attentionMatrixGB: Math.max(
        0,
        live(metrics.attentionMatrixGB, 0.08, 4.4),
      ),
      breakdown: {
        ...metrics.breakdown,
        embeddings: Math.max(
          0,
          live(
            metrics.breakdown.embeddings,
            Math.max(5_000_000, metrics.breakdown.embeddings * 0.003),
            0.9,
          ),
        ),
        attention: Math.max(
          0,
          live(
            metrics.breakdown.attention,
            Math.max(5_000_000, metrics.breakdown.attention * 0.003),
            1.5,
          ),
        ),
        ffn: Math.max(
          0,
          live(
            metrics.breakdown.ffn,
            Math.max(5_000_000, metrics.breakdown.ffn * 0.003),
            2.1,
          ),
        ),
        norm: Math.max(
          0,
          live(
            metrics.breakdown.norm,
            Math.max(5_000_000, metrics.breakdown.norm * 0.003),
            2.7,
          ),
        ),
        lmHead: Math.max(
          0,
          live(
            metrics.breakdown.lmHead,
            Math.max(5_000_000, metrics.breakdown.lmHead * 0.003),
            3.3,
          ),
        ),
      },
    };
  }, [config.hardware.vram, live, metrics]);
  const training = useMemo(
    () => trainingAtProgress(config, metrics, progress / 100),
    [config, metrics, progress],
  );
  const update = <K extends Exclude<keyof Config, "seed">>(
    key: K,
    patch: Partial<Config[K]>,
  ) => setConfig((prev) => ({ ...prev, [key]: { ...prev[key], ...patch } }));
  const chooseModel = useCallback(
    (name: string) => {
      const preset = modelLibrary[name];
      if (!preset) return;
      setConfig((prev) => ({
        ...prev,
        model: { ...initialConfig.model, ...preset.model, name },
        tokenizer: { ...initialConfig.tokenizer, ...preset.tokenizer },
        dataset: { ...initialConfig.dataset, ...(preset.dataset || {}) },
      }));
      setNodes(
        initialNodes.map((node) => ({ ...node, data: { ...node.data } })),
      );
      setEdges(initialEdges);
      setSelected("transformer");
      setActivePanel("inspect");
      setProgress(0);
      setPlaying(false);
      setShowPresets(false);
      flash(`${name} architecture loaded`);
    },
    [setEdges, setNodes],
  );
  useEffect(() => {
    const id = window.setInterval(() => setLiveTick((tick) => tick + 1), 180);
    return () => window.clearInterval(id);
  }, []);
  useEffect(() => {
    try {
      const raw = localStorage.getItem("tensorforge-project");
      if (raw) {
        const p = JSON.parse(raw);
        if (p.config) setConfig(p.config);
        if (p.nodes) setNodes(p.nodes);
        if (p.edges) setEdges(p.edges);
      }
    } catch {}
  }, [setNodes, setEdges]);
  useEffect(() => {
    if (progress >= 100 && playing) setPlaying(false);
  }, [progress, playing]);
  useEffect(() => {
    if (!playing) return;
    const delay =
      speed === "Instant"
        ? 30
        : speed === "1000×"
          ? 50
          : speed === "100×"
            ? 100
            : speed === "10×"
              ? 240
              : 600;
    const id = setInterval(
      () => setProgress((v) => Math.min(100, v + 1)),
      delay,
    );
    return () => clearInterval(id);
  }, [playing, speed]);
  const connectedCount = edges.length;
  const displayEdges = useMemo(() => {
    const base = edges.map((e) => ({
      ...e,
      style: {
        ...e.style,
        stroke:
          e.source === "gpu" || e.source === "cluster" ? "#e8b66a" : "#8ddac3",
      },
    }));
    if (!playing) return base;
    return [
      ...base,
      ...edges
        .filter((e) => e.source !== "gpu" && e.source !== "dataset")
        .map((e) => ({
          ...e,
          id: "back-" + e.id,
          source: e.target,
          target: e.source,
          animated: true,
          selectable: false,
          style: {
            stroke: "#afa9ef",
            strokeWidth: 1.5,
            strokeDasharray: "3 8",
            opacity: 0.8,
          },
          markerEnd: { type: MarkerType.ArrowClosed, color: "#afa9ef" },
        })),
    ];
  }, [edges, playing]);
  const displayNodes = useMemo(
    () =>
      nodes.map((n) => {
        const kind = n.data.kind;
        let metric = n.data.metric,
          detail = n.data.detail,
          subtitle = n.data.subtitle,
          tensor = "";
        if (kind === "transformer") {
          metric =
            format(
              animatedMetrics.breakdown.attention +
                animatedMetrics.breakdown.ffn,
            ) + " params";
          detail = `${config.model.layers} layers`;
          subtitle = `${config.model.layers} layers · ${config.model.attention}`;
          tensor = `[B, ${config.model.sequence}, ${config.model.hidden}]`;
        }
        if (kind === "embedding") {
          metric = format(animatedMetrics.breakdown.embeddings) + " params";
          detail = `${((animatedMetrics.breakdown.embeddings * 2) / 1024 ** 3).toFixed(1)} GB`;
          tensor = `[B, ${config.model.sequence}, ${config.model.hidden}]`;
        }
        if (kind === "tokenizer") {
          subtitle = `${config.tokenizer.type} · ${format(config.model.vocab, 0)} vocabulary`;
          metric = `${format(config.model.vocab, 0)} tokens`;
          tensor = `[B, ${config.model.sequence}]`;
        }
        if (kind === "head") {
          metric = config.model.tiedEmbeddings
            ? "Tied weights"
            : format(animatedMetrics.breakdown.lmHead) + " params";
          tensor = `[B, ${config.model.sequence}, ${config.model.vocab}]`;
        }
        if (kind === "dataset") {
          subtitle = `${format(config.dataset.tokens)} tokens · ${config.dataset.epochs} epoch`;
          metric = `${format(animatedMetrics.totalTokens)} total`;
          detail = `${format(animatedMetrics.steps)} steps`;
        }
        if (kind === "gpu" || kind === "cluster") {
          subtitle = `${animatedMetrics.gpus} × ${config.hardware.gpu}`;
          metric = `${Math.round((animatedMetrics.memory.total / config.hardware.vram) * 100)}%`;
          detail = `${Math.round(animatedMetrics.utilization * 100)}% compute`;
        }
        if (kind === "optimizer") {
          subtitle = config.training.optimizer + " · " + config.model.precision;
          metric = `${animatedMetrics.memory.optimizer.toFixed(1)} GB states`;
        }
        if (kind === "attention") {
          metric = `${animatedMetrics.attentionMatrixGB.toFixed(1)} GB matrix`;
          tensor = `[B, ${config.model.heads}, T, T]`;
        }
        if (kind === "ffn")
          metric = format(animatedMetrics.breakdown.ffn) + " params";
        if (kind === "input") tensor = `[B, ${config.model.sequence}]`;
        const phaseIndex = progress % 12;
        const activeKinds: Kind[][] = [
          ["input", "dataset"],
          ["tokenizer"],
          ["embedding", "position"],
          ["attention", "transformer"],
          ["ffn", "transformer"],
          ["head", "loss"],
          ["loss", "head"],
          ["transformer", "ffn"],
          ["transformer", "attention"],
          ["embedding"],
          ["gpu", "cluster"],
          ["optimizer", "gpu"],
        ];
        return {
          ...n,
          data: {
            ...n.data,
            subtitle: tensorMode && tensor ? tensor : subtitle,
            metric,
            detail,
            executing: playing && activeKinds[phaseIndex].includes(kind),
            warning:
              (kind === "gpu" || kind === "cluster") && animatedMetrics.oom,
          },
        };
      }),
    [nodes, config, animatedMetrics, tensorMode, playing, progress],
  );
  const onConnect = useCallback(
    (connection: Connection) =>
      setEdges((eds) =>
        addEdge(
          {
            ...connection,
            type: "smoothstep",
            animated: true,
            markerEnd: { type: MarkerType.ArrowClosed, color: "#8ddac3" },
            style: { stroke: "#8ddac3", strokeWidth: 2 },
          },
          eds,
        ),
      ),
    [setEdges],
  );
  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const kind = e.dataTransfer.getData(
        "application/tensorforge-kind",
      ) as Kind;
      const title = e.dataTransfer.getData("application/tensorforge-title");
      if (!kind) return;
      const position = rf.screenToFlowPosition({ x: e.clientX, y: e.clientY });
      const id = `${kind}-${Date.now()}`;
      setNodes((ns) => [
        ...ns,
        {
          id,
          type: "studio",
          position,
          data: { kind, title, subtitle: "Configure in inspector" },
        },
      ]);
      setSelected(id);
      setActivePanel("inspect");
    },
    [rf, setNodes],
  );
  const save = () => {
    localStorage.setItem(
      "tensorforge-project",
      JSON.stringify({ config, nodes, edges }),
    );
    flash("Project saved locally");
  };
  const flash = (s: string) => {
    setNotice(s);
    setTimeout(() => setNotice(""), 3500);
  };
  const download = (
    name: string,
    content: string,
    type = "application/json",
  ) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([content], { type }));
    a.download = name;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  const exportProject = () =>
    download(
      "tensorforge-project.json",
      JSON.stringify({ version: 1, config, nodes, edges }, null, 2),
    );
  const report = () =>
    download(
      "training-report.json",
      JSON.stringify(
        {
          configuration: config,
          metrics,
          generatedAt: new Date().toISOString(),
        },
        null,
        2,
      ),
    );
  const share = () => {
    const data = btoa(
      unescape(encodeURIComponent(JSON.stringify({ config, nodes, edges }))),
    );
    const url = new URL(location.href);
    url.hash = "project=" + encodeURIComponent(data);
    navigator.clipboard
      .writeText(url.toString())
      .then(() => flash("Share link copied"))
      .catch(() => flash("Could not copy link"));
  };
  useEffect(() => {
    if (location.hash.startsWith("#project=")) {
      try {
        const p = JSON.parse(
          decodeURIComponent(
            escape(atob(decodeURIComponent(location.hash.slice(9)))),
          ),
        );
        if (p.config) setConfig(p.config);
        if (p.nodes) setNodes(p.nodes);
        if (p.edges) setEdges(p.edges);
        flash("Shared project loaded");
      } catch {}
    }
  }, [setNodes, setEdges]);
  const importProject = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    try {
      const p = JSON.parse(await f.text());
      if (!p.config || !p.nodes || !p.edges) throw Error();
      setConfig(p.config);
      setNodes(p.nodes);
      setEdges(p.edges);
      flash("Project imported");
    } catch {
      flash("Invalid project JSON");
    }
    e.target.value = "";
  };
  const exportImage = async () => {
    if (!canvasRef.current) return;
    try {
      const { toPng } = await import("html-to-image");
      const data = await toPng(canvasRef.current, {
        backgroundColor: "#151d20",
        pixelRatio: 2,
      });
      const a = document.createElement("a");
      a.href = data;
      a.download = "tensorforge-architecture.png";
      a.click();
    } catch {
      flash("Image export failed");
    }
  };
  const selectedNode = displayNodes.find((n) => n.id === selected);
  const selectedEdge = edges.find((e) => e.id === selected);
  const sk = selectedNode?.data.kind;
  const tabs: [typeof activePanel, string][] = [
    ["inspect", "Inspector"],
    ["model", "Model"],
    ["dataset", "Dataset"],
    ["training", "Training"],
    ["hardware", "Hardware"],
    ["distributed", "Parallelism"],
  ];
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Image src="/logo.png" alt="logo" fill />
          </div>
          <div>
            <strong>tensorforge</strong>
            <small>TRAINING STUDIO</small>
          </div>
        </div>
        <div className="top-separator" />
        <div className="model-picker-wrap">
          <button
            className="model-picker"
            onClick={() => setShowPresets((open) => !open)}
            aria-expanded={showPresets}
            aria-haspopup="menu"
          >
            <span className="status-dot" />
            <span className="model-picker-copy">
              <small>MODEL ARCHITECTURE</small>
              <strong>{config.model.name}</strong>
            </span>
            <ChevronDown size={14} />
          </button>
          {showPresets && (
            <div className="model-preset-menu" role="menu">
              <div className="model-preset-heading">
                <span>REFERENCE MODELS</span>
                <small>Replaces model, tokenizer + corpus fields</small>
              </div>
              {Object.entries(modelLibrary).map(([name, preset]) => {
                const ProviderIcon = providerIcons[preset.provider] || TbBrain;
                return (
                  <button
                    key={name}
                    className={
                      "model-preset-option " +
                      (config.model.name === name ? "active" : "")
                    }
                    role="menuitem"
                    onClick={() => chooseModel(name)}
                  >
                    <span className="preset-brand">
                      <ProviderIcon size={17} aria-hidden="true" />
                    </span>
                    <span>
                      <strong>{name}</strong>
                      <small>{preset.family}</small>
                    </span>
                    <em>{preset.model.layers}L</em>
                  </button>
                );
              })}
            </div>
          )}
        </div>
        <div className="top-metrics">
          <div>
            <small>PARAMETERS</small>
            <strong className="telemetry-value">
              {format(animatedMetrics.parameters, 2)}
            </strong>
          </div>
          <div>
            <small>GPU MEMORY</small>
            <strong className={animatedMetrics.oom ? "error" : ""}>
              <span className="telemetry-value">
                {animatedMetrics.memory.total.toFixed(1)}
              </span>{" "}
              <span>/ {config.hardware.vram} GB</span>
            </strong>
          </div>
          <div>
            <small>THROUGHPUT</small>
            <strong>
              <span className="telemetry-value">
                {format(animatedMetrics.tokensPerSecond, 2)}
              </span>{" "}
              <span>tok/s</span>
            </strong>
          </div>
          <div>
            <small>EST. COST</small>
            <strong className="telemetry-value">
              {money(animatedMetrics.cost)}
            </strong>
          </div>
        </div>
        <div className="run-controls">
          <button
            className="mobile-inspector-toggle"
            onClick={() => setMobileInspectorOpen((v) => !v)}
            aria-label="Toggle inspector"
          >
            <Settings2 size={17} />
          </button>
          <button
            className="secondary-btn"
            onClick={() => {
              setProgress(0);
              setPlaying(false);
            }}
            title="Reset simulation"
          >
            <RotateCcw size={15} />
          </button>
          <button
            className="run-btn"
            onClick={() => {
              if (progress >= 100) setProgress(0);
              setPlaying(!playing);
            }}
          >
            {playing ? (
              <Pause size={15} fill="currentColor" />
            ) : (
              <Play size={15} fill="currentColor" />
            )}
            {playing ? "Pause" : "Simulate"}
          </button>
        </div>
      </header>
      <div className="workspace">
        <aside className="palette">
          <div className="side-title">
            <span>COMPONENTS</span>
            <span className="count">
              {catalog.reduce((a, g) => a + g.items.length, 0)}
            </span>
          </div>
          <div className="search">
            <Search size={15} />
            <input
              placeholder="Search components"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <div className="palette-scroll">
            {catalog.map((g) => {
              const items = g.items.filter((i) =>
                i.title.toLowerCase().includes(search.toLowerCase()),
              );
              return items.length ? (
                <div className="palette-group" key={g.group}>
                  <div className="group-label">{g.group}</div>
                  {items.map((i) => (
                    <div
                      key={i.title}
                      draggable
                      onDragStart={(e) => {
                        e.dataTransfer.setData(
                          "application/tensorforge-kind",
                          i.kind,
                        );
                        e.dataTransfer.setData(
                          "application/tensorforge-title",
                          i.title,
                        );
                        e.dataTransfer.effectAllowed = "move";
                      }}
                      className={"palette-item " + i.kind}
                      onClick={() => {
                        const id = `${i.kind}-${Date.now()}`;
                        setNodes((ns) => [
                          ...ns,
                          {
                            id,
                            type: "studio",
                            position: rf.screenToFlowPosition({
                              x: window.innerWidth / 2,
                              y: window.innerHeight / 2,
                            }),
                            data: {
                              kind: i.kind,
                              title: i.title,
                              subtitle: i.desc,
                            },
                          },
                        ]);
                        setSelected(id);
                      }}
                    >
                      <span className="palette-icon">
                        <KindIcon kind={i.kind} size={18} />
                      </span>
                      <span>
                        <b>{i.title}</b>
                        <small>{i.desc}</small>
                      </span>
                      <Plus size={14} className="add-icon" />
                    </div>
                  ))}
                </div>
              ) : null;
            })}
          </div>
          <div className="palette-bottom">
            <Sparkles size={14} />
            <span>Drag onto canvas or click to add</span>
          </div>
        </aside>
        <main
          className="canvas-area"
          ref={canvasRef}
          onDrop={onDrop}
          onDragOver={(e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = "move";
          }}
        >
          <ReactFlow
            nodes={displayNodes}
            edges={displayEdges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={(_, n) => {
              setSelected(n.id);
              setActivePanel("inspect");
              setMobileInspectorOpen(true);
            }}
            onEdgeClick={(_, e) => {
              setSelected(e.id);
              setActivePanel("inspect");
              setMobileInspectorOpen(true);
            }}
            onPaneClick={() => {
              setSelected(null);
              setMobileInspectorOpen(false);
            }}
            fitView
            fitViewOptions={{ padding: 0.18, maxZoom: 1.1 }}
            minZoom={0.3}
            maxZoom={1.7}
            proOptions={{ hideAttribution: true }}
          >
            <Background color="#2c3a3b" gap={28} size={1} />
            <Controls position="bottom-left" showInteractive={false} />
            <MiniMap
              position="bottom-right"
              nodeColor={(n) =>
                n.data?.kind === "gpu"
                  ? "#e8b66a"
                  : n.data?.kind === "transformer"
                    ? "#a9a3f2"
                    : "#78ceb5"
              }
              maskColor="rgba(11,18,20,.6)"
            />
          </ReactFlow>
          <div className="canvas-label">
            <span className="live-dot" /> MODEL GRAPH{" "}
            <span className="label-divider">/</span> {nodes.length} components ·{" "}
            {connectedCount} connections
          </div>
          <div className="canvas-tools">
            <button
              className={tensorMode ? "active" : ""}
              onClick={() => setTensorMode(!tensorMode)}
            >
              <Braces size={15} /> Tensor shapes
            </button>
            <button onClick={() => rf.fitView({ padding: 0.2, duration: 250 })}>
              <Maximize2 size={15} /> Fit view
            </button>
          </div>
          <div className="flow-legend">
            <span>
              <i className="forward" />
              Forward pass
            </span>
            <span>
              <i className="backward" />
              Backward pass
            </span>
            <span>
              <i className="hardware" />
              Hardware link
            </span>
          </div>
          {playing && (
            <div className="simulation-indicator">
              <span className="pulse" /> Step{" "}
              {training.completedSteps.toLocaleString()} /{" "}
              {animatedMetrics.steps.toLocaleString()}
              <div className="phase-label">
                {progress % 12 < 6
                  ? "FORWARD PASS"
                  : progress % 12 < 10
                    ? "BACKWARD PASS"
                    : progress % 12 === 10
                      ? "ALL-REDUCE"
                      : "OPTIMIZER STEP"}
              </div>
              <div className="training-live">
                <span>loss {training.loss.toFixed(2)}</span>
                <span>{format(training.processedTokens)} tokens</span>
                <span>{money(training.spent)} spent</span>
              </div>
              <div className="progress-line">
                <span style={{ width: `${progress}%` }} />
              </div>
            </div>
          )}
        </main>
        <aside
          className={`inspector ${mobileInspectorOpen ? "mobile-open" : ""}`}
        >
          <div className="inspector-top">
            <div>
              <span className="eyebrow">CONFIGURATION</span>
              <h2>
                {activePanel === "inspect"
                  ? selectedNode?.data.title ||
                    (selectedEdge ? "Tensor transfer" : "Overview")
                  : tabs.find((t) => t[0] === activePanel)?.[1]}
              </h2>
            </div>
            <Settings2 size={17} />
          </div>
          <div className="inspector-nav">
            {tabs.map(([key, label]) => (
              <button
                key={key}
                className={activePanel === key ? "active" : ""}
                onClick={() => setActivePanel(key)}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="inspector-scroll">
            {activePanel === "inspect" && (
              <>
                {selectedNode ? (
                  <>
                    <div className="selection-summary">
                      <span className={"summary-icon " + sk}>
                        <KindIcon kind={sk!} size={19} />
                      </span>
                      <div>
                        <strong>{selectedNode.data.title}</strong>
                        <small>{selectedNode.data.subtitle}</small>
                      </div>
                    </div>
                    {sk === "gpu" || sk === "cluster" ? (
                      <>
                        <div
                          className={
                            "health-card " +
                            (animatedMetrics.oom ? "danger" : "")
                          }
                        >
                          <div>
                            <MemoryStick size={17} />
                            <strong>
                              {animatedMetrics.oom
                                ? "Out of memory"
                                : "Memory healthy"}
                            </strong>
                          </div>
                          <p>
                            {animatedMetrics.memory.total.toFixed(1)} GB
                            required on each {config.hardware.vram} GB GPU.{" "}
                            {animatedMetrics.oom
                              ? `${Math.abs(animatedMetrics.memory.free).toFixed(1)} GB over capacity.`
                              : `${animatedMetrics.memory.free.toFixed(1)} GB free.`}
                          </p>
                        </div>
                        <Section title="GPU memory">
                          <Bar
                            label="Parameters"
                            value={animatedMetrics.memory.parameters}
                            total={config.hardware.vram}
                            color="#a9a3f2"
                          />
                          <Bar
                            label="Gradients"
                            value={animatedMetrics.memory.gradients}
                            total={config.hardware.vram}
                            color="#89bcf2"
                          />
                          <Bar
                            label="Optimizer"
                            value={animatedMetrics.memory.optimizer}
                            total={config.hardware.vram}
                            color="#e8b66a"
                          />
                          <Bar
                            label="Activations"
                            value={animatedMetrics.memory.activations}
                            total={config.hardware.vram}
                            color="#78ceb5"
                          />
                          <Bar
                            label="Temporary"
                            value={
                              animatedMetrics.memory.temporary +
                              animatedMetrics.memory.runtime
                            }
                            total={config.hardware.vram}
                            color="#d99ba9"
                          />
                        </Section>
                        <button
                          className="wide-link"
                          onClick={() => setActivePanel("hardware")}
                        >
                          Configure hardware <ChevronRight size={15} />
                        </button>
                      </>
                    ) : sk === "dataset" ? (
                      <>
                        <Section title="Dataset">
                          <Field
                            label="Training tokens"
                            value={config.dataset.tokens}
                            onChange={(v) => update("dataset", { tokens: v })}
                          />
                          <Field
                            label="Epochs"
                            value={config.dataset.epochs}
                            onChange={(v) => update("dataset", { epochs: v })}
                          />
                          <div className="readout">
                            <span>Total steps</span>
                            <strong>{format(animatedMetrics.steps)}</strong>
                          </div>
                        </Section>
                        <button
                          className="wide-link"
                          onClick={() => setActivePanel("dataset")}
                        >
                          All dataset settings <ChevronRight size={15} />
                        </button>
                      </>
                    ) : sk === "tokenizer" ? (
                      <>
                        <Section title="Tokenizer">
                          <SelectField
                            label="Algorithm"
                            value={config.tokenizer.type}
                            options={[
                              "BPE",
                              "WordPiece",
                              "SentencePiece",
                              "Unigram",
                              "Byte-level",
                              "Character",
                              "Custom",
                            ]}
                            onChange={(v) => update("tokenizer", { type: v })}
                          />
                          <Field
                            label="Vocabulary size"
                            value={config.model.vocab}
                            onChange={(v) => update("model", { vocab: v })}
                          />
                          <Field
                            label="Special tokens"
                            value={config.tokenizer.specialTokens}
                            onChange={(v) =>
                              update("tokenizer", { specialTokens: v })
                            }
                          />
                          <Field
                            label="Characters / token"
                            value={config.tokenizer.charsPerToken}
                            onChange={(v) =>
                              update("tokenizer", { charsPerToken: v })
                            }
                            step={0.1}
                          />
                          <Field
                            label="Bytes / token"
                            value={config.tokenizer.bytesPerToken}
                            onChange={(v) =>
                              update("tokenizer", { bytesPerToken: v })
                            }
                            step={0.1}
                          />
                        </Section>
                      </>
                    ) : (
                      <>
                        <Section title="Component details">
                          <div className="readout">
                            <span>Parameters</span>
                            <strong>
                              {sk === "transformer"
                                ? format(
                                    animatedMetrics.breakdown.attention +
                                      animatedMetrics.breakdown.ffn,
                                  )
                                : sk === "embedding"
                                  ? format(animatedMetrics.breakdown.embeddings)
                                  : sk === "attention"
                                    ? format(
                                        animatedMetrics.breakdown.attention,
                                      )
                                    : sk === "ffn"
                                      ? format(animatedMetrics.breakdown.ffn)
                                      : format(animatedMetrics.parameters)}
                            </strong>
                          </div>
                          <div className="readout">
                            <span>Precision</span>
                            <strong>{config.model.precision}</strong>
                          </div>
                          <div className="readout">
                            <span>Tensor shape</span>
                            <strong className="mono">
                              {selectedNode.data.subtitle?.startsWith("[")
                                ? selectedNode.data.subtitle
                                : `[B, ${config.model.sequence}, ${config.model.hidden}]`}
                            </strong>
                          </div>
                          <div className="readout">
                            <span>Forward time</span>
                            <strong>
                              {(animatedMetrics.forwardSeconds * 1000).toFixed(
                                0,
                              )}{" "}
                              ms
                            </strong>
                          </div>
                        </Section>
                        {(sk === "attention" || sk === "transformer") && (
                          <Section title="Attention path">
                            <div className="stage-list">
                              {[
                                ["Input", `[B, T, ${config.model.hidden}]`],
                                [
                                  "Q projection",
                                  `[B, T, ${config.model.heads} × ${config.model.hidden / config.model.heads}]`,
                                ],
                                [
                                  "K / V projection",
                                  `[B, T, ${config.model.kvHeads} × ${config.model.hidden / config.model.heads}]`,
                                ],
                                [
                                  "Scores QKᵀ",
                                  `[B, ${config.model.heads}, T, T]`,
                                ],
                                [
                                  "Softmax + V",
                                  `[B, T, ${config.model.hidden}]`,
                                ],
                              ].map(([name, shape], i) => (
                                <div key={name}>
                                  <span>{i + 1}</span>
                                  <strong>{name}</strong>
                                  <code>{shape}</code>
                                </div>
                              ))}
                            </div>
                            <div className="readout total">
                              <span>Score matrix per batch</span>
                              <strong>
                                {animatedMetrics.attentionMatrixGB.toFixed(2)}{" "}
                                GB
                              </strong>
                            </div>
                            <p className="section-note">
                              Doubling sequence length makes the score matrix 4×
                              larger. {config.model.attention} reduces
                              materialized activation memory.
                            </p>
                          </Section>
                        )}
                        <Section title="Architecture">
                          <Field
                            label="Hidden size"
                            value={config.model.hidden}
                            onChange={(v) => update("model", { hidden: v })}
                          />
                          <Field
                            label="Layers"
                            value={config.model.layers}
                            onChange={(v) => update("model", { layers: v })}
                          />
                          <Field
                            label="Attention heads"
                            value={config.model.heads}
                            onChange={(v) => update("model", { heads: v })}
                          />
                          <Field
                            label="KV heads"
                            value={config.model.kvHeads}
                            onChange={(v) => update("model", { kvHeads: v })}
                          />
                          <SelectField
                            label="Attention"
                            value={config.model.attention}
                            options={[
                              "Standard",
                              "FlashAttention",
                              "Memory-efficient",
                              "Grouped query",
                              "Multi-query",
                              "MLA",
                              "Sliding window",
                              "Sparse",
                            ]}
                            onChange={(v) =>
                              update("model", {
                                attention: v as Config["model"]["attention"],
                              })
                            }
                          />
                        </Section>
                        <button
                          className="wide-link"
                          onClick={() => setActivePanel("model")}
                        >
                          All model settings <ChevronRight size={15} />
                        </button>
                      </>
                    )}
                    {selectedNode && (
                      <button
                        className="delete-btn"
                        onClick={() => {
                          setNodes((ns) => ns.filter((n) => n.id !== selected));
                          setEdges((es) =>
                            es.filter(
                              (e) =>
                                e.source !== selected && e.target !== selected,
                            ),
                          );
                          setSelected(null);
                        }}
                      >
                        Remove component
                      </button>
                    )}
                  </>
                ) : selectedEdge ? (
                  <>
                    <div className="selection-summary">
                      <span className="summary-icon">
                        <TbArrowsShuffle size={19} aria-hidden="true" />
                      </span>
                      <div>
                        <strong>Tensor transfer</strong>
                        <small>
                          {selectedEdge.source} → {selectedEdge.target}
                        </small>
                      </div>
                    </div>
                    <Section title="Connection">
                      <div className="readout">
                        <span>Shape</span>
                        <strong className="mono">
                          [B, {config.model.sequence}, {config.model.hidden}]
                        </strong>
                      </div>
                      <div className="readout">
                        <span>Dtype</span>
                        <strong>{config.model.precision}</strong>
                      </div>
                      <div className="readout">
                        <span>Payload</span>
                        <strong>
                          {(
                            (config.training.microBatch *
                              config.model.sequence *
                              config.model.hidden *
                              (config.model.precision === "FP32" ? 4 : 2)) /
                            1024 ** 2
                          ).toFixed(1)}{" "}
                          MB
                        </strong>
                      </div>
                    </Section>
                    <button
                      className="delete-btn"
                      onClick={() => {
                        setEdges((es) => es.filter((e) => e.id !== selected));
                        setSelected(null);
                      }}
                    >
                      Remove connection
                    </button>
                  </>
                ) : (
                  <div className="empty-inspector">
                    <Workflow size={26} />
                    <strong>Select a component</strong>
                    <p>
                      Inspect tensor shapes, memory use, and editable settings
                      here.
                    </p>
                  </div>
                )}
              </>
            )}
            {activePanel === "model" && (
              <>
                <Section title="Architecture preset">
                  <SelectField
                    label="Preset"
                    value="Choose a preset"
                    options={["Choose a preset", ...Object.keys(modelPresets)]}
                    onChange={(v) => {
                      if (modelLibrary[v]) {
                        chooseModel(v);
                      } else if (modelPresets[v]) {
                        setConfig((prev) => ({
                          ...prev,
                          model: {
                            ...initialConfig.model,
                            ...modelPresets[v],
                            name: v,
                          },
                        }));
                        setProgress(0);
                      }
                    }}
                  />
                </Section>
                <Section title="Dimensions">
                  <Field
                    label="Model name"
                    value={config.model.name}
                    onChange={(v) => update("model", { name: v })}
                  />
                  <Field
                    label="Vocabulary"
                    value={config.model.vocab}
                    onChange={(v) => update("model", { vocab: v })}
                  />
                  <Field
                    label="Hidden size"
                    value={config.model.hidden}
                    onChange={(v) => update("model", { hidden: v })}
                  />
                  <Field
                    label="Layers"
                    value={config.model.layers}
                    onChange={(v) => update("model", { layers: v })}
                  />
                  <Field
                    label="Attention heads"
                    value={config.model.heads}
                    onChange={(v) => update("model", { heads: v })}
                  />
                  <Field
                    label="KV heads"
                    value={config.model.kvHeads}
                    onChange={(v) => update("model", { kvHeads: v })}
                  />
                  <Field
                    label="FFN size"
                    value={config.model.ffn}
                    onChange={(v) => update("model", { ffn: v })}
                  />
                  {config.model.experts ? (
                    <>
                      <Field
                        label="Experts"
                        value={config.model.experts}
                        onChange={(v) => update("model", { experts: v })}
                      />
                      <Field
                        label="Active experts"
                        value={config.model.activeExperts || 1}
                        onChange={(v) => update("model", { activeExperts: v })}
                      />
                    </>
                  ) : null}
                  <Field
                    label="Sequence length"
                    value={config.model.sequence}
                    onChange={(v) => update("model", { sequence: v })}
                  />
                </Section>
                <Section title="Layer choices">
                  <SelectField
                    label="Attention"
                    value={config.model.attention}
                    options={[
                      "Standard",
                      "FlashAttention",
                      "Memory-efficient",
                      "Grouped query",
                      "Multi-query",
                      "MLA",
                      "Sliding window",
                      "Sparse",
                    ]}
                    onChange={(v) =>
                      update("model", {
                        attention: v as Config["model"]["attention"],
                      })
                    }
                  />
                  <SelectField
                    label="Activation"
                    value={config.model.activation}
                    options={["SwiGLU", "GELU", "GeGLU", "ReLU"]}
                    onChange={(v) =>
                      update("model", {
                        activation: v as Config["model"]["activation"],
                      })
                    }
                  />
                  <SelectField
                    label="Norm"
                    value={config.model.norm}
                    options={["RMSNorm", "LayerNorm"]}
                    onChange={(v) =>
                      update("model", { norm: v as Config["model"]["norm"] })
                    }
                  />
                  <SelectField
                    label="Position"
                    value={config.model.position}
                    options={["RoPE", "Learned", "Sinusoidal", "ALiBi"]}
                    onChange={(v) =>
                      update("model", {
                        position: v as Config["model"]["position"],
                      })
                    }
                  />
                  <SelectField
                    label="Precision"
                    value={config.model.precision}
                    options={["FP32", "FP16", "BF16", "FP8", "Mixed"]}
                    onChange={(v) =>
                      update("model", {
                        precision: v as Config["model"]["precision"],
                      })
                    }
                  />
                  <Field
                    label="Dropout"
                    value={config.model.dropout}
                    onChange={(v) => update("model", { dropout: v })}
                    step={0.01}
                  />
                  <label className="switch-row">
                    <span>Tie input and output embeddings</span>
                    <input
                      type="checkbox"
                      checked={config.model.tiedEmbeddings}
                      onChange={(e) =>
                        update("model", { tiedEmbeddings: e.target.checked })
                      }
                    />
                  </label>
                </Section>
                <Section title="Parameter explorer">
                  <div className="readout">
                    <span>Embeddings</span>
                    <strong>
                      {format(animatedMetrics.breakdown.embeddings)}
                    </strong>
                  </div>
                  <div className="readout">
                    <span>Attention</span>
                    <strong>
                      {format(animatedMetrics.breakdown.attention)}
                    </strong>
                  </div>
                  <div className="readout">
                    <span>Feed forward</span>
                    <strong>{format(animatedMetrics.breakdown.ffn)}</strong>
                  </div>
                  <div className="readout">
                    <span>Normalization</span>
                    <strong>{format(animatedMetrics.breakdown.norm)}</strong>
                  </div>
                  <div className="readout total">
                    <span>Total</span>
                    <strong>{format(animatedMetrics.parameters)}</strong>
                  </div>
                </Section>
              </>
            )}
            {activePanel === "dataset" && (
              <>
                <Section title="Corpus">
                  <Field
                    label="Dataset name"
                    value={config.dataset.name}
                    onChange={(v) => update("dataset", { name: v })}
                  />
                  <Field
                    label="Available tokens"
                    value={config.dataset.tokens}
                    onChange={(v) => update("dataset", { tokens: v })}
                  />
                  <Field
                    label="Average document"
                    value={config.dataset.avgDocument}
                    onChange={(v) => update("dataset", { avgDocument: v })}
                    unit="tokens"
                  />
                  <Field
                    label="Epochs"
                    value={config.dataset.epochs}
                    onChange={(v) => update("dataset", { epochs: v })}
                  />
                </Section>
                <Section title="Tokenizer">
                  <SelectField
                    label="Algorithm"
                    value={config.tokenizer.type}
                    options={[
                      "BPE",
                      "WordPiece",
                      "SentencePiece",
                      "Unigram",
                      "Byte-level",
                      "Character",
                      "Custom",
                    ]}
                    onChange={(v) => update("tokenizer", { type: v })}
                  />
                  <Field
                    label="Characters / token"
                    value={config.tokenizer.charsPerToken}
                    onChange={(v) => update("tokenizer", { charsPerToken: v })}
                    step={0.1}
                  />
                  <Field
                    label="Bytes / token"
                    value={config.tokenizer.bytesPerToken}
                    onChange={(v) => update("tokenizer", { bytesPerToken: v })}
                    step={0.1}
                  />
                  <Field
                    label="Special tokens"
                    value={config.tokenizer.specialTokens}
                    onChange={(v) => update("tokenizer", { specialTokens: v })}
                  />
                </Section>
                <Section title="Derived">
                  <div className="readout">
                    <span>Tokens / step</span>
                    <strong>
                      {format(
                        config.training.globalBatch * config.model.sequence,
                      )}
                    </strong>
                  </div>
                  <div className="readout">
                    <span>Total steps</span>
                    <strong>{format(animatedMetrics.steps)}</strong>
                  </div>
                  <div className="readout">
                    <span>Total tokens</span>
                    <strong>{format(animatedMetrics.totalTokens)}</strong>
                  </div>
                </Section>
              </>
            )}
            {activePanel === "training" && (
              <>
                <Section title="Batch & schedule">
                  <Field
                    label="Micro batch / GPU"
                    value={config.training.microBatch}
                    onChange={(v) => update("training", { microBatch: v })}
                  />
                  <Field
                    label="Global batch"
                    value={config.training.globalBatch}
                    onChange={(v) => update("training", { globalBatch: v })}
                    unit="sequences"
                  />
                  <Field
                    label="Accumulation"
                    value={config.training.accumulation}
                    onChange={(v) => update("training", { accumulation: v })}
                    unit="steps"
                  />
                  <Field
                    label="Learning rate"
                    value={config.training.learningRate}
                    onChange={(v) => update("training", { learningRate: v })}
                    step={0.00001}
                  />
                  <SelectField
                    label="Scheduler"
                    value={config.training.scheduler}
                    options={["Cosine", "Linear", "Constant", "One cycle"]}
                    onChange={(v) => update("training", { scheduler: v })}
                  />
                  <Field
                    label="Warmup steps"
                    value={config.training.warmup}
                    onChange={(v) => update("training", { warmup: v })}
                  />
                </Section>
                <Section title="Optimizer">
                  <SelectField
                    label="Type"
                    value={config.training.optimizer}
                    options={[
                      "SGD",
                      "Adam",
                      "AdamW",
                      "Adafactor",
                      "8-bit Adam",
                    ]}
                    onChange={(v) =>
                      update("training", {
                        optimizer: v as Config["training"]["optimizer"],
                      })
                    }
                  />
                  <Field
                    label="Weight decay"
                    value={config.training.weightDecay}
                    onChange={(v) => update("training", { weightDecay: v })}
                    step={0.01}
                  />
                  <Field
                    label="Gradient clip"
                    value={config.training.gradClip}
                    onChange={(v) => update("training", { gradClip: v })}
                    step={0.1}
                  />
                  <Field
                    label="Checkpoint every"
                    value={config.training.checkpointInterval}
                    onChange={(v) =>
                      update("training", { checkpointInterval: v })
                    }
                    unit="steps"
                  />
                </Section>
                <div className="hint-card">
                  Global batch sets tokens per optimizer step. Micro batch
                  drives activation memory per GPU.
                </div>
              </>
            )}
            {activePanel === "hardware" && (
              <>
                <Section title="Accelerator">
                  <SelectField
                    label="GPU preset"
                    value={config.hardware.gpu}
                    options={Object.keys(gpuPresets)}
                    onChange={(v) =>
                      update("hardware", { gpu: v, ...gpuPresets[v] })
                    }
                  />
                  <Field
                    label="VRAM"
                    value={config.hardware.vram}
                    onChange={(v) => update("hardware", { vram: v })}
                    unit="GB"
                  />
                  <Field
                    label="HBM bandwidth"
                    value={config.hardware.bandwidth}
                    onChange={(v) => update("hardware", { bandwidth: v })}
                    unit="GB/s"
                  />
                  <Field
                    label="BF16 compute"
                    value={config.hardware.compute}
                    onChange={(v) => update("hardware", { compute: v })}
                    unit="TFLOPS"
                  />
                  <Field
                    label="FP8 compute"
                    value={config.hardware.fp8Compute}
                    onChange={(v) => update("hardware", { fp8Compute: v })}
                    unit="TFLOPS"
                  />
                  <Field
                    label="Power"
                    value={config.hardware.power}
                    onChange={(v) => update("hardware", { power: v })}
                    unit="W"
                  />
                </Section>
                <Section title="Cluster & pricing">
                  <Field
                    label="GPUs / node"
                    value={config.hardware.gpusPerNode}
                    onChange={(v) => update("hardware", { gpusPerNode: v })}
                  />
                  <Field
                    label="Nodes"
                    value={config.hardware.nodes}
                    onChange={(v) => update("hardware", { nodes: v })}
                  />
                  <SelectField
                    label="Connection"
                    value={config.hardware.link}
                    options={[
                      "NVLink + InfiniBand",
                      "NVSwitch + InfiniBand",
                      "PCIe + Ethernet",
                      "NVLink",
                      "InfiniBand",
                      "Ethernet",
                    ]}
                    onChange={(v) => update("hardware", { link: v })}
                  />
                  <Field
                    label="Interconnect"
                    value={config.hardware.interconnect}
                    onChange={(v) => update("hardware", { interconnect: v })}
                    unit="GB/s"
                  />
                  <Field
                    label="Network"
                    value={config.hardware.network}
                    onChange={(v) => update("hardware", { network: v })}
                    unit="Gb/s"
                  />
                  <Field
                    label="GPU cost / hour"
                    value={config.hardware.hourlyCost}
                    onChange={(v) => update("hardware", { hourlyCost: v })}
                    unit="$"
                    step={0.01}
                  />
                  <Field
                    label="Network / hour"
                    value={config.hardware.networkCost}
                    onChange={(v) => update("hardware", { networkCost: v })}
                    unit="$"
                    step={0.01}
                  />
                  <Field
                    label="Storage / hour"
                    value={config.hardware.storageCost}
                    onChange={(v) => update("hardware", { storageCost: v })}
                    unit="$"
                    step={0.01}
                  />
                  <Field
                    label="CPU / hour"
                    value={config.hardware.cpuCost}
                    onChange={(v) => update("hardware", { cpuCost: v })}
                    unit="$"
                    step={0.01}
                  />
                </Section>
              </>
            )}
            {activePanel === "distributed" && (
              <>
                <div className="hint-card">
                  Topology and communication reduce scaling efficiency. Tune the
                  strategy to fit the model across {animatedMetrics.gpus} GPUs.
                </div>
                <Section title="Parallelism strategy">
                  <Field
                    label="Data parallel"
                    value={config.distributed.dp}
                    onChange={(v) => update("distributed", { dp: v })}
                  />
                  <Field
                    label="Tensor parallel"
                    value={config.distributed.tp}
                    onChange={(v) => update("distributed", { tp: v })}
                  />
                  <Field
                    label="Pipeline parallel"
                    value={config.distributed.pp}
                    onChange={(v) => update("distributed", { pp: v })}
                  />
                  <Field
                    label="Sequence parallel"
                    value={config.distributed.sp}
                    onChange={(v) => update("distributed", { sp: v })}
                  />
                  <Field
                    label="Expert parallel"
                    value={config.distributed.ep}
                    onChange={(v) => update("distributed", { ep: v })}
                  />
                </Section>
                <Section title="Efficiency">
                  <div className="readout">
                    <span>Required devices</span>
                    <strong>
                      {config.distributed.dp *
                        config.distributed.tp *
                        config.distributed.pp *
                        config.distributed.ep}
                    </strong>
                  </div>
                  <div className="readout">
                    <span>Available devices</span>
                    <strong>{animatedMetrics.gpus}</strong>
                  </div>
                  <div className="readout">
                    <span>Scaling efficiency</span>
                    <strong>
                      {Math.round(animatedMetrics.efficiency * 100)}%
                    </strong>
                  </div>
                  <div className="readout">
                    <span>Communication / step</span>
                    <strong>
                      {animatedMetrics.communicationSeconds.toFixed(2)} s
                    </strong>
                  </div>
                </Section>
                {config.distributed.dp *
                  config.distributed.tp *
                  config.distributed.pp *
                  config.distributed.ep !==
                  animatedMetrics.gpus && (
                  <div className="health-card danger">
                    <div>
                      <ShieldAlert size={17} />
                      <strong>Device mapping mismatch</strong>
                    </div>
                    <p>
                      DP × TP × PP × EP should equal {animatedMetrics.gpus}{" "}
                      available GPUs.
                    </p>
                  </div>
                )}
              </>
            )}
          </div>
          <div className="inspector-footer">
            <div>
              <span>EST. TRAINING TIME</span>
              <strong>{duration(animatedMetrics.hours)}</strong>
            </div>
            <div>
              <span>BOTTLENECK</span>
              <strong className={animatedMetrics.oom ? "error" : ""}>
                {animatedMetrics.bottleneck}
              </strong>
            </div>
          </div>
        </aside>
      </div>
      <div className={"bottom-drawer " + (bottomOpen ? "open" : "")}>
        <div className="bottom-heading">
          <div className="bottom-tabs">
            {(
              ["timeline", "memory", "parameters", "compare", "logs"] as const
            ).map((t) => (
              <button
                key={t}
                className={bottomTab === t ? "active" : ""}
                onClick={() => {
                  setBottomTab(t);
                  setBottomOpen(true);
                }}
              >
                {t}
              </button>
            ))}
          </div>
          <button
            className="drawer-toggle"
            onClick={() => setBottomOpen(!bottomOpen)}
          >
            {bottomOpen ? <ChevronDown size={16} /> : <ChevronUp size={16} />}{" "}
            {bottomOpen ? "Hide profiler" : "Show profiler"}
          </button>
        </div>
        {bottomOpen && (
          <div className="drawer-content">
            {bottomTab === "timeline" && (
              <>
                <div className="timeline-info">
                  <strong>Iteration profile</strong>
                  <span>
                    {animatedMetrics.stepSeconds.toFixed(2)} s / step ·{" "}
                    {format(animatedMetrics.tokensPerSecond)} tokens/s ·{" "}
                    {Math.round(animatedMetrics.utilization * 100)}% GPU
                    utilization
                  </span>
                </div>
                {Array.from(
                  { length: Math.min(4, animatedMetrics.gpus) },
                  (_, i) => (
                    <div className="timeline-row" key={i}>
                      <span>GPU {i}</span>
                      <div className="timeline-track">
                        <i
                          className="forward-segment"
                          style={{
                            width: `${(animatedMetrics.forwardSeconds / animatedMetrics.stepSeconds) * 100}%`,
                          }}
                        />
                        <i
                          className="backward-segment"
                          style={{
                            width: `${(animatedMetrics.backwardSeconds / animatedMetrics.stepSeconds) * 100}%`,
                          }}
                        />
                        <i
                          className="comm-segment"
                          style={{
                            width: `${(animatedMetrics.communicationSeconds / animatedMetrics.stepSeconds) * 100}%`,
                          }}
                        />
                        <i
                          className="opt-segment"
                          style={{
                            width: `${(animatedMetrics.optimizerSeconds / animatedMetrics.stepSeconds) * 100}%`,
                          }}
                        />
                      </div>
                    </div>
                  ),
                )}
                <div className="timeline-legend">
                  <span>
                    <i className="forward-segment" />
                    Forward
                  </span>
                  <span>
                    <i className="backward-segment" />
                    Backward
                  </span>
                  <span>
                    <i className="comm-segment" />
                    All-reduce
                  </span>
                  <span>
                    <i className="opt-segment" />
                    Optimizer
                  </span>
                </div>
              </>
            )}
            {bottomTab === "memory" && (
              <div className="drawer-grid">
                <div>
                  <small>PARAMETERS</small>
                  <strong>
                    {animatedMetrics.memory.parameters.toFixed(1)} GB
                  </strong>
                </div>
                <div>
                  <small>GRADIENTS</small>
                  <strong>
                    {animatedMetrics.memory.gradients.toFixed(1)} GB
                  </strong>
                </div>
                <div>
                  <small>OPTIMIZER</small>
                  <strong>
                    {animatedMetrics.memory.optimizer.toFixed(1)} GB
                  </strong>
                </div>
                <div>
                  <small>ACTIVATIONS</small>
                  <strong>
                    {animatedMetrics.memory.activations.toFixed(1)} GB
                  </strong>
                </div>
                <div>
                  <small>CHECKPOINT</small>
                  <strong>{animatedMetrics.checkpointGB.toFixed(1)} GB</strong>
                </div>
              </div>
            )}
            {bottomTab === "parameters" && (
              <div className="drawer-grid">
                <div>
                  <small>EMBEDDINGS</small>
                  <strong>
                    {format(animatedMetrics.breakdown.embeddings)}
                  </strong>
                </div>
                <div>
                  <small>ATTENTION</small>
                  <strong>{format(animatedMetrics.breakdown.attention)}</strong>
                </div>
                <div>
                  <small>FFN</small>
                  <strong>{format(animatedMetrics.breakdown.ffn)}</strong>
                </div>
                <div>
                  <small>NORM</small>
                  <strong>{format(animatedMetrics.breakdown.norm)}</strong>
                </div>
                <div>
                  <small>TOTAL</small>
                  <strong>{format(animatedMetrics.parameters)}</strong>
                </div>
              </div>
            )}
            {bottomTab === "compare" && (
              <div className="compare-panel">
                {baseline ? (
                  <>
                    <div className="compare-head">
                      <strong>Saved baseline: {baseline.name}</strong>
                      <span>Current setup updates as you edit</span>
                    </div>
                    {[
                      [
                        "Parameters",
                        baseline.parameters,
                        animatedMetrics.parameters,
                        format,
                      ],
                      [
                        "VRAM / GPU",
                        baseline.memory,
                        animatedMetrics.memory.total,
                        (v: number) => v.toFixed(1) + " GB",
                      ],
                      [
                        "Tokens / second",
                        baseline.throughput,
                        animatedMetrics.tokensPerSecond,
                        format,
                      ],
                      [
                        "Training time",
                        baseline.hours,
                        animatedMetrics.hours,
                        duration,
                      ],
                      [
                        "Total cost",
                        baseline.cost,
                        animatedMetrics.cost,
                        money,
                      ],
                    ].map(([label, before, after, formatter]) => (
                      <div className="compare-row" key={String(label)}>
                        <span>{String(label)}</span>
                        <b>
                          {(formatter as (v: number) => string)(
                            before as number,
                          )}
                        </b>
                        <b>
                          {(formatter as (v: number) => string)(
                            after as number,
                          )}
                        </b>
                        <em
                          className={
                            (after as number) < (before as number)
                              ? "decrease"
                              : "increase"
                          }
                        >
                          {(
                            ((after as number) / (before as number) - 1) *
                            100
                          ).toFixed(0)}
                          %
                        </em>
                      </div>
                    ))}
                  </>
                ) : (
                  <div className="compare-empty">
                    Save a baseline, then change the model or GPUs to compare
                    cost and performance.
                  </div>
                )}
              </div>
            )}
            {bottomTab === "logs" && (
              <div className="logs">
                <p>
                  <span>00:00</span> Configuration loaded with seed{" "}
                  {config.seed}
                </p>
                <p>
                  <span>00:01</span> {animatedMetrics.gpus} ×{" "}
                  {config.hardware.gpu} allocated
                </p>
                <p>
                  <span>00:02</span>{" "}
                  {animatedMetrics.oom
                    ? `OOM: ${Math.abs(animatedMetrics.memory.free).toFixed(1)} GB over VRAM limit`
                    : `Memory fit: ${animatedMetrics.memory.free.toFixed(1)} GB headroom per GPU`}
                </p>
                <p>
                  <span>00:03</span> Estimated bottleneck:{" "}
                  {animatedMetrics.bottleneck}
                </p>
              </div>
            )}
          </div>
        )}
      </div>
      <div className="utility-bar">
        <div className="utility-left">
          <span className="ready-pill">
            <span />{" "}
            {animatedMetrics.oom ? "OOM predicted" : "Simulation ready"}
          </span>
          <span className="utility-divider" />
          <span>{animatedMetrics.gpus} GPUs</span>
          <span>·</span>
          <span>{Math.round(animatedMetrics.efficiency * 100)}% scaling</span>
          <span>·</span>
          <span>{format(animatedMetrics.energyKWh)} kWh</span>
        </div>
        <div className="utility-right">
          <select
            value={speed}
            onChange={(e) => setSpeed(e.target.value)}
            aria-label="Simulation speed"
          >
            {["1×", "10×", "100×", "1000×", "Instant"].map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
          <button
            onClick={() => {
              setBaseline({
                name: config.model.name,
                parameters: metrics.parameters,
                memory: metrics.memory.total,
                throughput: metrics.tokensPerSecond,
                hours: metrics.hours,
                cost: metrics.cost,
              });
              setBottomTab("compare");
              setBottomOpen(true);
              flash("Baseline captured");
            }}
          >
            <Braces size={14} /> Compare
          </button>
          <button onClick={save}>
            <Save size={14} /> Save
          </button>
          <button onClick={() => fileRef.current?.click()}>
            <Upload size={14} /> Import
          </button>
          <button onClick={exportProject}>
            <FileJson size={14} /> JSON
          </button>
          <button onClick={report}>
            <Download size={14} /> Report
          </button>
          <button onClick={exportImage}>
            <ArrowDownToLine size={14} /> PNG
          </button>
          <button onClick={share}>
            <Share2 size={14} /> Share
          </button>
        </div>
      </div>
      <input
        ref={fileRef}
        type="file"
        accept="application/json,.json"
        hidden
        onChange={importProject}
      />
      {notice && <div className="toast">{notice}</div>}
    </div>
  );
}
export default function Studio() {
  return (
    <ReactFlowProvider>
      <StudioInner />
    </ReactFlowProvider>
  );
}

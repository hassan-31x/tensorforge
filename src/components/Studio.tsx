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
  NodeResizer,
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
import { resolveNetwork } from "@/sim/topology";
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
  | "group"
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
    hardware?: Config["hardware"];
  },
  "studio" | "group"
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
      {
        kind: "group",
        title: "Model group",
        desc: "Frame the full architecture",
      },
      { kind: "gpu", title: "GPU node", desc: "Accelerator memory" },
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
  group: TbBinaryTree,
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
function GroupCard({ data, selected }: NodeProps<StudioNode>) {
  return (
    <div className={`model-group-card ${selected ? "is-selected" : ""}`}>
      <NodeResizer isVisible={selected} minWidth={300} minHeight={180} />
      <div className="model-group-title">
        <KindIcon kind="group" size={17} />
        <strong>{data.title}</strong>
        <span>{data.subtitle}</span>
      </div>
      <Handle
        type="target"
        position={Position.Bottom}
        className="node-handle group-handle"
      />
    </div>
  );
}
const nodeTypes = { studio: NodeCard, group: GroupCard };

function makeGpuNode(
  id: string,
  position: { x: number; y: number },
  hardware: Config["hardware"],
  kind: "gpu" | "cluster" = "gpu",
): StudioNode {
  return {
    id,
    type: "studio",
    position,
    data: {
      kind,
      title: hardware.gpu,
      subtitle: `${hardware.vram} GB VRAM · connect to model group`,
      hardware: { ...hardware },
    },
  };
}

function makeGroupNode(
  id: string,
  nodes: StudioNode[],
  position?: { x: number; y: number },
): StudioNode {
  const modelNodes = nodes.filter(
    (node) => !["gpu", "cluster", "group"].includes(node.data.kind),
  );
  const left = Math.min(...modelNodes.map((node) => node.position.x), 0) - 40;
  const top = Math.min(...modelNodes.map((node) => node.position.y), 0) - 75;
  const right =
    Math.max(
      ...modelNodes.map((node) => node.position.x + (node.width || 218)),
      600,
    ) + 45;
  const bottom =
    Math.max(
      ...modelNodes.map((node) => node.position.y + (node.height || 150)),
      300,
    ) + 45;
  return {
    id,
    type: "group",
    position: position || { x: left, y: top },
    width: position ? 650 : right - left,
    height: position ? 340 : bottom - top,
    zIndex: -1,
    data: {
      kind: "group",
      title: "Model group",
      subtitle: "Connect GPUs to this frame",
    },
  };
}

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
  const [drawingGroup, setDrawingGroup] = useState(false);
  const [drawOrigin, setDrawOrigin] = useState<{ x: number; y: number } | null>(
    null,
  );
  const [drawRect, setDrawRect] = useState<{
    x: number;
    y: number;
    width: number;
    height: number;
  } | null>(null);
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
    const count = (kind: Kind) =>
      nodes.filter((n) => n.data.kind === kind).length;
    return {
      transformer: count("transformer"),
      attention: count("attention"),
      ffn: count("ffn"),
      embedding: count("embedding"),
      head: count("head"),
    };
  }, [nodes]);
  const network = useMemo(
    () => resolveNetwork(nodes, edges, config.hardware),
    [nodes, edges, config.hardware],
  );
  const gpuDevices = network.devices;
  const framedGroupIds = network.framedGroupIds;
  const metrics = useMemo(
    () => simulate(config, architecture, gpuDevices),
    [config, architecture, gpuDevices],
  );
  const live = useCallback(
    (value: number, amplitude = 1, phase = 0) =>
      value + Math.sin(liveTick * 0.78 + phase) * amplitude,
    [liveTick],
  );
  const animatedMetrics = useMemo(() => {
    const memory = {
      ...metrics.memory,
      parameters: metrics.gpus
        ? Math.max(0, live(metrics.memory.parameters, 0.16, 0.4))
        : 0,
      gradients: metrics.gpus
        ? Math.max(0, live(metrics.memory.gradients, 0.14, 1.2))
        : 0,
      optimizer: metrics.gpus
        ? Math.max(0, live(metrics.memory.optimizer, 0.12, 2.1))
        : 0,
      activations: metrics.gpus
        ? Math.max(0, live(metrics.memory.activations, 0.2, 2.8))
        : 0,
      temporary: metrics.gpus
        ? Math.max(0, live(metrics.memory.temporary, 0.08, 3.4))
        : 0,
      runtime: metrics.gpus
        ? Math.max(0, live(metrics.memory.runtime, 0.04, 4.1))
        : 0,
    };
    memory.total =
      memory.parameters +
      memory.gradients +
      memory.optimizer +
      memory.activations +
      memory.temporary +
      memory.runtime;
    memory.free = metrics.gpuDevices.length
      ? Math.min(
          ...metrics.gpuDevices.map((device) => device.vram - memory.total),
        )
      : 0;
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
      utilization:
        metrics.oom || !metrics.gpus
          ? 0
          : Math.max(0, Math.min(1, live(metrics.utilization, 0.008, 1.3))),
      mfu:
        metrics.oom || !metrics.gpus
          ? 0
          : Math.max(0, Math.min(1, live(metrics.mfu, 0.006, 2.4))),
      bandwidthUtil:
        metrics.oom || !metrics.gpus
          ? 0
          : Math.max(0, Math.min(1, live(metrics.bandwidthUtil, 0.01, 3.1))),
      tokensPerSecond:
        metrics.gpus && !metrics.oom
          ? Math.max(
              0,
              live(
                metrics.tokensPerSecond,
                Math.max(1, metrics.tokensPerSecond * 0.012),
                1.7,
              ),
            )
          : 0,
      cost:
        metrics.gpus && !metrics.oom
          ? Math.max(
              0,
              live(metrics.cost, Math.max(0.15, metrics.cost * 0.004), 2.9),
            )
          : 0,
      hourlyCost: metrics.gpus
        ? Math.max(0, live(metrics.hourlyCost, 0.05, 2.9))
        : 0,
      energyKWh:
        metrics.gpus && !metrics.oom
          ? Math.max(
              0,
              live(
                metrics.energyKWh,
                Math.max(0.05, metrics.energyKWh * 0.005),
                3.6,
              ),
            )
          : 0,
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
  }, [live, metrics]);
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
    if (!drawingGroup) return;
    const cancel = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setDrawingGroup(false);
        setDrawOrigin(null);
        setDrawRect(null);
      }
    };
    window.addEventListener("keydown", cancel);
    return () => window.removeEventListener("keydown", cancel);
  }, [drawingGroup]);
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
    if ((!metrics.gpus || metrics.oom) && playing) setPlaying(false);
  }, [metrics.gpus, metrics.oom, playing]);
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
          const hardware = n.data.hardware || config.hardware;
          const active = metrics.gpuDevices.find(
            (device) => device.id === n.id,
          );
          subtitle = `${hardware.gpu} · ${hardware.vram} GB VRAM`;
          metric = active
            ? `${Math.round((animatedMetrics.memory.total / hardware.vram) * 100)}%`
            : "0%";
          detail = active
            ? `${animatedMetrics.memory.total.toFixed(1)} / ${hardware.vram} GB`
            : "Not connected";
        }
        if (kind === "group") {
          metric = framedGroupIds.includes(n.id)
            ? `${gpuDevices.length} GPU${gpuDevices.length === 1 ? "" : "s"} connected`
            : "Resize to frame architecture";
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
              (kind === "gpu" || kind === "cluster") &&
              !!metrics.gpuDevices.find((device) => device.id === n.id)?.oom,
          },
        };
      }),
    [
      nodes,
      config,
      animatedMetrics,
      metrics.gpuDevices,
      framedGroupIds,
      gpuDevices.length,
      tensorMode,
      playing,
      progress,
    ],
  );
  const onConnect = useCallback(
    (connection: Connection) => {
      const sourceKind = nodes.find((node) => node.id === connection.source)
        ?.data.kind;
      const targetKind = nodes.find((node) => node.id === connection.target)
        ?.data.kind;
      const gpuLink =
        (sourceKind === "gpu" || sourceKind === "cluster") &&
        targetKind === "group";
      const modelLink =
        sourceKind &&
        targetKind &&
        !["gpu", "cluster", "group"].includes(sourceKind) &&
        !["gpu", "cluster", "group"].includes(targetKind);
      if (!gpuLink && !modelLink) return;
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
      );
    },
    [nodes, setEdges],
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
        kind === "group"
          ? makeGroupNode(id, ns, position)
          : kind === "gpu" || kind === "cluster"
            ? makeGpuNode(id, position, config.hardware, kind)
            : {
                id,
                type: "studio",
                position,
                data: { kind, title, subtitle: "Configure in inspector" },
              },
      ]);
      setSelected(id);
      setActivePanel("inspect");
    },
    [config.hardware, rf, setNodes],
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
  const selectedGpuNode = nodes.find(
    (node) =>
      node.id === selected &&
      (node.data.kind === "gpu" || node.data.kind === "cluster"),
  );
  const hardwareView = selectedGpuNode?.data.hardware || config.hardware;
  const updateHardware = (patch: Partial<Config["hardware"]>) => {
    if (!selectedGpuNode) {
      update("hardware", patch);
      return;
    }
    if (
      "network" in patch ||
      "networkCost" in patch ||
      "storageCost" in patch ||
      "cpuCost" in patch ||
      "link" in patch
    ) {
      update("hardware", patch);
    }
    setNodes((current) =>
      current.map((node) =>
        node.id === selectedGpuNode.id
          ? {
              ...node,
              data: {
                ...node.data,
                title: patch.gpu || node.data.title,
                hardware: {
                  ...(node.data.hardware || config.hardware),
                  ...patch,
                },
              },
            }
          : node,
      ),
    );
  };
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
            <small>NETWORK VRAM</small>
            <strong className={animatedMetrics.oom ? "error" : ""}>
              <span className="telemetry-value">
                {metrics.capacity
                  ? Math.round(
                      ((animatedMetrics.memory.total * metrics.gpus) /
                        metrics.capacity) *
                        100,
                    )
                  : 0}
                %
              </span>{" "}
              <span>
                {(animatedMetrics.memory.total * metrics.gpus).toFixed(1)} /{" "}
                {metrics.capacity} GB
              </span>
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
            <small>RUN COST / HOUR</small>
            <strong className="telemetry-value">
              ${animatedMetrics.hourlyCost.toFixed(2)}
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
            disabled={!metrics.gpus || metrics.oom}
            title={
              !metrics.gpus
                ? "Connect a GPU to a model group first"
                : metrics.oom
                  ? "Model exceeds connected GPU memory"
                  : undefined
            }
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
                        if (i.kind === "group") {
                          setDrawingGroup(true);
                          setDrawOrigin(null);
                          setDrawRect(null);
                          return;
                        }
                        const id = `${i.kind}-${Date.now()}`;
                        setNodes((ns) => {
                          const group = ns.find(
                            (node) => node.data.kind === "group",
                          );
                          const gpuCount = ns.filter(
                            (node) =>
                              node.data.kind === "gpu" ||
                              node.data.kind === "cluster",
                          ).length;
                          const position =
                            group && (i.kind === "gpu" || i.kind === "cluster")
                              ? {
                                  x: group.position.x + 60 + gpuCount * 255,
                                  y:
                                    group.position.y +
                                    (group.measured?.height ||
                                      group.height ||
                                      340) +
                                    90,
                                }
                              : rf.screenToFlowPosition({
                                  x: window.innerWidth / 2,
                                  y: window.innerHeight / 2,
                                });
                          const next =
                            i.kind === "group"
                              ? makeGroupNode(id, ns)
                              : i.kind === "gpu" || i.kind === "cluster"
                                ? makeGpuNode(
                                    id,
                                    position,
                                    config.hardware,
                                    i.kind,
                                  )
                                : {
                                    id,
                                    type: "studio" as const,
                                    position,
                                    data: {
                                      kind: i.kind,
                                      title: i.title,
                                      subtitle: i.desc,
                                    },
                                  };
                          return [...ns, next];
                        });
                        setSelected(id);
                        window.setTimeout(
                          () => rf.fitView({ padding: 0.16, duration: 250 }),
                          40,
                        );
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
            <span>Click group, then draw around the model</span>
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
            elevateNodesOnSelect={false}
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
          {drawingGroup && (
            <div
              className="group-draw-layer"
              onPointerDown={(event) => {
                const bounds = event.currentTarget.getBoundingClientRect();
                const origin = {
                  x: event.clientX - bounds.left,
                  y: event.clientY - bounds.top,
                };
                event.currentTarget.setPointerCapture(event.pointerId);
                setDrawOrigin(origin);
                setDrawRect({ ...origin, width: 0, height: 0 });
              }}
              onPointerMove={(event) => {
                if (!drawOrigin) return;
                const bounds = event.currentTarget.getBoundingClientRect();
                const x = event.clientX - bounds.left;
                const y = event.clientY - bounds.top;
                setDrawRect({
                  x: Math.min(drawOrigin.x, x),
                  y: Math.min(drawOrigin.y, y),
                  width: Math.abs(x - drawOrigin.x),
                  height: Math.abs(y - drawOrigin.y),
                });
              }}
              onPointerUp={(event) => {
                if (!drawOrigin) return;
                const bounds = event.currentTarget.getBoundingClientRect();
                const end = {
                  x: event.clientX - bounds.left,
                  y: event.clientY - bounds.top,
                };
                const x = Math.min(drawOrigin.x, end.x);
                const y = Math.min(drawOrigin.y, end.y);
                const startFlow = rf.screenToFlowPosition({
                  x: bounds.left + x,
                  y: bounds.top + y,
                });
                const endFlow = rf.screenToFlowPosition({
                  x: bounds.left + Math.max(drawOrigin.x, end.x),
                  y: bounds.top + Math.max(drawOrigin.y, end.y),
                });
                const id = `group-${Date.now()}`;
                setNodes((current) => [
                  ...current,
                  {
                    ...makeGroupNode(id, current, startFlow),
                    width: Math.max(300, endFlow.x - startFlow.x),
                    height: Math.max(180, endFlow.y - startFlow.y),
                  },
                ]);
                setSelected(id);
                setActivePanel("inspect");
                setDrawingGroup(false);
                setDrawOrigin(null);
                setDrawRect(null);
              }}
            >
              <div className="group-draw-hint">
                Drag to frame the architecture · Esc to cancel
              </div>
              {drawRect && (
                <div
                  className="group-draw-preview"
                  style={{
                    left: drawRect.x,
                    top: drawRect.y,
                    width: drawRect.width,
                    height: drawRect.height,
                  }}
                />
              )}
            </div>
          )}
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
                    {sk === "group" ? (
                      <div className="group-inspector">
                        <strong>
                          {framedGroupIds.includes(selected!)
                            ? "Architecture framed"
                            : "Frame the architecture"}
                        </strong>
                        <p>
                          Resize this group around every model component, then
                          connect GPU nodes to its bottom handle. Only connected
                          GPUs count toward estimates.
                        </p>
                        <div className="readout">
                          <span>Connected GPUs</span>
                          <strong>{gpuDevices.length}</strong>
                        </div>
                        <button
                          className="wide-link"
                          onClick={() => {
                            setNodes((current) => {
                              const fitted = makeGroupNode(selected!, current);
                              return current.map((node) =>
                                node.id === selected
                                  ? {
                                      ...node,
                                      position: fitted.position,
                                      width: fitted.width,
                                      height: fitted.height,
                                    }
                                  : node,
                              );
                            });
                          }}
                        >
                          Fit to architecture <Maximize2 size={14} />
                        </button>
                      </div>
                    ) : sk === "gpu" || sk === "cluster" ? (
                      <>
                        <Section title="Device">
                          <SelectField
                            label="GPU model"
                            value={hardwareView.gpu}
                            options={Object.keys(gpuPresets)}
                            onChange={(value) =>
                              updateHardware({
                                gpu: value,
                                ...gpuPresets[value],
                              })
                            }
                          />
                          <div className="readout">
                            <span>Connection</span>
                            <strong>
                              {metrics.gpuDevices.some(
                                (device) => device.id === selected,
                              )
                                ? "Active"
                                : "Not connected"}
                            </strong>
                          </div>
                          <div className="readout">
                            <span>Hourly rate</span>
                            <strong>
                              ${hardwareView.hourlyCost.toFixed(2)} / hr
                            </strong>
                          </div>
                        </Section>
                        <div
                          className={
                            "health-card " +
                            (metrics.gpuDevices.find(
                              (device) => device.id === selected,
                            )?.oom
                              ? "danger"
                              : "")
                          }
                        >
                          <div>
                            <MemoryStick size={17} />
                            <strong>
                              {!metrics.gpuDevices.some(
                                (device) => device.id === selected,
                              )
                                ? "Connect to model group"
                                : metrics.gpuDevices.find(
                                      (device) => device.id === selected,
                                    )?.oom
                                  ? "Out of memory"
                                  : "Memory healthy"}
                            </strong>
                          </div>
                          <p>
                            {metrics.gpuDevices.some(
                              (device) => device.id === selected,
                            )
                              ? `${animatedMetrics.memory.total.toFixed(1)} GB used of ${hardwareView.vram} GB. ${(hardwareView.vram - animatedMetrics.memory.total).toFixed(1)} GB free.`
                              : "Connect this GPU to a group that surrounds the full model."}
                          </p>
                        </div>
                        <Section title="GPU memory">
                          <Bar
                            label="Parameters"
                            value={
                              metrics.gpuDevices.some(
                                (device) => device.id === selected,
                              )
                                ? animatedMetrics.memory.parameters
                                : 0
                            }
                            total={hardwareView.vram}
                            color="#a9a3f2"
                          />
                          <Bar
                            label="Gradients"
                            value={
                              metrics.gpuDevices.some(
                                (device) => device.id === selected,
                              )
                                ? animatedMetrics.memory.gradients
                                : 0
                            }
                            total={hardwareView.vram}
                            color="#89bcf2"
                          />
                          <Bar
                            label="Optimizer"
                            value={
                              metrics.gpuDevices.some(
                                (device) => device.id === selected,
                              )
                                ? animatedMetrics.memory.optimizer
                                : 0
                            }
                            total={hardwareView.vram}
                            color="#e8b66a"
                          />
                          <Bar
                            label="Activations"
                            value={
                              metrics.gpuDevices.some(
                                (device) => device.id === selected,
                              )
                                ? animatedMetrics.memory.activations
                                : 0
                            }
                            total={hardwareView.vram}
                            color="#78ceb5"
                          />
                          <Bar
                            label="Temporary"
                            value={
                              metrics.gpuDevices.some(
                                (device) => device.id === selected,
                              )
                                ? animatedMetrics.memory.temporary +
                                  animatedMetrics.memory.runtime
                                : 0
                            }
                            total={hardwareView.vram}
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
                    value={hardwareView.gpu}
                    options={Object.keys(gpuPresets)}
                    onChange={(v) =>
                      updateHardware({ gpu: v, ...gpuPresets[v] })
                    }
                  />
                  <Field
                    label="VRAM"
                    value={hardwareView.vram}
                    onChange={(v) => updateHardware({ vram: v })}
                    unit="GB"
                  />
                  <Field
                    label="HBM bandwidth"
                    value={hardwareView.bandwidth}
                    onChange={(v) => updateHardware({ bandwidth: v })}
                    unit="GB/s"
                  />
                  <Field
                    label="BF16 compute"
                    value={hardwareView.compute}
                    onChange={(v) => updateHardware({ compute: v })}
                    unit="TFLOPS"
                  />
                  <Field
                    label="FP8 compute"
                    value={hardwareView.fp8Compute}
                    onChange={(v) => updateHardware({ fp8Compute: v })}
                    unit="TFLOPS"
                  />
                  <Field
                    label="Power"
                    value={hardwareView.power}
                    onChange={(v) => updateHardware({ power: v })}
                    unit="W"
                  />
                </Section>
                <Section title="Network & pricing">
                  <p className="section-note">
                    GPU count comes from devices connected to the model group.
                  </p>
                  <SelectField
                    label="Connection"
                    value={hardwareView.link}
                    options={[
                      "NVLink + InfiniBand",
                      "NVSwitch + InfiniBand",
                      "PCIe + Ethernet",
                      "NVLink",
                      "InfiniBand",
                      "Ethernet",
                    ]}
                    onChange={(v) => updateHardware({ link: v })}
                  />
                  <Field
                    label="Interconnect"
                    value={hardwareView.interconnect}
                    onChange={(v) => updateHardware({ interconnect: v })}
                    unit="GB/s"
                  />
                  <Field
                    label="Network"
                    value={config.hardware.network}
                    onChange={(v) => updateHardware({ network: v })}
                    unit="Gb/s"
                  />
                  <Field
                    label="GPU cost / hour"
                    value={hardwareView.hourlyCost}
                    onChange={(v) => updateHardware({ hourlyCost: v })}
                    unit="$"
                    step={0.01}
                  />
                  <Field
                    label="Network / hour"
                    value={config.hardware.networkCost}
                    onChange={(v) => updateHardware({ networkCost: v })}
                    unit="$"
                    step={0.01}
                  />
                  <Field
                    label="Storage / hour"
                    value={config.hardware.storageCost}
                    onChange={(v) => updateHardware({ storageCost: v })}
                    unit="$"
                    step={0.01}
                  />
                  <Field
                    label="CPU / hour"
                    value={config.hardware.cpuCost}
                    onChange={(v) => updateHardware({ cpuCost: v })}
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
                {metrics.gpus > 0 &&
                  config.distributed.dp *
                    config.distributed.tp *
                    config.distributed.pp *
                    config.distributed.ep !==
                    animatedMetrics.gpus && (
                    <div className="hint-card">
                      The estimate uses {animatedMetrics.gpus}-way data parallel
                      until DP × TP × PP × EP matches the connected GPU count.
                    </div>
                  )}
              </>
            )}
          </div>
          <div className="inspector-footer">
            <div>
              <span>EST. TRAINING TIME</span>
              <strong>
                {metrics.gpus && !metrics.oom
                  ? duration(animatedMetrics.hours)
                  : "—"}
              </strong>
            </div>
            <div>
              <span>BOTTLENECK</span>
              <strong className={animatedMetrics.oom ? "error" : ""}>
                {animatedMetrics.bottleneck}
              </strong>
            </div>
            <div className="footer-cost">
              <span>EST. TOTAL COST</span>
              <strong>
                {metrics.gpus && !metrics.oom
                  ? money(animatedMetrics.cost)
                  : "—"}
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
            {bottomTab === "timeline" && (!metrics.gpus || metrics.oom) && (
              <div className="compare-empty">
                {metrics.oom
                  ? "The model exceeds a connected GPU's VRAM. Add capacity to see the iteration profile."
                  : "Connect a GPU to a framed model to see the iteration profile."}
              </div>
            )}
            {bottomTab === "timeline" && metrics.gpus > 0 && !metrics.oom && (
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
                  <span>00:01</span>{" "}
                  {animatedMetrics.gpus
                    ? `${animatedMetrics.gpus} connected GPU${animatedMetrics.gpus === 1 ? "" : "s"} allocated`
                    : "No GPU connected to a framed model"}
                </p>
                <p>
                  <span>00:02</span>{" "}
                  {!animatedMetrics.gpus
                    ? "Connect a GPU node to the model group to estimate memory and cost"
                    : animatedMetrics.oom
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
            {!animatedMetrics.gpus
              ? "Connect GPU to model group"
              : animatedMetrics.oom
                ? "OOM predicted"
                : "Simulation ready"}
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

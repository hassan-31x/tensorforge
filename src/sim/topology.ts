import type { Config, GpuDevice } from "./engine";

export type TopologyNode = {
  id: string;
  position: { x: number; y: number };
  width?: number;
  height?: number;
  measured?: { width?: number; height?: number };
  data: { kind: string; hardware?: Config["hardware"] };
};

export type TopologyEdge = { source: string; target: string };

export function containsNode(group: TopologyNode, node: TopologyNode): boolean {
  const width = group.measured?.width || group.width || 650;
  const height = group.measured?.height || group.height || 340;
  const centerX =
    node.position.x + (node.measured?.width || node.width || 218) / 2;
  const centerY =
    node.position.y + (node.measured?.height || node.height || 140) / 2;
  return (
    centerX >= group.position.x &&
    centerX <= group.position.x + width &&
    centerY >= group.position.y &&
    centerY <= group.position.y + height
  );
}

export function resolveNetwork(
  nodes: TopologyNode[],
  edges: TopologyEdge[],
  fallback: Config["hardware"],
): { framedGroupIds: string[]; devices: GpuDevice[] } {
  const modelNodes = nodes.filter(
    (node) => !["gpu", "cluster", "group"].includes(node.data.kind),
  );
  const framedGroupIds = nodes
    .filter(
      (group) =>
        group.data.kind === "group" &&
        modelNodes.some((node) => node.data.kind === "transformer") &&
        modelNodes.every((node) => containsNode(group, node)),
    )
    .map((group) => group.id);
  const framedIds = new Set(framedGroupIds);
  const devices = nodes
    .filter(
      (node) =>
        (node.data.kind === "gpu" || node.data.kind === "cluster") &&
        edges.some(
          (edge) => edge.source === node.id && framedIds.has(edge.target),
        ),
    )
    .map((node) => {
      const hardware = node.data.hardware || fallback;
      return {
        id: node.id,
        gpu: hardware.gpu,
        vram: hardware.vram,
        bandwidth: hardware.bandwidth,
        compute: hardware.compute,
        fp8Compute: hardware.fp8Compute,
        interconnect: hardware.interconnect,
        power: hardware.power,
        hourlyCost: hardware.hourlyCost,
      };
    });
  return { framedGroupIds, devices };
}

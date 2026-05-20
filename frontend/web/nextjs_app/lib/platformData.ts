export const platformSummary = {
  benchmarkPassRate: 0.94,
  p95LatencyMs: 1280,
  throughputRps: 42,
  safetyFindingsOpen: 3,
  activeModels: 4,
  deploymentStage: "staging-ready",
};

export const benchmarkSeries = [
  { label: "Hallucination", value: 0.91 },
  { label: "Latency", value: 0.88 },
  { label: "Throughput", value: 0.86 },
  { label: "Robustness", value: 0.9 },
  { label: "Bias", value: 0.84 },
  { label: "Adversarial", value: 0.82 },
  { label: "Reproducibility", value: 0.99 },
];

export const deploymentChecks = [
  { name: "API health", status: "passing", detail: "/health and /ready endpoints defined" },
  { name: "Metrics", status: "passing", detail: "Prometheus text endpoint exposed at /metrics" },
  { name: "Container", status: "passing", detail: "Dockerfile and Compose stack present" },
  { name: "Kubernetes", status: "ready", detail: "Deployment, service, probes, and HPA configured" },
  { name: "Cloud deploy", status: "needs-secrets", detail: "Vercel/Render/Railway manifests prepared" },
];

export const traces = [
  {
    id: "trace-align-001",
    route: "vllm-a10g-primary",
    reward: 0.87,
    latency: 1180,
    events: ["request.received", "safety.assessed", "retrieval.completed", "routing.selected", "evaluation.completed"],
  },
  {
    id: "trace-align-002",
    route: "hosted-alignment-api",
    reward: 0.81,
    latency: 1540,
    events: ["request.received", "safety.assessed", "routing.selected", "evaluation.completed"],
  },
];

export const modelBackends = [
  { name: "vllm-a10g-primary", health: "healthy", quantization: "fp16", latency: 1180, memory: "17GB free" },
  { name: "hosted-alignment-api", health: "healthy", quantization: "bf16", latency: 1540, memory: "managed" },
  { name: "cpu-safety-fallback", health: "degraded", quantization: "cpu", latency: 3600, memory: "n/a" },
  { name: "mock-local-dev", health: "healthy", quantization: "cpu", latency: 120, memory: "dev only" },
];

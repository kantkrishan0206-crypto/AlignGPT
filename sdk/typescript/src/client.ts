export interface AlignRequest {
  prompt: string;
  task?: string;
  metadata?: Record<string, unknown>;
}

export interface AlignResponse {
  request_id: string;
  output: string;
  model_backend: string;
  safety_findings: Array<Record<string, unknown>>;
  citations: string[];
  metadata: Record<string, unknown>;
  created_at: string;
}

export class AlignGPTClient {
  constructor(private readonly baseUrl = "http://localhost:8000") {}

  async align(payload: AlignRequest): Promise<AlignResponse> {
    const response = await fetch(`${this.baseUrl}/v1/align`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task: "chat", metadata: {}, ...payload })
    });
    if (!response.ok) {
      throw new Error(`AlignGPT request failed: ${response.status}`);
    }
    return response.json() as Promise<AlignResponse>;
  }
}

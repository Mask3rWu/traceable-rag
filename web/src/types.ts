export type RunStatus =
  | 'queued'
  | 'running'
  | 'cancel_requested'
  | 'cancelled'
  | 'completed'
  | 'failed'

export interface RouteDecision {
  mode: 'fast' | 'supervisor'
  reason: string
}

export interface EvidenceVisual {
  block_id: string
  block_type: string
  page: number
  relation: string
  image_crop: string | null
  description: string | null
  status: string
}

export interface RetrievalTrace {
  query: string
  final_rank: number
  dense_rank: number | null
  dense_score: number | null
  bm25_rank: number | null
  bm25_score: number | null
  fusion_score: number | null
}

export interface Evidence {
  evidence_id: string
  chunk_id: string
  document_id: string
  source_file: string
  page_start: number
  page_end: number
  section_path: string[]
  quote: string
  quote_truncated: boolean
  visual_assets: EvidenceVisual[]
  retrieval: RetrievalTrace[]
  verified: boolean
}

export interface Citation {
  evidence_id: string
  quote: string
}

export interface Claim {
  claim_id: string
  text: string
  conclusion_type: 'direct' | 'synthesized' | 'hypothesis'
  citations: Citation[]
  citation_verified: boolean
}

export interface Conflict {
  conflict_id: string
  description: string
  claim_ids: string[]
  evidence_ids: string[]
  status: 'open' | 'resolved'
  resolution: string | null
}

export interface ResearchPacket {
  task: string
  status: 'sufficient' | 'insufficient'
  summary: string
  claims: Claim[]
  conflicts: Conflict[]
  gaps: string[]
  evidence_ids: string[]
}

export interface AgentRun {
  run_id: string
  request: string
  route: RouteDecision
  answer: {
    content: string
    evidence_ids: string[]
    limitations: string[]
  }
  evidence: Evidence[]
  worker_packets: ResearchPacket[]
  trace_id: string | null
  created_at: string
}

export interface RunSummary {
  run_id: string
  request: string
  status: RunStatus
  route: 'fast' | 'supervisor' | null
  route_reason: string | null
  trace_id: string | null
  evidence_count: number
  worker_count: number
  created_at: string
  updated_at: string
  error: string | null
}

export interface RunDetail extends RunSummary {
  result: AgentRun | null
}

export interface RunEvent {
  sequence: number
  type: string
  created_at: string
  data: Record<string, unknown>
}

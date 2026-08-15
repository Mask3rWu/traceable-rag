export type RunStatus =
  | 'queued'
  | 'running'
  | 'cancel_requested'
  | 'cancelled'
  | 'completed'
  | 'incomplete'
  | 'failed'
  | 'routed_away'

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
}

export interface Citation {
  evidence_id: string
}

export interface Conflict {
  conflict_id: string
  description: string
  evidence_ids: string[]
  status: 'open' | 'resolved'
  resolution: string | null
}

export interface RuleRecord {
  basis: 'source' | 'designed' | 'synthesized'
  evidence_ids: string[]
  rationale: string | null
  contract_id: string | null
}

export interface ContractRecord {
  contract_id: string
  type: 'terms' | 'threshold' | 'classification'
  canonical_terms: string[]
  applies_to_chapters: string[]
}

export interface ChapterPlan {
  chapter_id: string
  ordinal: number
  title: string
  objective: string
  research_questions: string[]
  depends_on: string[]
  produces_contracts: string[]
  required_contracts: string[]
  acceptance_criteria: string[]
}

export interface DocumentPlan {
  title: string
  rationale: string
  deliverable_mode: 'evidence_summary' | 'normative_synthesis'
  chapters: ChapterPlan[]
}

export interface ConsistencyIssue {
  issue_id: string
  severity: 'warning' | 'error'
  chapter_ids: string[]
  description: string
  recommendation: string
}

export interface ResearchPacket {
  task: string
  chapter_id: string | null
  chapter_title: string | null
  depends_on: string[]
  status: 'sufficient' | 'insufficient' | 'failed' | 'blocked'
  summary: string
  prose: string
  rules: RuleRecord[]
  contracts: ContractRecord[]
  conflicts: Conflict[]
  gaps: string[]
  diagnostics: string[]
  evidence_ids: string[]
}

export interface AgentRun {
  run_id: string
  request: string
  route: RouteDecision
  outcome: 'completed' | 'incomplete'
  answer: {
    content: string
    evidence_ids: string[]
    limitations: string[]
  }
  document_plan: DocumentPlan | null
  consistency_issues: ConsistencyIssue[]
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

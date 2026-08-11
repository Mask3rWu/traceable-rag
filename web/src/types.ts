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

export interface Claim {
  claim_id: string
  text: string
  conclusion_type: 'direct' | 'synthesized' | 'normative' | 'hypothesis'
  citations: Citation[]
}

export interface Conflict {
  conflict_id: string
  description: string
  claim_ids: string[]
  evidence_ids: string[]
  status: 'open' | 'resolved'
  resolution: string | null
}

export interface ChapterPlan {
  chapter_id: string
  ordinal: number
  title: string
  objective: string
  research_questions: string[]
  depends_on: string[]
  produces_decisions: string[]
  required_decisions: string[]
  acceptance_criteria: string[]
}

export interface DocumentPlan {
  title: string
  rationale: string
  deliverable_mode: 'evidence_summary' | 'normative_synthesis'
  chapters: ChapterPlan[]
}

export interface ContentBlock {
  block_id: string
  heading: string | null
  markdown: string
  claim_ids: string[]
  decision_ids: string[]
  evidence_ids: string[]
}

export interface DecisionRecord {
  decision_id: string
  statement: string
  decision_type: 'direct' | 'synthesized' | 'normative' | 'hypothesis'
  rationale: string
  claim_ids: string[]
  evidence_ids: string[]
  assumptions: string[]
  alternatives: string[]
  validation_requirements: string[]
  confidence: 'high' | 'medium' | 'low'
  applies_to_chapters: string[]
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
  content_blocks: ContentBlock[]
  claims: Claim[]
  decisions: DecisionRecord[]
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

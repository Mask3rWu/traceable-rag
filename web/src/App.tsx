import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity, AlertTriangle, Bot, Check, CheckCircle2, ChevronRight, CircleStop,
  Copy, FileSearch, History, Lightbulb, Link2, ListTree, PanelRight, Plus, RefreshCw,
  Search, Send, Sparkles, Users, X,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { cancelRun, createRun, getRun, listRuns, subscribeRun, visualUrl } from './api'
import type { Claim, DecisionRecord, Evidence, ResearchPacket, RunDetail, RunEvent, RunStatus, RunSummary } from './types'

const activeStatuses = new Set<RunStatus>(['queued', 'running', 'cancel_requested'])
const statusLabels: Record<RunStatus, string> = {
  queued: '排队中', running: '研究中', cancel_requested: '停止中', cancelled: '已取消', completed: '已完成', incomplete: '未完整生成', failed: '失败',
}
const eventLabels: Record<string, string> = {
  queued: '任务已进入队列', running: '研究运行已启动', stage_started: '进入执行阶段',
  route_selected: '路由决策完成', tool_started: '调用研究工具', tool_completed: '工具返回结果',
  tool_failed: '工具调用失败', cancel_requested: '已请求停止', cancelled: '任务已取消',
  completed: '研究完成', incomplete: '研究未完整生成', failed: '研究失败',
}
const confidenceLabels = { high: '高', medium: '中', low: '低' }
const conclusionLabels: Record<string, string> = { direct: '直接', synthesized: '综合', normative: '规范', hypothesis: '假设' }
const packetStatusLabels: Record<ResearchPacket['status'], string> = {
  sufficient: '研究完成', insufficient: '参考依据不足', failed: '执行失败', blocked: '依赖阻塞',
}

function formatTime(value: string): string {
  return new Intl.DateTimeFormat('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }).format(new Date(value))
}

function pageLabel(evidence: Evidence): string {
  return evidence.page_start === evidence.page_end ? `第 ${evidence.page_start} 页` : `第 ${evidence.page_start}-${evidence.page_end} 页`
}

function StatusBadge({ status }: { status: RunStatus }) {
  return <span className={`status status-${status}`}><span className="status-dot" />{statusLabels[status]}</span>
}

function IdChip({ id, label, copied, onCopy, hint }: { id: string; label: string; copied: boolean; onCopy: (id: string) => void; hint: string }) {
  return (
    <button className={`run-id-chip${copied ? ' copied' : ''}`} onClick={() => onCopy(id)} title={`点击复制 · ${hint}`}>
      <span className="run-id-label">{label}</span>
      <code>{id}</code>
      {copied ? <Check size={12} /> : <Copy size={12} />}
    </button>
  )
}

function EmptyWorkspace() {
  return (
    <div className="empty-workspace">
      <div className="empty-mark"><FileSearch size={28} /></div>
      <h2>开始一次可溯源研究</h2>
      <p>输入领域问题或标准生成任务，系统会自动选择快速检索或章节化多 Agent 研究。</p>
      <div className="example-list">
        <button data-example="装甲车辆的F级毁伤是什么意思？">装甲车辆的 F 级毁伤是什么意思？</button>
        <button data-example="生成一份装甲车辆视觉毁伤评估标准，并给出各项规则的原文依据。">生成装甲车辆视觉毁伤评估标准</button>
      </div>
    </div>
  )
}

function ActivityTimeline({ events }: { events: RunEvent[] }) {
  return (
    <div className="activity-list">
      {events.length === 0 ? <div className="activity-placeholder"><Activity size={16} />等待运行事件...</div> : events.map((event) => {
        const detail = event.data.chapter_title ?? event.data.query ?? event.data.task ?? event.data.stage ?? event.data.mode
        return (
          <div className="activity-item" key={event.sequence}>
            <div className="activity-node" /><div><strong>{eventLabels[event.type] ?? event.type}</strong>{detail ? <p>{String(detail)}</p> : null}</div>
            <time>{new Date(event.created_at).toLocaleTimeString('zh-CN', { hour12: false })}</time>
          </div>
        )
      })}
    </div>
  )
}

function EvidenceCard({ run, evidence, claims, decisions, selected, onSelect }: {
  run: RunDetail
  evidence: Evidence
  claims: Claim[]
  decisions: DecisionRecord[]
  selected: boolean
  onSelect: (id: string | null) => void
}) {
  const ref = useRef<HTMLElement | null>(null)
  useEffect(() => {
    if (selected) ref.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [selected])
  return (
    <article ref={ref} className={`evidence-card${selected ? ' selected' : ''}`} onClick={() => onSelect(selected ? null : evidence.evidence_id)}>
      <div className="evidence-card-head">
        <strong className="evidence-card-title">{evidence.source_file}</strong>
        <span className="evidence-card-meta"><code>{evidence.evidence_id}</code><ChevronRight size={15} /></span>
      </div>
      {decisions.length > 0 ? (
        <section className="ec-sec">
          <span className="ec-title">形成决策</span>
          <div className="ec-sec-body usage-decision">
            {decisions.map((decision) => <p key={decision.decision_id}><em>{decision.decision_id} [{conclusionLabels[decision.decision_type]}]</em>{decision.statement}</p>)}
          </div>
        </section>
      ) : null}
      {selected ? (
        <>
          {claims.length > 0 ? (
            <section className="ec-sec">
              <span className="ec-title">关键片段</span>
              <div className="ec-sec-body usage-claim">
                {claims.map((claim) => <p key={claim.claim_id}><em>{claim.claim_id} [{conclusionLabels[claim.conclusion_type]}]</em>{claim.text}</p>)}
              </div>
            </section>
          ) : null}
          <section className="ec-sec">
            <span className="ec-title">片段概要</span>
            <p className="evidence-card-preview">{evidence.quote}</p>
          </section>
          <section className="ec-sec">
            <span className="ec-title">原文片段</span>
            <p className="evidence-card-src">{evidence.section_path.join(' › ') || '未标注章节'} · {pageLabel(evidence)}</p>
            <blockquote className="evidence-card-quote">{evidence.quote}</blockquote>
            {evidence.visual_assets.length > 0 ? <div className="evidence-visuals">{evidence.visual_assets.map((visual) => visual.image_crop ? <figure key={visual.block_id}><img src={visualUrl(run.run_id, evidence.evidence_id, visual.block_id)} alt={visual.description ?? visual.block_type} />{visual.description ? <figcaption>{visual.description}</figcaption> : null}</figure> : null)}</div> : null}
            {evidence.retrieval.length > 0 ? <section className="trace-list"><h4>检索轨迹</h4>{evidence.retrieval.map((trace, index) => <div key={`${trace.query}-${index}`}><span>#{trace.final_rank}</span><p>{trace.query}</p></div>)}</section> : null}
          </section>
        </>
      ) : null}
    </article>
  )
}

function ChapterResearch({ run, packet, evidenceById, cardIds, selectedId, onSelect }: {
  run: RunDetail
  packet: ResearchPacket | null
  evidenceById: Map<string, Evidence>
  cardIds: string[]
  selectedId: string | null
  onSelect: (id: string | null) => void
}) {
  if (cardIds.length === 0) return <div className="inspector-empty"><ListTree size={22} /><p>暂无可展示的研究依据。</p></div>
  return (
    <div className="chapter-research">
      {packet ? (
        <div className="research-summary">
          <span className={`packet-status ${packet.status}`}>{packetStatusLabels[packet.status]}</span>
          <p>{packet.summary}</p>
        </div>
      ) : null}
      <div className="evidence-panel-head"><span>{packet ? '本章依据' : '采用依据'}</span><em>{cardIds.length}</em></div>
      <div className="evidence-card-list">
        {cardIds.map((id) => {
          const evidence = evidenceById.get(id)
          if (!evidence) return null
          const claims = packet?.claims.filter((c) => c.citations.some((citation) => citation.evidence_id === id)) ?? []
          const decisions = packet?.decisions.filter((d) => d.evidence_ids.includes(id)) ?? []
          return <EvidenceCard key={id} run={run} evidence={evidence} claims={claims} decisions={decisions} selected={selectedId === id} onSelect={onSelect} />
        })}
      </div>
      {packet?.decisions.length ? (
        <details className="decision-details">
          <summary><Lightbulb size={14} />关键决策明细 <span>{packet.decisions.length}</span></summary>
          {packet.decisions.map((decision) => (
            <article className="decision-record" key={decision.decision_id}>
              <div className="record-label"><code>{decision.decision_id}</code><span>{conclusionLabels[decision.decision_type]}</span><span>置信度 {confidenceLabels[decision.confidence]}</span></div>
              <h3>{decision.statement}</h3>
              <p>{decision.rationale}</p>
              {decision.alternatives.length > 0 ? <div className="record-meta"><strong>考虑的替代方案</strong>{decision.alternatives.join('、')}</div> : null}
              {decision.assumptions.length > 0 ? <div className="record-meta"><strong>适用假设</strong>{decision.assumptions.join('；')}</div> : null}
              {decision.validation_requirements.length > 0 ? <div className="record-meta"><strong>验证要求</strong>{decision.validation_requirements.join('；')}</div> : null}
            </article>
          ))}
        </details>
      ) : null}
      {packet?.gaps.length ? <section className="research-gaps"><h4>证据缺口</h4><ul>{packet.gaps.map((gap) => <li key={gap}>{gap}</li>)}</ul></section> : null}
      {(packet?.diagnostics ?? []).length ? <section className="research-gaps"><h4>执行诊断</h4><ul>{(packet?.diagnostics ?? []).map((item) => <li key={item}>{item}</li>)}</ul></section> : null}
    </div>
  )
}

function App() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [selectedRun, setSelectedRun] = useState<RunDetail | null>(null)
  const [events, setEvents] = useState<RunEvent[]>([])
  const [prompt, setPrompt] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(null)
  const [selectedChapterId, setSelectedChapterId] = useState<string | null>(null)
  const [mobilePanel, setMobilePanel] = useState<'history' | 'result' | 'sources'>('result')
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const subscription = useRef<(() => void) | null>(null)

  const refreshRuns = useCallback(async () => setRuns((await listRuns()).items), [])

  const copyId = useCallback((id: string) => {
    navigator.clipboard?.writeText(id).then(() => {
      setCopiedId(id)
      window.setTimeout(() => setCopiedId((cur) => (cur === id ? null : cur)), 1200)
    }).catch(() => {})
  }, [])

  useEffect(() => {
    const timer = window.setTimeout(() => refreshRuns().catch((reason: Error) => setError(reason.message)), 0)
    return () => { window.clearTimeout(timer); subscription.current?.() }
  }, [refreshRuns])

  useEffect(() => {
    const handleExample = (event: MouseEvent) => {
      const example = (event.target as HTMLElement).closest<HTMLElement>('[data-example]')?.dataset.example
      if (example) setPrompt(example)
    }
    document.addEventListener('click', handleExample)
    return () => document.removeEventListener('click', handleExample)
  }, [])

  const selectInitialChapter = useCallback((detail: RunDetail) => {
    setSelectedChapterId(detail.result?.document_plan?.chapters[0]?.chapter_id ?? detail.result?.worker_packets[0]?.chapter_id ?? null)
  }, [])

  const openRun = useCallback(async (runId: string) => {
    subscription.current?.(); setError(null); setEvents([]); setSelectedEvidenceId(null)
    try {
      const detail = await getRun(runId)
      setSelectedRun(detail); selectInitialChapter(detail); setMobilePanel('result')
      if (activeStatuses.has(detail.status)) {
        subscription.current = subscribeRun(runId, (event) => {
          setEvents((current) => [...current.filter((item) => item.sequence !== event.sequence), event])
          if (['completed', 'incomplete', 'failed', 'cancelled'].includes(event.type)) {
            getRun(runId).then((next) => { setSelectedRun(next); selectInitialChapter(next) }).catch(() => undefined)
            refreshRuns().catch(() => undefined)
          }
        }, () => undefined)
      }
    } catch (reason) { setError((reason as Error).message) }
  }, [refreshRuns, selectInitialChapter])

  const submit = async (event: FormEvent) => {
    event.preventDefault(); const request = prompt.trim(); if (!request || submitting) return
    setSubmitting(true); setError(null)
    try { const run = await createRun(request); setPrompt(''); await refreshRuns(); await openRun(run.run_id) }
    catch (reason) { setError((reason as Error).message) } finally { setSubmitting(false) }
  }

  const stop = async () => {
    if (!selectedRun) return
    try { const summary = await cancelRun(selectedRun.run_id); setSelectedRun((current) => current ? { ...current, ...summary } : current); await refreshRuns() }
    catch (reason) { setError((reason as Error).message) }
  }

  const evidence = useMemo(() => selectedRun?.result?.evidence ?? [], [selectedRun])
  const workers = useMemo(() => selectedRun?.result?.worker_packets ?? [], [selectedRun])
  const plan = selectedRun?.result?.document_plan ?? null
  const evidenceById = useMemo(() => new Map(evidence.map((item) => [item.evidence_id, item])), [evidence])
  const selectedPacket = workers.find((item) => item.chapter_id === selectedChapterId) ?? null
  const answerEvidence = useMemo(() => new Set(selectedRun?.result?.answer.evidence_ids ?? []), [selectedRun])
  const chapterEvidence = useMemo(() => new Set(selectedPacket?.evidence_ids ?? []), [selectedPacket])

  const showEvidence = (id: string | null) => { setSelectedEvidenceId(id); setMobilePanel('sources') }
  const chooseChapter = (chapterId: string) => { setSelectedChapterId(chapterId); setSelectedEvidenceId(null) }
  const cardIds = useMemo(() => {
    if (selectedPacket?.evidence_ids?.length) return selectedPacket.evidence_ids
    return Array.from(answerEvidence)
  }, [selectedPacket, answerEvidence])

  return (
    <div className="app-shell">
      <header className="topbar"><div className="brand"><div className="brand-mark"><Search size={18} /></div><span>证据研究工作台</span></div><div className="topbar-meta"><span className="system-state"><span />系统就绪</span></div></header>
      <nav className="mobile-tabs" aria-label="工作区导航">
        <button className={mobilePanel === 'history' ? 'active' : ''} onClick={() => setMobilePanel('history')}><History size={17} />历史</button>
        <button className={mobilePanel === 'result' ? 'active' : ''} onClick={() => setMobilePanel('result')}><Bot size={17} />结果</button>
        <button className={mobilePanel === 'sources' ? 'active' : ''} onClick={() => setMobilePanel('sources')}><PanelRight size={17} />依据</button>
      </nav>

      <main className="workspace">
        <aside className={`history-panel mobile-${mobilePanel}`}>
          <div className="panel-header"><div><span className="eyebrow">RUNS</span><h2>研究历史</h2></div><button className="icon-button" onClick={() => refreshRuns()} title="刷新历史"><RefreshCw size={16} /></button></div>
          <button className="new-run" onClick={() => { setSelectedRun(null); setEvents([]); setMobilePanel('result') }}><Plus size={16} />新建研究</button>
          <div className="run-list">{runs.map((run) => <button key={run.run_id} className={`run-item ${selectedRun?.run_id === run.run_id ? 'selected' : ''}`} onClick={() => openRun(run.run_id)}><div className="run-item-top"><StatusBadge status={run.status} /><time>{formatTime(run.created_at)}</time></div><p>{run.request}</p><div className="run-meta"><span>{run.route === 'fast' ? '快速路径' : run.route === 'supervisor' ? '章节研究' : '等待路由'}</span><span>{run.evidence_count} 条证据</span></div></button>)}{runs.length === 0 ? <p className="no-runs">尚无研究记录</p> : null}</div>
        </aside>

        <section className={`result-panel mobile-${mobilePanel}`}>
          <form className="composer" onSubmit={submit}><textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="输入领域问题，或描述需要生成的评估标准..." rows={2} /><button type="submit" disabled={!prompt.trim() || submitting} title="提交研究任务">{submitting ? <RefreshCw className="spin" size={18} /> : <Send size={18} />}<span>开始研究</span></button></form>
          {error ? <div className="error-banner"><AlertTriangle size={17} />{error}<button onClick={() => setError(null)}><X size={15} /></button></div> : null}
          {!selectedRun ? <EmptyWorkspace /> : (
            <div className="run-view">
              <div className="run-titlebar"><div><div className="route-line"><StatusBadge status={selectedRun.status} />{selectedRun.route ? <span className="route-badge">{selectedRun.route === 'fast' ? <Sparkles size={14} /> : <Users size={14} />}{selectedRun.route === 'fast' ? '快速检索' : '章节化多 Agent'}</span> : null}</div><h1>{selectedRun.request}</h1><div className="run-ids"><IdChip id={selectedRun.run_id} label="Run" copied={copiedId === selectedRun.run_id} onCopy={copyId} hint="运行 ID（对应 processed/research/agent-runs 目录）" />{selectedRun.trace_id ? <IdChip id={selectedRun.trace_id} label="Trace" copied={copiedId === selectedRun.trace_id} onCopy={copyId} hint="Langfuse 链路追踪 ID" /> : null}</div></div>{activeStatuses.has(selectedRun.status) ? <button className="stop-button" onClick={stop}><CircleStop size={16} />停止</button> : null}</div>
              {activeStatuses.has(selectedRun.status) ? <section className="activity-section"><div className="section-heading"><Activity size={17} /><h2>运行活动</h2></div><ActivityTimeline events={events} /></section> : null}
              {selectedRun.error ? <div className="failure"><AlertTriangle size={18} /><div><strong>运行失败</strong><p>{selectedRun.error}</p></div></div> : null}
              {selectedRun.result ? (
                plan ? (
                  <div className="chapter-workspace">
                    <nav className="chapter-toc" aria-label="章节目录"><div><ListTree size={15} /><strong>章节目录</strong></div>{[...plan.chapters].sort((a, b) => a.ordinal - b.ordinal).map((chapter) => { const packet = workers.find((item) => item.chapter_id === chapter.chapter_id); return <button key={chapter.chapter_id} className={selectedChapterId === chapter.chapter_id ? 'selected' : ''} onClick={() => chooseChapter(chapter.chapter_id)}><span>{chapter.ordinal}</span><div><strong>{chapter.title}</strong><small>{packet ? packetStatusLabels[packet.status] : '待研究'}</small></div><i className={packet?.status ?? 'pending'} /></button>})}</nav>
                    <article className="answer-document chapter-document">
                      <div className="answer-header"><span>{selectedPacket?.chapter_title ?? plan.title}</span><div><span>{chapterEvidence.size} 条本章依据</span></div></div>
                      {selectedPacket?.content_blocks?.length ? selectedPacket.content_blocks.map((block) => <section className="content-block" id={block.block_id} key={block.block_id}>{block.heading ? <h3>{block.heading}</h3> : null}<ReactMarkdown remarkPlugins={[remarkGfm]}>{block.markdown}</ReactMarkdown><div className="block-sources">{block.evidence_ids.map((id) => <button key={id} onClick={() => showEvidence(id)}><Link2 size={12} />{id}</button>)}</div></section>) : <div className="chapter-empty"><AlertTriangle size={20} /><p>{selectedPacket?.summary ?? '该章节尚未产生研究正文。'}</p></div>}
                      {selectedPacket?.gaps.length ? <section className="limitations"><h3><AlertTriangle size={17} />本章限制与缺口</h3><ul>{selectedPacket.gaps.map((item) => <li key={item}>{item}</li>)}</ul></section> : null}
                    </article>
                  </div>
                ) : (
                  <article className="answer-document"><div className="answer-header"><span>研究结果</span><div><span>{answerEvidence.size} 条证据</span></div></div><ReactMarkdown remarkPlugins={[remarkGfm]}>{selectedRun.result.answer.content}</ReactMarkdown>{selectedRun.result.answer.limitations.length > 0 ? <section className="limitations"><h3><AlertTriangle size={17} />限制与缺口</h3><ul>{selectedRun.result.answer.limitations.map((item) => <li key={item}>{item}</li>)}</ul></section> : null}<section className="answer-sources"><h3>引用来源</h3><div className="source-buttons">{selectedRun.result.answer.evidence_ids.map((id) => <button key={id} onClick={() => showEvidence(id)}>{id}<ChevronRight size={14} /></button>)}</div></section></article>
                )
              ) : null}
            </div>
          )}
        </section>

        <aside className={`inspector-panel mobile-${mobilePanel}`}>
          <ChapterResearch run={selectedRun ?? ({ run_id: '' } as RunDetail)} packet={selectedPacket} evidenceById={evidenceById} cardIds={cardIds} selectedId={selectedEvidenceId} onSelect={showEvidence} />
        </aside>
      </main>
    </div>
  )
}

export default App

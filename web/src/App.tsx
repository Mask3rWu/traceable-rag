import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  BookOpen,
  Bot,
  CheckCircle2,
  ChevronRight,
  CircleStop,
  FileSearch,
  History,
  PanelRight,
  Plus,
  RefreshCw,
  Search,
  Send,
  Sparkles,
  Users,
  X,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { cancelRun, createRun, getRun, listRuns, subscribeRun, visualUrl } from './api'
import type { Evidence, ResearchPacket, RunDetail, RunEvent, RunStatus, RunSummary } from './types'

const activeStatuses = new Set<RunStatus>(['queued', 'running', 'cancel_requested'])

const statusLabels: Record<RunStatus, string> = {
  queued: '排队中',
  running: '研究中',
  cancel_requested: '停止中',
  cancelled: '已取消',
  completed: '已完成',
  failed: '失败',
}

const eventLabels: Record<string, string> = {
  queued: '任务已进入队列',
  running: '研究运行已启动',
  stage_started: '进入执行阶段',
  route_selected: '路由决策完成',
  tool_started: '调用研究工具',
  tool_completed: '工具返回结果',
  tool_failed: '工具调用失败',
  cancel_requested: '已请求停止',
  cancelled: '任务已取消',
  completed: '研究完成',
  failed: '研究失败',
}

function formatTime(value: string): string {
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value))
}

function pageLabel(evidence: Evidence): string {
  return evidence.page_start === evidence.page_end
    ? `第 ${evidence.page_start} 页`
    : `第 ${evidence.page_start}-${evidence.page_end} 页`
}

function StatusBadge({ status }: { status: RunStatus }) {
  return (
    <span className={`status status-${status}`}>
      <span className="status-dot" />
      {statusLabels[status]}
    </span>
  )
}

function EmptyWorkspace() {
  return (
    <div className="empty-workspace">
      <div className="empty-mark"><FileSearch size={28} /></div>
      <h2>开始一次可溯源研究</h2>
      <p>输入领域问题或标准生成任务，系统会自动选择快速检索或多 Agent 研究路径。</p>
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
      {events.length === 0 ? (
        <div className="activity-placeholder"><Activity size={16} />等待运行事件...</div>
      ) : events.map((event) => {
        const detail = event.data.query ?? event.data.task ?? event.data.stage ?? event.data.mode
        return (
          <div className="activity-item" key={event.sequence}>
            <div className="activity-node" />
            <div>
              <strong>{eventLabels[event.type] ?? event.type}</strong>
              {detail ? <p>{String(detail)}</p> : null}
            </div>
            <time>{new Date(event.created_at).toLocaleTimeString('zh-CN', { hour12: false })}</time>
          </div>
        )
      })}
    </div>
  )
}

function EvidenceInspector({
  run,
  evidence,
}: {
  run: RunDetail
  evidence: Evidence | null
}) {
  if (!evidence) {
    return (
      <div className="inspector-empty">
        <BookOpen size={22} />
        <p>选择一条来源查看原文、页码和检索轨迹。</p>
      </div>
    )
  }
  return (
    <article className="evidence-detail">
      <div className="evidence-heading">
        <span className="evidence-id">{evidence.evidence_id}</span>
        {evidence.verified ? <span className="verified"><CheckCircle2 size={14} />已核验</span> : null}
      </div>
      <h3>{evidence.source_file}</h3>
      <p className="source-path">{pageLabel(evidence)} · {evidence.section_path.join(' › ') || '未标注章节'}</p>
      <blockquote>{evidence.quote}</blockquote>
      {evidence.visual_assets.map((visual) => (
        <figure key={visual.block_id}>
          {visual.image_crop ? (
            <img
              src={visualUrl(run.run_id, evidence.evidence_id, visual.block_id)}
              alt={visual.description ?? visual.block_type}
            />
          ) : null}
          {visual.description ? <figcaption>{visual.description}</figcaption> : null}
        </figure>
      ))}
      <section className="trace-list">
        <h4>检索轨迹</h4>
        {evidence.retrieval.map((trace, index) => (
          <div key={`${trace.query}-${index}`}>
            <span>#{trace.final_rank}</span>
            <p>{trace.query}</p>
          </div>
        ))}
      </section>
    </article>
  )
}

function WorkerInspector({ packet }: { packet: ResearchPacket }) {
  return (
    <article className="worker-detail">
      <div className="worker-title">
        <Users size={17} />
        <h3>{packet.task}</h3>
      </div>
      <span className={`packet-status ${packet.status}`}>{packet.status === 'sufficient' ? '证据充分' : '证据不足'}</span>
      <p>{packet.summary}</p>
      {packet.claims.length > 0 ? (
        <section>
          <h4>研究结论</h4>
          {packet.claims.map((claim) => <p className="worker-claim" key={claim.claim_id}>{claim.text}</p>)}
        </section>
      ) : null}
      {packet.gaps.length > 0 ? (
        <section>
          <h4>证据缺口</h4>
          <ul>{packet.gaps.map((gap) => <li key={gap}>{gap}</li>)}</ul>
        </section>
      ) : null}
    </article>
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
  const [selectedWorker, setSelectedWorker] = useState<number | null>(null)
  const [mobilePanel, setMobilePanel] = useState<'history' | 'result' | 'sources'>('result')
  const subscription = useRef<(() => void) | null>(null)

  const refreshRuns = useCallback(async () => {
    const response = await listRuns()
    setRuns(response.items)
  }, [])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      refreshRuns().catch((reason: Error) => setError(reason.message))
    }, 0)
    return () => {
      window.clearTimeout(timer)
      subscription.current?.()
    }
  }, [refreshRuns])

  useEffect(() => {
    const handleExample = (event: MouseEvent) => {
      const target = event.target as HTMLElement
      const example = target.closest<HTMLElement>('[data-example]')?.dataset.example
      if (example) setPrompt(example)
    }
    document.addEventListener('click', handleExample)
    return () => document.removeEventListener('click', handleExample)
  }, [])

  const openRun = useCallback(async (runId: string) => {
    subscription.current?.()
    setError(null)
    setEvents([])
    setSelectedEvidenceId(null)
    setSelectedWorker(null)
    try {
      const detail = await getRun(runId)
      setSelectedRun(detail)
      setMobilePanel('result')
      if (activeStatuses.has(detail.status)) {
        subscription.current = subscribeRun(
          runId,
          (event) => {
            setEvents((current) => [...current.filter((item) => item.sequence !== event.sequence), event])
            if (['completed', 'failed', 'cancelled'].includes(event.type)) {
              getRun(runId).then(setSelectedRun).catch(() => undefined)
              refreshRuns().catch(() => undefined)
            }
          },
          () => undefined,
        )
      }
    } catch (reason) {
      setError((reason as Error).message)
    }
  }, [refreshRuns])

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    const request = prompt.trim()
    if (!request || submitting) return
    setSubmitting(true)
    setError(null)
    try {
      const run = await createRun(request)
      setPrompt('')
      await refreshRuns()
      await openRun(run.run_id)
    } catch (reason) {
      setError((reason as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  const stop = async () => {
    if (!selectedRun) return
    try {
      const summary = await cancelRun(selectedRun.run_id)
      setSelectedRun((current) => current ? { ...current, ...summary } : current)
      await refreshRuns()
    } catch (reason) {
      setError((reason as Error).message)
    }
  }

  const evidence = selectedRun?.result?.evidence ?? []
  const answerEvidence = useMemo(() => new Set(selectedRun?.result?.answer.evidence_ids ?? []), [selectedRun])
  const workerEvidence = useMemo(() => {
    const ids = new Set<string>()
    for (const packet of selectedRun?.result?.worker_packets ?? []) {
      packet.evidence_ids.forEach((id) => ids.add(id))
      packet.claims.forEach((claim) => claim.citations.forEach((citation) => ids.add(citation.evidence_id)))
    }
    return ids
  }, [selectedRun])
  const selectedEvidence = evidence.find((item) => item.evidence_id === selectedEvidenceId) ?? null
  const workers = selectedRun?.result?.worker_packets ?? []

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><div className="brand-mark"><Search size={18} /></div><span>证据研究工作台</span></div>
        <div className="topbar-meta">
          <span className="system-state"><span />系统就绪</span>
          {selectedRun?.trace_id ? <span className="trace-label">Trace {selectedRun.trace_id.slice(0, 8)}</span> : null}
        </div>
      </header>

      <nav className="mobile-tabs" aria-label="工作区导航">
        <button className={mobilePanel === 'history' ? 'active' : ''} onClick={() => setMobilePanel('history')}><History size={17} />历史</button>
        <button className={mobilePanel === 'result' ? 'active' : ''} onClick={() => setMobilePanel('result')}><Bot size={17} />结果</button>
        <button className={mobilePanel === 'sources' ? 'active' : ''} onClick={() => setMobilePanel('sources')}><PanelRight size={17} />来源</button>
      </nav>

      <main className="workspace">
        <aside className={`history-panel mobile-${mobilePanel}`}>
          <div className="panel-header">
            <div><span className="eyebrow">RUNS</span><h2>研究历史</h2></div>
            <button className="icon-button" onClick={() => refreshRuns()} title="刷新历史"><RefreshCw size={16} /></button>
          </div>
          <button className="new-run" onClick={() => { setSelectedRun(null); setEvents([]); setMobilePanel('result') }}><Plus size={16} />新建研究</button>
          <div className="run-list">
            {runs.map((run) => (
              <button key={run.run_id} className={`run-item ${selectedRun?.run_id === run.run_id ? 'selected' : ''}`} onClick={() => openRun(run.run_id)}>
                <div className="run-item-top"><StatusBadge status={run.status} /><time>{formatTime(run.created_at)}</time></div>
                <p>{run.request}</p>
                <div className="run-meta">
                  {run.route ? <span>{run.route === 'fast' ? '快速路径' : '多 Agent'}</span> : <span>等待路由</span>}
                  <span>{run.evidence_count} 条证据</span>
                </div>
              </button>
            ))}
            {runs.length === 0 ? <p className="no-runs">尚无研究记录</p> : null}
          </div>
        </aside>

        <section className={`result-panel mobile-${mobilePanel}`}>
          <form className="composer" onSubmit={submit}>
            <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="输入领域问题，或描述需要生成的评估标准..." rows={2} />
            <button type="submit" disabled={!prompt.trim() || submitting} title="提交研究任务">
              {submitting ? <RefreshCw className="spin" size={18} /> : <Send size={18} />}
              <span>开始研究</span>
            </button>
          </form>
          {error ? <div className="error-banner"><AlertTriangle size={17} />{error}<button onClick={() => setError(null)}><X size={15} /></button></div> : null}

          {!selectedRun ? <EmptyWorkspace /> : (
            <div className="run-view">
              <div className="run-titlebar">
                <div>
                  <div className="route-line">
                    <StatusBadge status={selectedRun.status} />
                    {selectedRun.route ? <span className="route-badge">{selectedRun.route === 'fast' ? <Sparkles size={14} /> : <Users size={14} />}{selectedRun.route === 'fast' ? '快速检索' : 'Supervisor + Workers'}</span> : null}
                  </div>
                  <h1>{selectedRun.request}</h1>
                  {selectedRun.route_reason ? <p>{selectedRun.route_reason}</p> : null}
                </div>
                {activeStatuses.has(selectedRun.status) ? <button className="stop-button" onClick={stop}><CircleStop size={16} />停止</button> : null}
              </div>

              {activeStatuses.has(selectedRun.status) ? (
                <section className="activity-section">
                  <div className="section-heading"><Activity size={17} /><h2>运行活动</h2></div>
                  <ActivityTimeline events={events} />
                </section>
              ) : null}

              {selectedRun.error ? <div className="failure"><AlertTriangle size={18} /><div><strong>运行失败</strong><p>{selectedRun.error}</p></div></div> : null}

              {selectedRun.result ? (
                <article className="answer-document">
                  <div className="answer-header"><span>研究结果</span><div><span>{evidence.length} 条证据</span><span>{workers.length} 个 Worker</span></div></div>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{selectedRun.result.answer.content}</ReactMarkdown>
                  {selectedRun.result.answer.limitations.length > 0 ? (
                    <section className="limitations"><h3><AlertTriangle size={17} />限制与缺口</h3><ul>{selectedRun.result.answer.limitations.map((item) => <li key={item}>{item}</li>)}</ul></section>
                  ) : null}
                  <section className="answer-sources">
                    <h3>引用来源</h3>
                    <div className="source-buttons">
                      {selectedRun.result.answer.evidence_ids.map((id) => (
                        <button key={id} onClick={() => { setSelectedEvidenceId(id); setSelectedWorker(null); setMobilePanel('sources') }}>{id}<ChevronRight size={14} /></button>
                      ))}
                    </div>
                  </section>
                </article>
              ) : null}
            </div>
          )}
        </section>

        <aside className={`inspector-panel mobile-${mobilePanel}`}>
          <div className="inspector-tabs">
            <button className={selectedWorker === null ? 'active' : ''} onClick={() => setSelectedWorker(null)}><BookOpen size={15} />证据 <span>{evidence.length}</span></button>
            <button className={selectedWorker !== null ? 'active' : ''} onClick={() => workers.length && setSelectedWorker(0)}><Users size={15} />Workers <span>{workers.length}</span></button>
          </div>
          {selectedWorker === null ? (
            <>
              {evidence.length > 0 ? (
                <div className="evidence-legend">
                  <span><i className="legend-cited" />最终引用</span>
                  <span><i className="legend-worker" />Worker 依据</span>
                  <span><i className="legend-candidate" />检索候选</span>
                </div>
              ) : null}
              <div className="evidence-list">
                {evidence.map((item) => (
                  <button key={item.evidence_id} className={`${selectedEvidenceId === item.evidence_id ? 'selected' : ''} ${answerEvidence.has(item.evidence_id) ? 'cited' : ''}`} onClick={() => setSelectedEvidenceId(item.evidence_id)}>
                    <div>
                      <span>{item.evidence_id}</span>
                      {answerEvidence.has(item.evidence_id) ? (
                        <em className="final">最终引用</em>
                      ) : workerEvidence.has(item.evidence_id) ? (
                        <em className="worker">Worker 依据</em>
                      ) : (
                        <em className="candidate">检索候选</em>
                      )}
                    </div>
                    <strong>{item.source_file}</strong>
                    <p>{pageLabel(item)} · {item.section_path.at(-1) ?? '未标注章节'}</p>
                  </button>
                ))}
              </div>
              <EvidenceInspector run={selectedRun ?? ({ run_id: '' } as RunDetail)} evidence={selectedEvidence} />
            </>
          ) : (
            <div className="worker-inspector">
              <div className="worker-list">
                {workers.map((worker, index) => <button key={`${worker.task}-${index}`} className={selectedWorker === index ? 'selected' : ''} onClick={() => setSelectedWorker(index)}>Worker {index + 1}<span>{worker.status === 'sufficient' ? '充分' : '不足'}</span></button>)}
              </div>
              {workers[selectedWorker] ? <WorkerInspector packet={workers[selectedWorker]} /> : <div className="inspector-empty"><Users size={22} /><p>当前运行没有 Worker 研究包。</p></div>}
            </div>
          )}
        </aside>
      </main>
    </div>
  )
}

export default App

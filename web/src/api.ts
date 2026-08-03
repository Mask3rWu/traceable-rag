import type { RunDetail, RunEvent, RunSummary } from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(body.detail ?? `Request failed: ${response.status}`)
  }
  return response.json() as Promise<T>
}

export function listRuns(): Promise<{ items: RunSummary[] }> {
  return request('/api/runs')
}

export function getRun(runId: string): Promise<RunDetail> {
  return request(`/api/runs/${runId}`)
}

export function createRun(prompt: string): Promise<RunSummary> {
  return request('/api/runs', {
    method: 'POST',
    body: JSON.stringify({ request: prompt }),
  })
}

export function cancelRun(runId: string): Promise<RunSummary> {
  return request(`/api/runs/${runId}/cancel`, { method: 'POST' })
}

const eventNames = [
  'queued',
  'running',
  'stage_started',
  'route_selected',
  'tool_started',
  'tool_completed',
  'tool_failed',
  'cancel_requested',
  'cancelled',
  'completed',
  'failed',
]

export function subscribeRun(
  runId: string,
  onEvent: (event: RunEvent) => void,
  onDisconnect: () => void,
): () => void {
  const source = new EventSource(`/api/runs/${runId}/events`)
  for (const name of eventNames) {
    source.addEventListener(name, (raw) => {
      const event = raw as MessageEvent<string>
      onEvent(JSON.parse(event.data) as RunEvent)
      if (['completed', 'failed', 'cancelled'].includes(name)) {
        source.close()
        onDisconnect()
      }
    })
  }
  source.onerror = () => {
    source.close()
    onDisconnect()
  }
  return () => source.close()
}

export function visualUrl(runId: string, evidenceId: string, blockId: string): string {
  return `/api/runs/${runId}/evidence/${evidenceId}/visuals/${blockId}`
}

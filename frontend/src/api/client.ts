const BASE = '/api'

export type MediaType = 'text' | 'image' | 'audio' | 'video'

export interface IngestResponse {
  task_id: string
  filename: string
  media_type: MediaType
  status: string
  message: string
}

export interface TaskStatus {
  task_id: string
  status: 'PENDING' | 'STARTED' | 'SUCCESS' | 'FAILURE' | 'RETRY'
  progress?: number
  result?: Record<string, unknown>
  error?: string
}

export interface ChunkMetadata {
  source_file: string
  media_type: MediaType
  chunk_index: number
  timestamp_start?: number
  timestamp_end?: number
  page_number?: number
  frame_index?: number
}

export interface RetrievedChunk {
  id: string
  score: number
  content: string
  metadata: ChunkMetadata
}

export interface RetrievalResponse {
  query: string
  chunks: RetrievedChunk[]
  total_found: number
}

export interface GenerateRequest {
  query: string
  top_k?: number
  modalities?: MediaType[]
  stream?: boolean
  system_prompt?: string
}

// ─── Ingest ──────────────────────────────────────────────────────────────────

export async function ingestFile(file: File): Promise<IngestResponse> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/ingest`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getTaskStatus(taskId: string): Promise<TaskStatus> {
  const res = await fetch(`${BASE}/task/${taskId}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

// ─── Retrieval ───────────────────────────────────────────────────────────────

export async function retrieveChunks(
  query: string,
  topK = 5,
  modalities: MediaType[] = ['text', 'image', 'audio', 'video'],
  scoreThreshold = 0.3
): Promise<RetrievalResponse> {
  const res = await fetch(`${BASE}/retrieve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK, modalities, score_threshold: scoreThreshold }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

// ─── Generate (streaming) ────────────────────────────────────────────────────

export async function* generateStream(req: GenerateRequest): AsyncGenerator<string> {
  const res = await fetch(`${BASE}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...req, stream: true }),
  })
  if (!res.ok) throw new Error(await res.text())

  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6)
        if (data === '[DONE]') return
        yield data
      }
    }
  }
}

// ─── Health ──────────────────────────────────────────────────────────────────

export async function getInfo(): Promise<{ provider: string; model: string }> {
  const res = await fetch(`${BASE}/info`)
  return res.json()
}
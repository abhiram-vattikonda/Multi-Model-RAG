import React, { useState, useEffect } from 'react'
import { Database, Zap } from 'lucide-react'
import UploadPanel from './components/UploadPanel'
import QueryInput from './components/QueryInput'
import RetrievalViewer from './components/RetrievalViewer'
import ResponsePanel from './components/ResponsePanel'
import { retrieveChunks, generateStream, getInfo, RetrievedChunk, MediaType } from './api/client'

export default function App() {
  const [chunks, setChunks] = useState<RetrievedChunk[]>([])
  const [answer, setAnswer] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [retrieving, setRetrieving] = useState(false)
  const [providerInfo, setProviderInfo] = useState<{ provider: string; model: string } | null>(null)
  const [activeTab, setActiveTab] = useState<'retrieve' | 'answer'>('answer')

  useEffect(() => {
    getInfo().then(setProviderInfo).catch(() => null)
  }, [])

  const handleRetrieve = async (query: string, modalities: MediaType[], topK: number) => {
    setRetrieving(true)
    setChunks([])
    setActiveTab('retrieve')
    try {
      const res = await retrieveChunks(query, topK, modalities)
      setChunks(res.chunks)
    } finally {
      setRetrieving(false)
    }
  }

  const handleGenerate = async (query: string, modalities: MediaType[], topK: number) => {
    setAnswer('')
    setChunks([])
    setStreaming(true)
    setActiveTab('answer')

    // Also retrieve chunks in parallel for display
    retrieveChunks(query, topK, modalities).then(res => setChunks(res.chunks)).catch(() => null)

    try {
      let full = ''
      for await (const token of generateStream({ query, top_k: topK, modalities })) {
        full += token
        setAnswer(full)
      }
    } finally {
      setStreaming(false)
    }
  }

  return (
    <div className="min-h-screen bg-ink-950 text-ink-100 font-sans">
      {/* Top bar */}
      <header className="border-b border-ink-800 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded bg-acid flex items-center justify-center">
            <Database size={13} className="text-ink-950" />
          </div>
          <span className="font-medium text-sm tracking-tight">Multimodal RAG</span>
          <span className="text-ink-600 text-xs">·</span>
          <span className="text-xs text-ink-500">vector search + LLM</span>
        </div>
        {providerInfo && (
          <div className="flex items-center gap-1.5 text-xs text-ink-500">
            <Zap size={11} className="text-acid" />
            <span className="font-mono">{providerInfo.provider}/{providerInfo.model}</span>
          </div>
        )}
      </header>

      {/* Body: three-column layout */}
      <div className="flex h-[calc(100vh-49px)]">
        {/* Left: Upload */}
        <aside className="w-72 shrink-0 border-r border-ink-800 flex flex-col">
          <div className="px-4 py-3 border-b border-ink-800">
            <p className="text-xs font-medium text-ink-300 uppercase tracking-wider">Ingest</p>
          </div>
          <div className="flex-1 overflow-y-auto p-4">
            <UploadPanel />
          </div>
        </aside>

        {/* Middle: Query + results */}
        <main className="flex-1 flex flex-col min-w-0">
          {/* Query bar */}
          <div className="border-b border-ink-800 p-4">
            <QueryInput
              onRetrieve={handleRetrieve}
              onGenerate={handleGenerate}
              loading={streaming || retrieving}
            />
          </div>

          {/* Tab switcher */}
          <div className="border-b border-ink-800 px-4 flex gap-1 pt-2">
            {(['answer', 'retrieve'] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`pb-2 px-1 text-xs font-medium border-b-2 transition-colors ${
                  activeTab === tab
                    ? 'border-acid text-acid'
                    : 'border-transparent text-ink-500 hover:text-ink-300'
                }`}
              >
                {tab === 'answer' ? 'LLM Answer' : `Retrieved Chunks${chunks.length ? ` (${chunks.length})` : ''}`}
              </button>
            ))}
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-4">
            {activeTab === 'answer' ? (
              <ResponsePanel
                answer={answer}
                streaming={streaming}
                model={providerInfo?.model}
                provider={providerInfo?.provider}
              />
            ) : (
              <RetrievalViewer chunks={chunks} loading={retrieving} />
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
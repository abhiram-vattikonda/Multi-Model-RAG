import React, { useState } from 'react'
import { Search, SlidersHorizontal } from 'lucide-react'
import clsx from 'clsx'
import { MediaType } from '../api/client'

interface Props {
  onRetrieve: (query: string, modalities: MediaType[], topK: number) => void
  onGenerate: (query: string, modalities: MediaType[], topK: number) => void
  loading: boolean
}

const ALL_MODALITIES: MediaType[] = ['text', 'image', 'audio', 'video']

const MODALITY_LABELS: Record<MediaType, string> = {
  text: 'Text',
  image: 'Image',
  audio: 'Audio',
  video: 'Video',
}

export default function QueryInput({ onRetrieve, onGenerate, loading }: Props) {
  const [query, setQuery] = useState('')
  const [modalities, setModalities] = useState<MediaType[]>([...ALL_MODALITIES])
  const [topK, setTopK] = useState(5)
  const [showFilters, setShowFilters] = useState(false)

  const toggleModality = (m: MediaType) => {
    setModalities(prev =>
      prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m]
    )
  }

  const canSubmit = query.trim().length > 0 && modalities.length > 0 && !loading

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && canSubmit) {
      e.preventDefault()
      onGenerate(query.trim(), modalities, topK)
    }
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Main input */}
      <div className="relative">
        <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-500 pointer-events-none" />
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question across your documents..."
          className={clsx(
            'w-full bg-ink-800 border rounded-lg pl-9 pr-12 py-2.5 text-sm text-ink-100 placeholder:text-ink-500',
            'focus:outline-none focus:border-acid/50 focus:bg-ink-800 transition-colors',
            'border-ink-600'
          )}
          disabled={loading}
        />
        <button
          onClick={() => setShowFilters(v => !v)}
          className={clsx(
            'absolute right-2.5 top-1/2 -translate-y-1/2 p-1 rounded transition-colors',
            showFilters ? 'text-acid' : 'text-ink-500 hover:text-ink-300'
          )}
        >
          <SlidersHorizontal size={13} />
        </button>
      </div>

      {/* Filters */}
      {showFilters && (
        <div className="bg-ink-800 border border-ink-700 rounded-lg p-3 flex flex-col gap-3 animate-slide-up">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-ink-500 w-16 shrink-0">Modalities</span>
            {ALL_MODALITIES.map(m => (
              <button
                key={m}
                onClick={() => toggleModality(m)}
                className={clsx(
                  'px-2.5 py-1 rounded text-xs border transition-all',
                  modalities.includes(m)
                    ? 'border-acid/40 text-acid bg-acid/10'
                    : 'border-ink-600 text-ink-500 hover:border-ink-400'
                )}
              >
                {MODALITY_LABELS[m]}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs text-ink-500 w-16 shrink-0">Top K</span>
            <input
              type="range"
              min={1}
              max={20}
              value={topK}
              onChange={e => setTopK(Number(e.target.value))}
              className="flex-1 accent-[#b4ff4e] h-1"
            />
            <span className="text-xs font-mono text-acid w-4 text-right">{topK}</span>
          </div>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-2">
        <button
          onClick={() => canSubmit && onRetrieve(query.trim(), modalities, topK)}
          disabled={!canSubmit}
          className={clsx(
            'flex-1 py-2 rounded-lg text-xs font-medium border transition-all',
            canSubmit
              ? 'border-ink-500 text-ink-200 hover:border-ink-300 hover:text-ink-100 bg-ink-800 hover:bg-ink-700'
              : 'border-ink-700 text-ink-600 cursor-not-allowed'
          )}
        >
          Retrieve only
        </button>
        <button
          onClick={() => canSubmit && onGenerate(query.trim(), modalities, topK)}
          disabled={!canSubmit}
          className={clsx(
            'flex-1 py-2 rounded-lg text-xs font-medium transition-all',
            canSubmit
              ? 'bg-acid text-ink-950 hover:bg-acid-dim'
              : 'bg-ink-700 text-ink-500 cursor-not-allowed'
          )}
        >
          {loading ? 'Generating...' : 'Ask LLM ↵'}
        </button>
      </div>
    </div>
  )
}
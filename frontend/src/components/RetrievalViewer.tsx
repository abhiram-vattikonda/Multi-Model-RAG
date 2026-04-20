import React from 'react'
import { FileVideo, FileAudio, FileImage, FileText, Clock } from 'lucide-react'
import clsx from 'clsx'
import { RetrievedChunk, MediaType } from '../api/client'

interface Props {
  chunks: RetrievedChunk[]
  loading: boolean
}

const ICONS: Record<MediaType, React.ReactNode> = {
  video: <FileVideo size={12} />,
  audio: <FileAudio size={12} />,
  image: <FileImage size={12} />,
  text:  <FileText size={12} />,
}

const COLORS: Record<MediaType, string> = {
  video: 'text-violet-soft bg-violet-soft/10 border-violet-soft/20',
  audio: 'text-acid bg-acid/10 border-acid/20',
  image: 'text-coral-soft bg-coral-soft/10 border-coral-soft/20',
  text:  'text-ink-300 bg-ink-700 border-ink-600',
}

function formatTimestamp(s?: number): string {
  if (s == null) return ''
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}

function inferImageMime(base64?: string, declaredMime?: string): string {
  if (declaredMime) return declaredMime
  if (!base64) return 'image/jpeg'
  if (base64.startsWith('/9j/')) return 'image/jpeg'
  if (base64.startsWith('iVBOR')) return 'image/png'
  if (base64.startsWith('R0lGOD')) return 'image/gif'
  if (base64.startsWith('UklGR')) return 'image/webp'
  return 'image/jpeg'
}

function imageSrc(chunk: RetrievedChunk): string | null {
  if (!chunk.metadata.image_base64) return null
  const mime = inferImageMime(chunk.metadata.image_base64, chunk.metadata.image_mime_type)
  return `data:${mime};base64,${chunk.metadata.image_base64}`
}

export default function RetrievalViewer({ chunks, loading }: Props) {
  if (loading) {
    return (
      <div className="flex flex-col gap-2">
        {[...Array(3)].map((_, i) => (
          <div key={i} className="h-20 rounded-lg bg-ink-800 animate-pulse" />
        ))}
      </div>
    )
  }

  if (chunks.length === 0) {
    return (
      <p className="text-xs text-ink-500 text-center py-8">
        No chunks retrieved yet — run a query above
      </p>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      {chunks.map((chunk, i) => (
        <div
          key={chunk.id}
          className="bg-ink-800 border border-ink-700 rounded-lg p-3 animate-fade-in"
          style={{ animationDelay: `${i * 40}ms` }}
        >
          {/* Header row */}
          <div className="flex items-center gap-2 mb-2">
            <span className="text-ink-500 font-mono text-xs w-4">{i + 1}</span>
            <span className={clsx(
              'flex items-center gap-1 px-1.5 py-0.5 rounded text-xs border',
              COLORS[chunk.metadata.media_type]
            )}>
              {ICONS[chunk.metadata.media_type]}
              {chunk.metadata.media_type}
            </span>
            <span className="text-xs text-ink-400 font-mono truncate">
              {chunk.metadata.source_file}
            </span>
          </div>

          {/* Content */}
          {imageSrc(chunk) && (
            <img
              src={imageSrc(chunk)!}
              alt={chunk.metadata.source_file}
              className="mb-3 max-h-56 w-full rounded-lg object-cover border border-ink-700"
            />
          )}
          <p className="text-xs text-ink-200 leading-relaxed line-clamp-3">{chunk.content}</p>

          {/* Footer: timestamps / page */}
          {(chunk.metadata.timestamp_start != null || chunk.metadata.page_number != null) && (
            <div className="flex items-center gap-1 mt-2 text-xs text-ink-500">
              <Clock size={10} />
              {chunk.metadata.timestamp_start != null ? (
                <span className="font-mono">
                  {formatTimestamp(chunk.metadata.timestamp_start)}
                  {chunk.metadata.timestamp_end != null && ` – ${formatTimestamp(chunk.metadata.timestamp_end)}`}
                </span>
              ) : (
                <span>Page {chunk.metadata.page_number}</span>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

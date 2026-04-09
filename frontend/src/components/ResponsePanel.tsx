import React, { useEffect, useRef } from 'react'
import { Bot, Loader2 } from 'lucide-react'
import clsx from 'clsx'

interface Props {
  answer: string
  streaming: boolean
  model?: string
  provider?: string
}

export default function ResponsePanel({ answer, streaming, model, provider }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [answer])

  if (!answer && !streaming) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-12 text-ink-600">
        <Bot size={28} strokeWidth={1.5} />
        <p className="text-xs">Answer will appear here</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Model badge */}
      {(model || provider) && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-ink-500 font-mono">{provider}</span>
          {model && <>
            <span className="text-ink-700">/</span>
            <span className="text-xs text-ink-400 font-mono">{model}</span>
          </>}
        </div>
      )}

      {/* Answer text */}
      <div className={clsx(
        'text-sm text-ink-100 leading-7 whitespace-pre-wrap',
        streaming && 'animate-fade-in'
      )}>
        {answer}
        {streaming && (
          <span className="inline-flex items-center ml-1 align-middle">
            <span className="w-1.5 h-4 bg-acid rounded-sm animate-pulse-fast" />
          </span>
        )}
      </div>
      <div ref={bottomRef} />
    </div>
  )
}
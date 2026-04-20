import React, { useEffect, useRef } from 'react'
import { Bot, Loader2 } from 'lucide-react'
import clsx from 'clsx'

interface Props {
  answer: string
  streaming: boolean
  model?: string
  provider?: string
}

function normalizeMarkdownish(text: string): string {
  return text
    .replace(/\r\n/g, '\n')
    .replace(/([^\n])(#{1,6}\s+)/g, '$1\n$2')
    .replace(/([^\n])((?:\*|-)\s+)/g, '$1\n$2')
    .replace(/([^\n])(\d+\.\s+)/g, '$1\n$2')
}

function renderInline(text: string): React.ReactNode[] {
  const parts = text.split(/(\*\*.*?\*\*)/g)
  return parts.filter(Boolean).map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**') && part.length >= 4) {
      return <strong key={i} className="font-semibold text-ink-50">{part.slice(2, -2)}</strong>
    }
    return <React.Fragment key={i}>{part}</React.Fragment>
  })
}

function renderMarkdown(answer: string): React.ReactNode[] {
  const lines = normalizeMarkdownish(answer).split('\n')
  const nodes: React.ReactNode[] = []
  let i = 0
  let key = 0

  while (i < lines.length) {
    const line = lines[i].trim()

    if (!line) {
      i += 1
      continue
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/)
    if (heading) {
      const level = Math.min(heading[1].length, 4)
      const text = heading[2].trim()
      const classNameByLevel = {
        1: 'text-xl font-semibold text-ink-50 mt-4 mb-2',
        2: 'text-lg font-semibold text-ink-50 mt-4 mb-2',
        3: 'text-base font-semibold text-ink-100 mt-3 mb-1',
        4: 'text-sm font-semibold text-ink-100 mt-3 mb-1',
      } as const
      const Tag = `h${level}` as keyof JSX.IntrinsicElements
      nodes.push(
        <Tag key={key++} className={classNameByLevel[level as 1 | 2 | 3 | 4]}>
          {renderInline(text)}
        </Tag>
      )
      i += 1
      continue
    }

    if (/^[-*]\s+/.test(line) || /^\d+\.\s+/.test(line)) {
      const items: string[] = []
      while (i < lines.length) {
        const itemLine = lines[i].trim()
        if (!itemLine || (!/^[-*]\s+/.test(itemLine) && !/^\d+\.\s+/.test(itemLine))) break
        items.push(itemLine.replace(/^[-*]\s+/, '').replace(/^\d+\.\s+/, ''))
        i += 1
      }
      nodes.push(
        <ul key={key++} className="list-disc pl-5 space-y-1 text-sm text-ink-100">
          {items.map((item, idx) => (
            <li key={idx}>{renderInline(item)}</li>
          ))}
        </ul>
      )
      continue
    }

    const paragraph: string[] = []
    while (i < lines.length) {
      const paragraphLine = lines[i].trim()
      if (!paragraphLine || /^(#{1,6})\s+/.test(paragraphLine) || /^[-*]\s+/.test(paragraphLine) || /^\d+\.\s+/.test(paragraphLine)) {
        break
      }
      paragraph.push(paragraphLine)
      i += 1
    }

    nodes.push(
      <p key={key++} className="text-sm text-ink-100 leading-7">
        {renderInline(paragraph.join(' '))}
      </p>
    )
  }

  return nodes
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
        'flex flex-col gap-3',
        streaming && 'animate-fade-in'
      )}>
        {renderMarkdown(answer)}
        {streaming && (
          <span className="inline-flex items-center align-middle">
            <span className="w-1.5 h-4 bg-acid rounded-sm animate-pulse-fast" />
          </span>
        )}
      </div>
      <div ref={bottomRef} />
    </div>
  )
}

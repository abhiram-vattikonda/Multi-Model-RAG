import React, { useCallback, useRef, useState } from 'react'
import { Upload, FileVideo, FileAudio, FileImage, FileText, CheckCircle2, XCircle, Loader2 } from 'lucide-react'
import clsx from 'clsx'
import { ingestFile, getTaskStatus, TaskStatus, MediaType } from '../api/client'

interface Job {
  taskId: string
  filename: string
  mediaType: MediaType
  status: TaskStatus['status']
  progress: number
  error?: string
}

const MEDIA_ICONS: Record<MediaType, React.ReactNode> = {
  video: <FileVideo size={14} />,
  audio: <FileAudio size={14} />,
  image: <FileImage size={14} />,
  text:  <FileText size={14} />,
}

const MEDIA_COLORS: Record<MediaType, string> = {
  video:  'text-violet-soft border-violet-soft/30 bg-violet-soft/10',
  audio:  'text-acid border-acid/30 bg-acid/10',
  image:  'text-coral-soft border-coral-soft/30 bg-coral-soft/10',
  text:   'text-ink-200 border-ink-600 bg-ink-800',
}

const ACCEPT = '.txt,.pdf,.md,.mp3,.wav,.m4a,.ogg,.mp4,.mov,.avi,.mkv,.webm,.jpg,.jpeg,.png,.gif,.webp'

export default function UploadPanel() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const pollTask = useCallback((taskId: string) => {
    const interval = setInterval(async () => {
      const status = await getTaskStatus(taskId)
      setJobs(prev => prev.map(j => j.taskId !== taskId ? j : {
        ...j,
        status: status.status,
        progress: status.progress ?? j.progress,
        error: status.error,
      }))
      if (status.status === 'SUCCESS' || status.status === 'FAILURE') {
        clearInterval(interval)
      }
    }, 10000)
  }, [])

  const handleFiles = useCallback(async (files: FileList | File[]) => {
    const arr = Array.from(files)
    for (const file of arr) {
      try {
        const res = await ingestFile(file)
        const job: Job = {
          taskId: res.task_id,
          filename: res.filename,
          mediaType: res.media_type,
          status: 'PENDING',
          progress: 0,
        }
        setJobs(prev => [job, ...prev])
        pollTask(res.task_id)
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e)
        setJobs(prev => [{
          taskId: crypto.randomUUID(),
          filename: file.name,
          mediaType: 'text',
          status: 'FAILURE',
          progress: 0,
          error: msg,
        }, ...prev])
      }
    }
  }, [pollTask])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    handleFiles(e.dataTransfer.files)
  }, [handleFiles])

  return (
    <div className="flex flex-col gap-4">
      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={clsx(
          'relative border border-dashed rounded-xl p-8 flex flex-col items-center justify-center gap-3 cursor-pointer transition-all duration-200',
          dragging
            ? 'border-acid/60 bg-acid/5 border-glow-acid'
            : 'border-ink-600 hover:border-ink-400 hover:bg-ink-800/50'
        )}
      >
        <div className={clsx(
          'w-10 h-10 rounded-full flex items-center justify-center transition-colors',
          dragging ? 'bg-acid/20' : 'bg-ink-700'
        )}>
          <Upload size={18} className={dragging ? 'text-acid' : 'text-ink-300'} />
        </div>
        <div className="text-center">
          <p className="text-sm font-medium text-ink-100">Drop files here or click to upload</p>
          <p className="text-xs text-ink-400 mt-1">Video · Audio · Image · PDF · Text</p>
        </div>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPT}
          className="hidden"
          onChange={e => e.target.files && handleFiles(e.target.files)}
        />
      </div>

      {/* Job list */}
      {jobs.length > 0 && (
        <div className="flex flex-col gap-2">
          {jobs.map(job => (
            <div key={job.taskId} className="bg-ink-800 border border-ink-700 rounded-lg p-3 animate-slide-up">
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  <span className={clsx(
                    'flex items-center gap-1 px-1.5 py-0.5 rounded text-xs border shrink-0',
                    MEDIA_COLORS[job.mediaType]
                  )}>
                    {MEDIA_ICONS[job.mediaType]}
                    {job.mediaType}
                  </span>
                  <span className="text-xs text-ink-200 truncate font-mono">{job.filename}</span>
                </div>
                <StatusIcon status={job.status} />
              </div>

              {/* Progress bar */}
              {(job.status === 'STARTED' || job.status === 'PENDING') && (
                <div className="mt-2 h-1 bg-ink-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-acid rounded-full transition-all duration-500"
                    style={{ width: `${job.status === 'PENDING' ? 5 : job.progress}%` }}
                  />
                </div>
              )}
              {job.error && (
                <p className="mt-1.5 text-xs text-coral-soft/80">{job.error}</p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function StatusIcon({ status }: { status: TaskStatus['status'] }) {
  if (status === 'SUCCESS') return <CheckCircle2 size={14} className="text-acid shrink-0" />
  if (status === 'FAILURE') return <XCircle size={14} className="text-coral-soft shrink-0" />
  return <Loader2 size={14} className="text-ink-400 animate-spin shrink-0" />
}
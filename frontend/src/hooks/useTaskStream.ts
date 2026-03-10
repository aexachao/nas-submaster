import { useEffect, useState, useRef } from 'react'
import type { Task } from '../types'

export function useTaskStream() {
  const [tasks, setTasks] = useState<Task[]>([])
  const [error, setError] = useState<string | null>(null)
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => {
    const connect = () => {
      const es = new EventSource('/api/tasks/stream')
      esRef.current = es

      es.onmessage = (e) => {
        try {
          setTasks(JSON.parse(e.data))
          setError(null)
        } catch {
          // ignore parse error
        }
      }

      es.onerror = () => {
        setError('连接中断，正在重连...')
        es.close()
        // 3 秒后重连
        setTimeout(connect, 3000)
      }
    }

    connect()
    return () => {
      esRef.current?.close()
    }
  }, [])

  return { tasks, error }
}

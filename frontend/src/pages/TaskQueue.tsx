import {
  Button,
  Card,
  CardBody,
  Chip,
  Progress,
  Accordion,
  AccordionItem,
  Spinner,
} from '@heroui/react'
import { useTaskStream } from '../hooks/useTaskStream'
import { tasksApi } from '../api/tasks'
import type { Task, TaskStatus } from '../types'

const STATUS_CONFIG: Record<TaskStatus, { color: 'default' | 'primary' | 'success' | 'danger' | 'warning'; label: string }> = {
  pending:    { color: 'default',  label: '等待中' },
  processing: { color: 'primary',  label: '处理中' },
  completed:  { color: 'success',  label: '完成'   },
  failed:     { color: 'danger',   label: '失败'   },
  cancelled:  { color: 'default',  label: '已取消' },
}

function TaskCard({ task, onRefresh }: { task: Task; onRefresh: () => void }) {
  const { color, label } = STATUS_CONFIG[task.status] ?? { color: 'default', label: task.status }

  const handleCancel = async () => {
    await tasksApi.cancel(task.id)
    onRefresh()
  }
  const handleRetry = async () => {
    await tasksApi.retry(task.id)
    onRefresh()
  }
  const handleDelete = async () => {
    await tasksApi.delete(task.id)
    onRefresh()
  }

  return (
    <Card className="mb-3" shadow="sm">
      <CardBody className="gap-2">
        {/* Header row */}
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <p className="font-semibold text-sm truncate">{task.file_name}</p>
            <p className="text-xs text-default-400 mt-1 truncate">&gt; {task.log}</p>
          </div>
          <div className="flex flex-col items-end gap-1 shrink-0">
            <span className="text-xs text-default-400">{task.created_at ?? ''}</span>
            <Chip size="sm" color={color} variant="flat">{label}</Chip>
          </div>
        </div>

        {/* Progress bar */}
        {task.status === 'processing' && (
          <Progress
            size="sm"
            value={task.progress}
            color="primary"
            className="mt-1"
            label={`${task.progress}%`}
            showValueLabel
          />
        )}

        {/* Log history */}
        {task.log_history && (
          <Accordion isCompact variant="light" className="mt-1 px-0">
            <AccordionItem key="log" title="查看执行日志" classNames={{ title: 'text-xs text-default-500' }}>
              <pre className="text-xs text-default-500 whitespace-pre-wrap font-mono bg-default-100 rounded p-2 max-h-48 overflow-auto">
                {task.log_history}
              </pre>
            </AccordionItem>
          </Accordion>
        )}

        {/* Action buttons */}
        <div className="flex justify-end gap-2 mt-1">
          {task.status === 'processing' && (
            <Button size="sm" variant="flat" color="warning" onPress={handleCancel}>取消</Button>
          )}
          {task.status === 'failed' && (
            <Button size="sm" variant="flat" color="primary" onPress={handleRetry}>重试</Button>
          )}
          {(task.status !== 'processing') && (
            <Button size="sm" variant="flat" color="danger" onPress={handleDelete}>删除</Button>
          )}
        </div>
      </CardBody>
    </Card>
  )
}

export default function TaskQueue() {
  const { tasks, error } = useTaskStream()

  const handleClearCompleted = async () => {
    await tasksApi.clearCompleted()
  }

  // SSE 已提供实时数据，按钮触发 DOM 刷新用 force re-render
  const handleRefresh = () => {
    // SSE 会自动更新，不需要额外操作
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold">任务队列</h2>
        <Button size="sm" variant="flat" onPress={handleClearCompleted}>
          清理记录
        </Button>
      </div>

      {error && (
        <p className="text-sm text-warning mb-3">{error}</p>
      )}

      {tasks.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-default-400">
          <p>队列为空</p>
        </div>
      ) : (
        tasks.map(task => (
          <TaskCard key={task.id} task={task} onRefresh={handleRefresh} />
        ))
      )}
    </div>
  )
}

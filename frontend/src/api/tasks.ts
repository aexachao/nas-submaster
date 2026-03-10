import { api } from './client'
import type { Task } from '../types'

export const tasksApi = {
  list: () => api.get<Task[]>('/tasks'),
  add: (file_paths: string[]) =>
    api.post<{ success: number; failed: unknown[] }>('/tasks', { file_paths }),
  cancel: (id: number) => api.post<{ ok: boolean }>(`/tasks/${id}/cancel`),
  retry: (id: number) => api.post<{ ok: boolean }>(`/tasks/${id}/retry`),
  delete: (id: number) => api.delete<{ ok: boolean }>(`/tasks/${id}`),
  clearCompleted: () => api.delete<{ ok: boolean }>('/tasks/completed'),
}

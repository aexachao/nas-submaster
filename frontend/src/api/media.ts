import { api } from './client'
import type { MediaListResponse } from '../types'

export interface MediaQuery {
  filter?: 'all' | 'has' | 'no'
  page?: number
  page_size?: number
  dirs?: string[]
}

export const mediaApi = {
  list: (q: MediaQuery = {}) => {
    const params = new URLSearchParams()
    if (q.filter) params.set('filter', q.filter)
    if (q.page) params.set('page', String(q.page))
    if (q.page_size) params.set('page_size', String(q.page_size))
    if (q.dirs?.length) q.dirs.forEach(d => params.append('dirs', d))
    return api.get<MediaListResponse>(`/media?${params}`)
  },

  subdirs: () => api.get<{ subdirs: string[] }>('/media/subdirs'),

  scan: (dirs: string[] = []) =>
    api.post<{ updated: number }>('/media/scan', { dirs }),

  rescan: (id: number) =>
    api.post<{ ok: boolean }>(`/media/${id}/rescan`),

  addTasks: (file_paths: string[]) =>
    api.post<{ success: number; failed: { file: string; reason: string }[] }>(
      '/media/add-tasks',
      { file_paths }
    ),
}

import { api } from './client'
import type { Settings, SettingsMeta } from '../types'

export const settingsApi = {
  get: () => api.get<Settings>('/settings'),
  save: (s: Settings) => api.put<{ saved: boolean }>('/settings', s),
  meta: () => api.get<SettingsMeta>('/settings/meta'),
  ollamaModels: (base_url: string) =>
    api.get<{ models: string[] }>(`/settings/ollama-models?base_url=${encodeURIComponent(base_url)}`),
  testConnection: (body: { api_key: string; base_url: string; model_name: string }) =>
    api.post<{ ok: boolean; message: string }>('/settings/test-connection', body),
}

export interface SubtitleInfo {
  path: string
  lang: string
  tag: string
}

export interface MediaFile {
  id: number
  file_path: string
  file_name: string
  file_size: number
  file_size_display: string
  subtitles: SubtitleInfo[]
  has_subtitle: boolean
  has_translated: boolean
  updated_at: string | null
}

export interface MediaListResponse {
  items: MediaFile[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

export type TaskStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled'

export interface Task {
  id: number
  file_path: string
  file_name: string
  status: TaskStatus
  progress: number
  log: string
  log_history: string
  created_at: string | null
  updated_at: string | null
}

export interface ProviderConfig {
  api_key: string
  base_url: string
  model_name: string
}

export interface WhisperConfig {
  model_size: string
  compute_type: string
  device: string
  source_language: string
}

export interface TranslationConfig {
  enabled: boolean
  target_language: string
  max_lines_per_batch: number
}

export interface ExportConfig {
  formats: string[]
}

export interface Settings {
  whisper: WhisperConfig
  translation: TranslationConfig
  export: ExportConfig
  content_type: string
  current_provider: string
  provider_configs: Record<string, ProviderConfig>
}

export interface ContentTypeOption {
  value: string
  label: string
  description: string
}

export interface ExportFormatOption {
  value: string
  label: string
  desc: string
}

export interface SettingsMeta {
  providers: Record<string, { base_url: string; model: string; help: string }>
  content_types: ContentTypeOption[]
  whisper_source_languages: Record<string, string>
  target_languages: Record<string, string>
  whisper_models: string[]
  devices: string[]
  compute_types: string[]
  export_formats: ExportFormatOption[]
}

import { useState, useEffect } from 'react'
import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Button,
  Select,
  SelectItem,
  Input,
  Switch,
  Tabs,
  Tab,
  Checkbox,
  Spinner,
} from '@heroui/react'
import { settingsApi } from '../api/settings'
import type { Settings, SettingsMeta } from '../types'

interface Props {
  isOpen: boolean
  onClose: () => void
}

export default function SettingsModal({ isOpen, onClose }: Props) {
  const [settings, setSettings] = useState<Settings | null>(null)
  const [meta, setMeta] = useState<SettingsMeta | null>(null)
  const [ollamaModels, setOllamaModels] = useState<string[]>([])
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null)

  useEffect(() => {
    if (!isOpen) return
    Promise.all([settingsApi.get(), settingsApi.meta()]).then(([s, m]) => {
      setSettings(s)
      setMeta(m)
      // fetch ollama models if provider is ollama
      if (s.current_provider.includes('Ollama')) {
        const cfg = s.provider_configs[s.current_provider]
        const url = cfg?.base_url || m.providers[s.current_provider]?.base_url || ''
        settingsApi.ollamaModels(url).then(r => setOllamaModels(r.models))
      }
    })
  }, [isOpen])

  const update = <K extends keyof Settings>(key: K, value: Settings[K]) => {
    setSettings(prev => prev ? { ...prev, [key]: value } : prev)
  }

  const updateWhisper = (key: string, value: string) => {
    setSettings(prev => prev ? { ...prev, whisper: { ...prev.whisper, [key]: value } } : prev)
  }

  const updateTranslation = (key: string, value: unknown) => {
    setSettings(prev => prev ? { ...prev, translation: { ...prev.translation, [key]: value } } : prev)
  }

  const updateProvider = () => {
    if (!settings || !meta) return
    const prov = settings.current_provider
    const existing = settings.provider_configs[prov]
    if (!existing) {
      const defaults = meta.providers[prov]
      setSettings(prev => prev ? {
        ...prev,
        provider_configs: {
          ...prev.provider_configs,
          [prov]: { api_key: '', base_url: defaults?.base_url ?? '', model_name: defaults?.model ?? '' }
        }
      } : prev)
    }
  }

  const currentProviderCfg = settings
    ? (settings.provider_configs[settings.current_provider] ?? {
        api_key: '',
        base_url: meta?.providers[settings.current_provider]?.base_url ?? '',
        model_name: meta?.providers[settings.current_provider]?.model ?? '',
      })
    : null

  const setProviderField = (field: 'api_key' | 'base_url' | 'model_name', value: string) => {
    if (!settings) return
    const prov = settings.current_provider
    setSettings(prev => prev ? {
      ...prev,
      provider_configs: {
        ...prev.provider_configs,
        [prov]: { ...(prev.provider_configs[prov] ?? {}), [field]: value }
      }
    } : prev)
  }

  const handleFetchOllamaModels = async () => {
    if (!currentProviderCfg) return
    const r = await settingsApi.ollamaModels(currentProviderCfg.base_url)
    setOllamaModels(r.models)
  }

  const handleTestConnection = async () => {
    if (!currentProviderCfg) return
    setTesting(true)
    setTestResult(null)
    try {
      const r = await settingsApi.testConnection({
        api_key: currentProviderCfg.api_key,
        base_url: currentProviderCfg.base_url,
        model_name: currentProviderCfg.model_name,
      })
      setTestResult(r)
    } finally {
      setTesting(false)
    }
  }

  const handleSave = async () => {
    if (!settings) return
    setSaving(true)
    try {
      await settingsApi.save(settings)
      onClose()
    } finally {
      setSaving(false)
    }
  }

  const toggleExportFormat = (fmt: string, checked: boolean) => {
    if (!settings) return
    const fmts = checked
      ? [...settings.export.formats, fmt]
      : settings.export.formats.filter(f => f !== fmt)
    setSettings(prev => prev ? { ...prev, export: { formats: fmts.length ? fmts : ['srt'] } } : prev)
  }

  if (!settings || !meta) {
    return (
      <Modal isOpen={isOpen} onClose={onClose} size="3xl">
        <ModalContent>
          <ModalBody className="py-12 flex justify-center"><Spinner /></ModalBody>
        </ModalContent>
      </Modal>
    )
  }

  const isOllama = settings.current_provider.includes('Ollama')

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="3xl" scrollBehavior="inside">
      <ModalContent>
        <ModalHeader className="text-xl font-bold">系统设置</ModalHeader>
        <ModalBody>
          <Tabs aria-label="设置标签" size="lg" variant="underlined">

            {/* Tab 1: Whisper 设置 */}
            <Tab key="whisper" title="Whisper 设置">
              <div className="grid grid-cols-2 gap-4 pt-2">
                <Select
                  label="Whisper 模型"
                  selectedKeys={new Set([settings.whisper.model_size])}
                  onSelectionChange={k => updateWhisper('model_size', Array.from(k)[0] as string)}
                >
                  {meta.whisper_models.map(m => <SelectItem key={m}>{m}</SelectItem>)}
                </Select>
                <Select
                  label="运行设备"
                  selectedKeys={new Set([settings.whisper.device])}
                  onSelectionChange={k => updateWhisper('device', Array.from(k)[0] as string)}
                >
                  {meta.devices.map(d => <SelectItem key={d}>{d}</SelectItem>)}
                </Select>
                <Select
                  label="计算精度"
                  selectedKeys={new Set([settings.whisper.compute_type])}
                  onSelectionChange={k => updateWhisper('compute_type', Array.from(k)[0] as string)}
                >
                  {meta.compute_types.map(c => <SelectItem key={c}>{c}</SelectItem>)}
                </Select>
              </div>
            </Tab>

            {/* Tab 2: 识别参数 */}
            <Tab key="params" title="识别参数">
              <div className="flex flex-col gap-4 pt-2">
                <Select
                  label="内容场景（自动优化 VAD）"
                  selectedKeys={new Set([settings.content_type])}
                  onSelectionChange={k => update('content_type', Array.from(k)[0] as string)}
                >
                  {meta.content_types.map(ct => (
                    <SelectItem key={ct.value} description={ct.description}>{ct.label}</SelectItem>
                  ))}
                </Select>
                <Select
                  label="视频原声语言"
                  selectedKeys={new Set([settings.whisper.source_language])}
                  onSelectionChange={k => updateWhisper('source_language', Array.from(k)[0] as string)}
                >
                  {Object.entries(meta.whisper_source_languages).map(([k, v]) => (
                    <SelectItem key={k}>{v}</SelectItem>
                  ))}
                </Select>
              </div>
            </Tab>

            {/* Tab 3: 翻译模型 */}
            <Tab key="model" title="翻译模型">
              <div className="flex flex-col gap-4 pt-2">
                <Select
                  label="AI 服务商"
                  selectedKeys={new Set([settings.current_provider])}
                  onSelectionChange={k => {
                    update('current_provider', Array.from(k)[0] as string)
                    updateProvider()
                  }}
                >
                  {Object.entries(meta.providers).map(([k, v]) => (
                    <SelectItem key={k} description={v.help}>{k}</SelectItem>
                  ))}
                </Select>

                <Input
                  label="Base URL"
                  value={currentProviderCfg?.base_url ?? ''}
                  onValueChange={v => setProviderField('base_url', v)}
                />

                {isOllama ? (
                  <div className="flex gap-2 items-end">
                    <Select
                      label="选择模型"
                      className="flex-1"
                      selectedKeys={currentProviderCfg?.model_name ? new Set([currentProviderCfg.model_name]) : new Set()}
                      onSelectionChange={k => setProviderField('model_name', Array.from(k)[0] as string)}
                    >
                      {ollamaModels.length > 0
                        ? ollamaModels.map(m => <SelectItem key={m}>{m}</SelectItem>)
                        : [<SelectItem key="_empty" isDisabled>未检测到模型</SelectItem>]
                      }
                    </Select>
                    <Button size="sm" variant="flat" onPress={handleFetchOllamaModels}>刷新</Button>
                  </div>
                ) : (
                  <>
                    <Input
                      label="模型名称"
                      value={currentProviderCfg?.model_name ?? ''}
                      onValueChange={v => setProviderField('model_name', v)}
                    />
                    <Input
                      label="API Key"
                      type="password"
                      value={currentProviderCfg?.api_key ?? ''}
                      onValueChange={v => setProviderField('api_key', v)}
                    />
                    <div className="flex items-center gap-3">
                      <Button
                        size="sm"
                        variant="flat"
                        isLoading={testing}
                        onPress={handleTestConnection}
                      >
                        测试连接
                      </Button>
                      {testResult && (
                        <span className={`text-sm ${testResult.ok ? 'text-success' : 'text-danger'}`}>
                          {testResult.message}
                        </span>
                      )}
                    </div>
                  </>
                )}
              </div>
            </Tab>

            {/* Tab 4: 翻译设置 */}
            <Tab key="translation" title="翻译设置">
              <div className="flex flex-col gap-4 pt-2">
                <Switch
                  isSelected={settings.translation.enabled}
                  onValueChange={v => updateTranslation('enabled', v)}
                >
                  启用翻译功能
                </Switch>
                <Select
                  label="目标语言"
                  selectedKeys={new Set([settings.translation.target_language])}
                  onSelectionChange={k => updateTranslation('target_language', Array.from(k)[0] as string)}
                >
                  {Object.entries(meta.target_languages).map(([k, v]) => (
                    <SelectItem key={k}>{v}</SelectItem>
                  ))}
                </Select>
                <Input
                  type="number"
                  label="批处理行数"
                  description="长视频分批翻译，建议 200–500"
                  value={String(settings.translation.max_lines_per_batch)}
                  onValueChange={v => updateTranslation('max_lines_per_batch', Number(v))}
                  min={50}
                  max={5000}
                />
              </div>
            </Tab>

            {/* Tab 5: 字幕格式 */}
            <Tab key="export" title="字幕格式">
              <div className="grid grid-cols-2 gap-3 pt-2">
                {meta.export_formats.map(fmt => (
                  <Checkbox
                    key={fmt.value}
                    isSelected={settings.export.formats.includes(fmt.value)}
                    onValueChange={v => toggleExportFormat(fmt.value, v)}
                    description={fmt.desc}
                  >
                    {fmt.label}
                  </Checkbox>
                ))}
              </div>
            </Tab>

          </Tabs>
        </ModalBody>
        <ModalFooter>
          <Button variant="flat" onPress={onClose}>取消</Button>
          <Button color="primary" isLoading={saving} onPress={handleSave}>保存设置</Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  )
}

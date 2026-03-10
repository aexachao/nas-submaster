import { useState } from 'react'
import { Tabs, Tab, Button } from '@heroui/react'
import MediaLibrary from './pages/MediaLibrary'
import TaskQueue from './pages/TaskQueue'
import SettingsModal from './components/SettingsModal'

export default function App() {
  const [settingsOpen, setSettingsOpen] = useState(false)

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-screen-xl mx-auto px-4 py-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="logo" className="w-10 h-10 object-contain" />
            <h1 className="text-2xl font-bold">NAS 字幕管家</h1>
          </div>
          <Button
            size="sm"
            variant="flat"
            startContent={<span>⚙️</span>}
            onPress={() => setSettingsOpen(true)}
          >
            系统配置
          </Button>
        </div>

        {/* Main tabs */}
        <Tabs aria-label="主导航" size="lg" variant="underlined" fullWidth={false}>
          <Tab key="library" title="媒体库">
            <div className="mt-4">
              <MediaLibrary />
            </div>
          </Tab>
          <Tab key="tasks" title="任务队列">
            <div className="mt-4">
              <TaskQueue />
            </div>
          </Tab>
        </Tabs>
      </div>

      <SettingsModal isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  )
}

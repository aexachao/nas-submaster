import { useState, useEffect, useCallback } from 'react'
import {
  Button,
  Card,
  CardBody,
  Chip,
  Checkbox,
  Select,
  SelectItem,
  SelectSection,
  Pagination,
  Spinner,
  Input,
} from '@heroui/react'
import { mediaApi } from '../api/media'
import type { MediaFile } from '../types'

const FILTER_OPTIONS = [
  { key: 'all', label: '全部' },
  { key: 'has', label: '有字幕' },
  { key: 'no',  label: '无字幕' },
]

const PAGE_SIZE_OPTIONS = [25, 50, 100]

function SubtitleChips({ file }: { file: MediaFile }) {
  if (!file.subtitles.length) {
    return <Chip size="sm" color="danger" variant="flat">无字幕</Chip>
  }
  return (
    <>
      {file.subtitles.map((s, i) => {
        const lang = s.lang.toLowerCase()
        const color = ['zh', 'chs', 'cht'].includes(lang)
          ? 'success'
          : ['en', 'eng'].includes(lang)
            ? 'primary'
            : 'default'
        return (
          <Chip key={i} size="sm" color={color as 'success' | 'primary' | 'default'} variant="flat" className="mr-1">
            {s.tag}
          </Chip>
        )
      })}
    </>
  )
}

export default function MediaLibrary() {
  const [filter, setFilter] = useState<'all' | 'has' | 'no'>('all')
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(50)
  const [subdirs, setSubdirs] = useState<string[]>([])
  const [selectedDirs, setSelectedDirs] = useState<Set<string>>(new Set())
  const [data, setData] = useState<{ items: MediaFile[]; total: number; total_pages: number } | null>(null)
  const [loading, setLoading] = useState(false)
  const [scanning, setScanning] = useState(false)
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set())
  const [selectAll, setSelectAll] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchSubdirs = useCallback(async () => {
    try {
      const res = await mediaApi.subdirs()
      setSubdirs(res.subdirs)
    } catch (e) {
      console.error(e)
    }
  }, [])

  const fetchMedia = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await mediaApi.list({
        filter,
        page,
        page_size: pageSize,
        dirs: Array.from(selectedDirs),
      })
      setData({ items: res.items, total: res.total, total_pages: res.total_pages })
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '加载失败')
    } finally {
      setLoading(false)
    }
  }, [filter, page, pageSize, selectedDirs])

  useEffect(() => { fetchSubdirs() }, [fetchSubdirs])
  useEffect(() => { fetchMedia() }, [fetchMedia])

  // 筛选条件变化时回到第 1 页
  useEffect(() => { setPage(1) }, [filter, selectedDirs, pageSize])

  const handleScan = async () => {
    setScanning(true)
    try {
      await mediaApi.scan(Array.from(selectedDirs))
      await fetchSubdirs()
      await fetchMedia()
    } finally {
      setScanning(false)
    }
  }

  const handleSelectAll = (checked: boolean) => {
    setSelectAll(checked)
    if (checked && data) {
      setSelectedIds(new Set(data.items.map(f => f.id)))
    } else {
      setSelectedIds(new Set())
    }
  }

  const toggleSelect = (id: number, checked: boolean) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (checked) next.add(id)
      else next.delete(id)
      return next
    })
  }

  const handleRescan = async (file: MediaFile) => {
    await mediaApi.rescan(file.id)
    await fetchMedia()
  }

  const handleAddTasks = async () => {
    if (!data) return
    const paths = data.items
      .filter(f => selectedIds.has(f.id))
      .map(f => f.file_path)
    if (!paths.length) return
    await mediaApi.addTasks(paths)
    setSelectedIds(new Set())
    setSelectAll(false)
  }

  const selectedCount = selectedIds.size

  return (
    <div>
      {/* Toolbar */}
      <div className="flex flex-wrap gap-2 items-end mb-4">
        {/* Filter tabs */}
        <div className="flex gap-1">
          {FILTER_OPTIONS.map(opt => (
            <Button
              key={opt.key}
              size="sm"
              variant={filter === opt.key ? 'solid' : 'flat'}
              color={filter === opt.key ? 'primary' : 'default'}
              onPress={() => setFilter(opt.key as 'all' | 'has' | 'no')}
            >
              {opt.label}
            </Button>
          ))}
        </div>

        {/* Dir selector */}
        <div className="flex-1 min-w-48 max-w-sm">
          <Select
            size="sm"
            placeholder="选择目录（留空显示全部）"
            selectionMode="multiple"
            selectedKeys={selectedDirs}
            onSelectionChange={(keys) => setSelectedDirs(new Set(keys as Set<string>))}
            aria-label="目录选择"
          >
            {subdirs.map(d => (
              <SelectItem key={d}>{d}</SelectItem>
            ))}
          </Select>
        </div>

        {/* Scan */}
        <Button
          size="sm"
          variant="flat"
          isLoading={scanning}
          onPress={handleScan}
        >
          {selectedDirs.size > 0 ? `扫描 (${selectedDirs.size})` : '扫描全部'}
        </Button>

        {/* Start */}
        <Button
          size="sm"
          color="primary"
          isDisabled={selectedCount === 0}
          onPress={handleAddTasks}
        >
          {selectedCount > 0 ? `处理 (${selectedCount})` : '开始处理'}
        </Button>
      </div>

      {/* Stats */}
      <p className="text-xs text-default-400 mb-3">
        {data ? `共 ${data.total} 个文件${selectedCount > 0 ? `，已选 ${selectedCount} 个` : ''}` : '加载中...'}
      </p>

      {error && (
        <Card className="mb-3 border border-danger-200">
          <CardBody><p className="text-sm text-danger">{error}</p></CardBody>
        </Card>
      )}

      {/* Select all */}
      {data && data.items.length > 0 && (
        <div className="mb-2">
          <Checkbox
            size="sm"
            isSelected={selectAll}
            onValueChange={handleSelectAll}
          >
            全选（当前页）
          </Checkbox>
        </div>
      )}

      {/* File list */}
      {loading ? (
        <div className="flex justify-center py-16"><Spinner /></div>
      ) : data?.items.length === 0 ? (
        <div className="flex justify-center py-16 text-default-400">
          <p>{selectedDirs.size > 0 ? '选中目录下暂无文件' : '暂无文件，请先扫描媒体库'}</p>
        </div>
      ) : (
        data?.items.map(file => (
          <Card key={file.id} className="mb-2" shadow="sm">
            <CardBody>
              <div className="flex items-center gap-3">
                <Checkbox
                  size="sm"
                  isSelected={selectedIds.has(file.id)}
                  onValueChange={(v) => toggleSelect(file.id, v)}
                  aria-label={file.file_name}
                />
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-sm truncate">{file.file_name}</p>
                  <p className="text-xs text-default-400 font-mono truncate mt-0.5">{file.file_path}</p>
                  <div className="mt-1.5"><SubtitleChips file={file} /></div>
                </div>
                <div className="flex flex-col items-end gap-1 shrink-0">
                  <span className="text-xs text-default-400">{file.file_size_display}</span>
                  <Button
                    size="sm"
                    variant="flat"
                    isIconOnly
                    aria-label="重新扫描"
                    onPress={() => handleRescan(file)}
                  >
                    ↻
                  </Button>
                </div>
              </div>
            </CardBody>
          </Card>
        ))
      )}

      {/* Pagination */}
      {data && data.total_pages > 1 && (
        <div className="flex justify-between items-center mt-4">
          <Pagination
            total={data.total_pages}
            page={page}
            onChange={setPage}
            size="sm"
            showControls
          />
          <Select
            size="sm"
            className="w-24"
            selectedKeys={new Set([String(pageSize)])}
            onSelectionChange={(keys) => {
              const v = Number(Array.from(keys)[0])
              if (!isNaN(v)) setPageSize(v)
            }}
            aria-label="每页数量"
          >
            {PAGE_SIZE_OPTIONS.map(n => (
              <SelectItem key={String(n)}>{String(n)} 条</SelectItem>
            ))}
          </Select>
        </div>
      )}
    </div>
  )
}

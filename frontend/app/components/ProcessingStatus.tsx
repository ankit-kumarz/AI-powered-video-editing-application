'use client'

import { useState, useEffect, useCallback } from 'react'

interface ProcessingStatusProps {
  status: any
  uploadId: string | null
}

export default function ProcessingStatus({ status, uploadId }: ProcessingStatusProps) {
  const [downloads, setDownloads] = useState<any[]>([])
  const [isLoadingDownloads, setIsLoadingDownloads] = useState(false)

  const fetchDownloads = useCallback(async () => {
    if (!uploadId) return

    setIsLoadingDownloads(true)
    try {
      const response = await fetch(`http://localhost:8000/downloads/${uploadId}`)
      if (response.ok) {
        const result = await response.json()
        setDownloads(result.files || [])
      }
    } catch (error) {
      console.error('Error fetching downloads:', error)
    } finally {
      setIsLoadingDownloads(false)
    }
  }, [uploadId])

  useEffect(() => {
    if (status?.status === 'completed' && uploadId) {
      fetchDownloads()
    }
  }, [status?.status, uploadId, fetchDownloads])

  const handleDownload = async (filename: string) => {
    try {
      const response = await fetch(`http://localhost:8000/download/${uploadId}`)
      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = filename
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
      }
    } catch (error) {
      console.error('Download error:', error)
      alert('Download failed. Please try again.')
    }
  }

  const handleCleanup = async () => {
    if (!uploadId) return

    try {
      const response = await fetch(`http://localhost:8000/cleanup/${uploadId}`, {
        method: 'DELETE'
      })

      if (response.ok) {
        alert('Files cleaned up successfully')
        // Reset UI after cleanup so user re-uploads before processing again
        if (typeof window !== 'undefined') {
          window.location.reload()
        }
      }
    } catch (error) {
      console.error('Cleanup error:', error)
      alert('Cleanup failed. Please try again.')
    }
  }

  if (!status) return null

  return (
    <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-6 shadow-xl">
      <h2 className="text-xl font-semibold neon-heading mb-4">Processing Status</h2>

      <div className="space-y-4">
        {/* Status Overview */}
        <div className="flex items-center justify-between p-4 rounded-lg bg-white/3 border border-white/6">
          <div className="flex items-center space-x-3">
            <span className={`text-2xl drop-shadow-[0_0_16px_rgba(99,102,241,0.22)] ${status.status === 'completed' ? 'text-green-400' : status.status === 'processing' ? 'text-yellow-400' : status.status === 'error' ? 'text-red-400' : 'text-indigo-300'}`}>
              {status.status === 'completed' ? '✅' : status.status === 'processing' ? '⏳' : status.status === 'error' ? '❌' : '📹'}
            </span>
            <div>
              <p className="font-medium text-gray-100">Status: <span className="font-semibold text-gray-100">{status.status}</span></p>
              {status.filename && (
                <p className="text-sm text-gray-400">{status.filename}</p>
              )}
            </div>
          </div>

          {status.progress !== undefined && (
            <div className="text-right">
              <p className="text-2xl font-bold text-indigo-300">{status.progress}%</p>
              <p className="text-sm text-gray-400">Complete</p>
            </div>
          )}
        </div>

        {/* Progress Bar */}
        {status.progress !== undefined && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm text-gray-400">
              <span>Processing Progress</span>
              <span>{status.progress}%</span>
            </div>
            <div className="w-full bg-white/6 rounded-full h-3">
              <div
                className="h-3 rounded-full transition-all duration-500 ease-out bg-gradient-to-r from-purple-500 via-indigo-500 to-indigo-300"
                style={{ width: `${status.progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Error Display */}
        {status.error && (
          <div className="p-4 bg-red-900/8 border border-red-600/20 rounded-lg">
            <div className="flex items-center space-x-3">
              <span className="text-2xl text-red-400 drop-shadow-[0_0_12px_rgba(255,84,84,0.14)]">❌</span>
              <div>
                <p className="font-medium text-red-300">Processing Error</p>
                <p className="text-sm text-red-200">{status.error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Voice Translation Results */}
        {status.status === 'completed' && status.target_language && (
          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center space-x-2 mb-3">
              <span className="text-blue-600">🌍</span>
              <h3 className="font-medium text-blue-800">Translation Results</h3>
            </div>
            <div className="space-y-2 text-sm">
              <p><strong>Target Language:</strong> {status.target_language}</p>
              {status.original_text && (
                <p><strong>Original Text:</strong> {status.original_text.substring(0, 100)}...</p>
              )}
              {status.translated_text && (
                <p><strong>Translated Text:</strong> {status.translated_text.substring(0, 100)}...</p>
              )}
              {status.subtitle_translated && (
                <p className="text-green-600"><strong>✅ Subtitles translated and burned in</strong></p>
              )}
            </div>
          </div>
        )}

        {/* Downloads Section */}
        {status.status === 'completed' && (
          <div className="space-y-3">
            <h3 className="font-medium text-gray-100">Download Results</h3>

            {isLoadingDownloads ? (
              <div className="text-center py-4">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-indigo-400 mx-auto"></div>
                <p className="text-sm text-gray-400 mt-2">Loading downloads...</p>
              </div>
            ) : downloads.length > 0 ? (
              <div className="space-y-2">
                {downloads.map((file, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-white/3 border border-white/6 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <span className="text-indigo-300">📁</span>
                      <div>
                        <p className="font-medium text-gray-100">{file.filename}</p>
                        <p className="text-xs text-gray-400">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                      </div>
                    </div>
                    <button
                      onClick={() => handleDownload(file.filename)}
                      className="px-4 py-2 bg-gradient-to-r from-purple-500 to-indigo-500 hover:from-purple-600 hover:to-indigo-600 text-white text-sm font-medium rounded-lg transition-colors"
                    >
                      Download
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-4 text-gray-400">
                <p>No processed files found</p>
              </div>
            )}
          </div>
        )}

        {/* Actions */}
        <div className="flex space-x-3 pt-4 border-t border-white/6">
          {status.status === 'completed' && (
            <button
              onClick={fetchDownloads}
              className="flex-1 px-4 py-2 bg-gradient-to-r from-purple-500 to-indigo-500 hover:from-purple-600 hover:to-indigo-600 text-white text-sm font-medium rounded-lg transition-colors"
            >
              Refresh Downloads
            </button>
          )}

          <button
            onClick={handleCleanup}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white text-sm font-medium rounded-lg transition-colors"
          >
            Cleanup Files
          </button>
        </div>
      </div>
    </div>
  )
}

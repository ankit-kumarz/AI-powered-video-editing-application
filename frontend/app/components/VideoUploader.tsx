'use client'

import { useState, useRef } from 'react'
import { motion } from 'framer-motion'

interface VideoUploaderProps {
    onUploadSuccess: (uploadId: string, file: File) => void
    disabled?: boolean
}

export default function VideoUploader({ onUploadSuccess, disabled }: VideoUploaderProps) {
    const [isDragging, setIsDragging] = useState(false)
    const [isUploading, setIsUploading] = useState(false)
    const [uploadProgress, setUploadProgress] = useState(0)
    const fileInputRef = useRef<HTMLInputElement>(null)

    const openFileDialog = () => {
        if (disabled) return
        fileInputRef.current?.click()
    }

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (file) handleFile(file)
    }

    const handleFile = (file: File) => {
        // Simulate upload progress
        setIsUploading(true)
        setUploadProgress(0)
        const interval = setInterval(() => {
            setUploadProgress((p) => {
                const next = Math.min(100, p + Math.floor(Math.random() * 20) + 10)
                if (next >= 100) {
                    clearInterval(interval)
                    setTimeout(() => {
                        setIsUploading(false)
                        onUploadSuccess(`upload_${Date.now()}`, file)
                    }, 300)
                }
                return next
            })
        }, 250)
    }

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault()
        if (!disabled) setIsDragging(true)
    }

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault()
        setIsDragging(false)
    }

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault()
        setIsDragging(false)
        const file = e.dataTransfer.files?.[0]
        if (file) handleFile(file)
    }

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, ease: 'easeOut' }}
            className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl shadow-xl transition-all duration-300 hover:border-purple-500 hover:shadow-[0_0_24px_rgba(168,85,247,0.4)] flex flex-col items-center justify-center mx-auto max-w-xl p-6"
        >
            <h2 className="text-xl font-semibold text-gray-100 mb-4">Upload Video</h2>

            <div
                className={`w-full border-2 border-dashed rounded-2xl p-8 text-center transition-colors duration-200 shadow-lg ${isDragging
                    ? 'border-purple-500 bg-purple-900/10' : 'border-gray-600 bg-black/10'} hover:border-purple-500 hover:shadow-[0_0_24px_rgba(168,85,247,0.4)] ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={!disabled ? openFileDialog : undefined}
            >
                <div className="space-y-4">
                    <div className="text-6xl text-purple-500 drop-shadow-[0_0_8px_rgba(168,85,247,0.7)] animate-pulse">📤</div>

                    <div>
                        <p className="text-lg font-medium text-gray-100">
                            {isUploading ? 'Uploading...' : 'Drop your video here'}
                        </p>
                        <p className="text-sm text-gray-500 mt-1">or click to browse files</p>
                    </div>

                    {isUploading && (
                        <div className="w-full bg-gray-700 rounded-full h-2">
                            <div
                                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${uploadProgress}%` }}
                            />
                        </div>
                    )}

                    <p className="text-xs text-gray-400">Supports MP4, WebM, AVI, MOV and other video formats</p>
                </div>
            </div>

            <input ref={fileInputRef} type="file" accept="video/*" onChange={handleFileSelect} className="hidden" disabled={disabled} />
        </motion.div>
    )
}

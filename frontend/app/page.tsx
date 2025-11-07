'use client'

import { useState, useRef } from 'react'
import VideoUploader from './components/VideoUploader'
import FeatureSelector from './components/FeatureSelector'
import ProcessingStatus from './components/ProcessingStatus'
import VideoPreview from './components/VideoPreview'
import CompilationPreview from './components/CompilationPreview'
import VoiceTranslationSelector from './components/VoiceTranslationSelector'
import VideoCompilationSelector from './components/VideoCompilationSelector'
import ObjectRemovalSelector from './components/ObjectRemovalSelector'
import AIEditingSelector from './components/AIEditingSelector'

interface BoundingBox {
    id: string
    x: number
    y: number
    width: number
    height: number
    label?: string
}

export default function Home() {
    const [mode, setMode] = useState<'single' | 'compilation'>('single')
    const [uploadId, setUploadId] = useState<string | null>(null)
    const [uploadedFile, setUploadedFile] = useState<File | null>(null)
    const [uploadedVideos, setUploadedVideos] = useState<{ id: string, file: File, name: string }[]>([])
    const [selectedVideos, setSelectedVideos] = useState<string[]>([])
    const [selectedFeatures, setSelectedFeatures] = useState<string[]>([])
    const [selectedStyle, setSelectedStyle] = useState<string>('cartoon')
    const [targetLanguage, setTargetLanguage] = useState<string>('es')
    const [voiceType, setVoiceType] = useState<string>('female')
    const [addSubtitles, setAddSubtitles] = useState<boolean>(true)
    const [maxDuration, setMaxDuration] = useState<number>(60)
    const [transitionStyle, setTransitionStyle] = useState<string>('fade')
    const [selectedPreset, setSelectedPreset] = useState<string>('youtube_shorts')
    const [applyEffects, setApplyEffects] = useState<boolean>(false)
    const [effectType, setEffectType] = useState<string>('none')
    const [isProcessing, setIsProcessing] = useState(false)
    const [processingStatus, setProcessingStatus] = useState<any>(null)
    const [objectRemovalBoxes, setObjectRemovalBoxes] = useState<BoundingBox[]>([])

    const handleUploadSuccess = (id: string, file: File) => {
        setUploadId(id)
        setUploadedFile(file)

        // Add to uploaded videos list for compilation
        setUploadedVideos(prev => [...prev, { id, file, name: file.name }])

        setIsProcessing(false)
        setProcessingStatus(null)
        setObjectRemovalBoxes([]) // Reset object removal boxes
    }

    const handleModeChange = (newMode: 'single' | 'compilation') => {
        setMode(newMode)
        // Clear selections when switching modes
        setSelectedFeatures([])
        setSelectedVideos([])
        setProcessingStatus(null)
        setObjectRemovalBoxes([])
    }

    const handleFeatureToggle = (feature: string) => {
        setSelectedFeatures(prev =>
            prev.includes(feature)
                ? prev.filter(f => f !== feature)
                : [...prev, feature]
        )

        // Clear object removal boxes when feature is deselected
        if (feature === 'object-remove' && selectedFeatures.includes(feature)) {
            setObjectRemovalBoxes([])
        }
    }

    const handleProcess = async () => {
        // For compilation mode, need multiple videos
        if (mode === 'compilation') {
            if (selectedVideos.length < 2) {
                alert('Please select at least 2 videos for compilation')
                return
            }
        } else {
            // For single mode, need single video and features
            if (!uploadId) return
            if (selectedFeatures.length === 0) return

            // Check if object removal is selected but no boxes are marked
            if (selectedFeatures.includes('object-remove') && objectRemovalBoxes.length === 0) {
                alert('Please mark objects to remove before processing')
                return
            }
        }

        setIsProcessing(true)
        setProcessingStatus({ status: 'starting', progress: 0 })

        try {
            if (mode === 'compilation') {
                // Single compilation request
                const url = `http://localhost:8000/process/video-compilation?upload_ids=${selectedVideos.join(',')}&max_duration=${maxDuration}&transition_style=${transitionStyle}&preset=${selectedPreset}&apply_effects=${applyEffects}&effect_type=${effectType}`

                const response = await fetch(url, {
                    method: 'POST',
                })

                if (!response.ok) {
                    throw new Error('Failed to process video compilation')
                }
            } else {
                // Process each selected feature individually for single mode
                for (const feature of selectedFeatures) {
                    let url = `http://localhost:8000/process/${feature}?upload_id=${uploadId}`

                    // Add style parameter for style filters
                    if (feature === 'style') {
                        url += `&style=${selectedStyle}`
                    }

                    // Add voice translation parameters
                    if (feature === 'voice-translate') {
                        url += `&target_language=${targetLanguage}&voice_type=${voiceType}&add_subtitles=${addSubtitles}`
                    }

                    // Add object removal parameters
                    if (feature === 'object-remove') {
                        const boxesParam = objectRemovalBoxes.map(box =>
                            `${box.x},${box.y},${box.x + box.width},${box.y + box.height}`
                        ).join(';')
                        url += `&bounding_boxes=${encodeURIComponent(boxesParam)}`
                    }

                    const response = await fetch(url, {
                        method: 'POST',
                    })

                    if (!response.ok) {
                        throw new Error(`Failed to process ${feature}`)
                    }
                }
            }
        } catch (error) {
            console.error('Error processing:', error)
            const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred'
            setProcessingStatus({ status: 'error', error: errorMessage })
            setIsProcessing(false)
            return
        }

        // Start polling for status
        const statusId = mode === 'compilation' ? selectedVideos[0] : uploadId
        pollProcessingStatus(statusId!)
    }

    const pollProcessingStatus = async (id: string) => {
        const interval = setInterval(async () => {
            try {
                const response = await fetch(`http://localhost:8000/status/${id}`)
                if (response.ok) {
                    const status = await response.json()
                    setProcessingStatus(status)

                    if (status.status === 'completed' || status.status === 'error') {
                        clearInterval(interval)
                        setIsProcessing(false)
                    }
                }
            } catch (error) {
                console.error('Error polling status:', error)
            }
        }, 2000)
    }

    return (
        <div className="min-h-screen w-full bg-gradient-to-tr from-[#0b0c10] via-[#121212] to-[#23272f] relative flex flex-col justify-center items-center">
            {/* Glowing corner accents */}
            <div className="absolute top-0 left-0 w-48 h-48 bg-gradient-to-tr from-purple-700/20 to-transparent rounded-full blur-2xl pointer-events-none"></div>
            <div className="absolute bottom-0 right-0 w-48 h-48 bg-gradient-to-tr from-blue-700/20 to-transparent rounded-full blur-2xl pointer-events-none"></div>
            <div className="container mx-auto px-4 py-12">
                <section className="text-center mb-12">
                    <h1 className="neon-heading mb-4">AI Video Editor MVP</h1>
                    <p className="text-lg text-gray-400 mb-8">Transform your videos with AI-powered editing features</p>
                    {/* Mode Toggle */}
                    <div className="flex justify-center mb-8">
                        <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl shadow-xl p-2">
                            <div className="flex gap-2">
                                <button
                                    onClick={() => handleModeChange('single')}
                                    className={`px-6 py-3 rounded-xl font-semibold transition-all duration-200 bg-gradient-to-r from-purple-500 to-blue-500 text-white shadow-md ${mode === 'single' ? 'scale-105 shadow-[0_0_15px_rgba(168,85,247,0.7)]' : 'opacity-70 hover:scale-105 hover:shadow-[0_0_15px_rgba(168,85,247,0.7)]'}`}
                                >
                                    🎬 Single Video Editing
                                </button>
                                <button
                                    onClick={() => handleModeChange('compilation')}
                                    className={`px-6 py-3 rounded-xl font-semibold transition-all duration-200 bg-gradient-to-r from-teal-500 to-violet-500 text-white shadow-md ${mode === 'compilation' ? 'scale-105 shadow-[0_0_15px_rgba(124,58,237,0.7)]' : 'opacity-70 hover:scale-105 hover:shadow-[0_0_15px_rgba(124,58,237,0.7)]'}`}
                                >
                                    📱 Social Media Compilation
                                </button>
                            </div>
                        </div>
                    </div>
                    {/* Mode Description */}
                    <div className="text-center">
                        {mode === 'single' ? (
                            <p className="text-sm text-gray-400">Upload a single video and apply AI editing features</p>
                        ) : (
                            <p className="text-sm text-gray-400">Upload up to 5 videos and create AI-powered social media compilations</p>
                        )}
                    </div>
                </section>
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
                    {/* Left Column - Upload & Preview */}
                    <div className="lg:col-span-2 space-y-8">
                        <VideoUploader onUploadSuccess={handleUploadSuccess} disabled={isProcessing} />
                        {mode === 'single' && uploadedFile && (
                            <VideoPreview file={uploadedFile} uploadId={uploadId} processingStatus={processingStatus} />
                        )}
                        {mode === 'compilation' && selectedVideos.length > 0 && (
                            <CompilationPreview uploadId={selectedVideos[0]} processingStatus={processingStatus} selectedVideos={selectedVideos} />
                        )}
                        {mode === 'compilation' && selectedVideos.length === 0 && uploadedVideos.length > 0 && (
                            <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl shadow-xl p-6">
                                <h2 className="text-xl font-semibold neon-heading mb-4 flex items-center">
                                    <span className="mr-2">📱</span> Compilation Setup
                                </h2>
                                <div className="bg-gradient-to-r from-purple-900/10 to-blue-900/10 border border-purple-500 rounded-xl p-4">
                                    <p className="text-gray-400 mb-3">You have uploaded {uploadedVideos.length} video(s). Select videos from the right panel to create your compilation.</p>
                                    <div className="grid grid-cols-2 gap-4 text-sm">
                                        <div>
                                            <span className="font-medium text-gray-400">Available Videos:</span>
                                            <p className="text-gray-100">{uploadedVideos.length}</p>
                                        </div>
                                        <div>
                                            <span className="font-medium text-gray-400">Selected Videos:</span>
                                            <p className="text-gray-100">0</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                        {/* Object Removal Selector - Only show when feature is selected */}
                        {mode === 'single' && selectedFeatures.includes('object-remove') && uploadedFile && (
                            <ObjectRemovalSelector videoFile={uploadedFile} onBoundingBoxesChange={setObjectRemovalBoxes} disabled={isProcessing} />
                        )}
                        {/* AI Editing Selector - Only show when feature is selected */}
                        {mode === 'single' && selectedFeatures.includes('ai-editing-suggestions') && uploadedFile && (
                            <AIEditingSelector uploadId={uploadId!} videoFile={uploadedFile} />
                        )}
                    </div>
                    {/* Right Column - Features & Controls */}
                    <div className="space-y-8">
                        <FeatureSelector selectedFeatures={selectedFeatures} onFeatureToggle={handleFeatureToggle} selectedStyle={selectedStyle} onStyleChange={setSelectedStyle} disabled={isProcessing} mode={mode} />
                        {/* Voice Translation Settings - Only in Single Mode */}
                        {mode === 'single' && selectedFeatures.includes('voice-translate') && (
                            <VoiceTranslationSelector onLanguageChange={setTargetLanguage} onVoiceTypeChange={setVoiceType} onSubtitleChange={setAddSubtitles} disabled={isProcessing} />
                        )}
                        {/* Compilation Settings - Only in Compilation Mode */}
                        {mode === 'compilation' && (
                            <VideoCompilationSelector onUploadIdsChange={setSelectedVideos} onMaxDurationChange={setMaxDuration} onTransitionStyleChange={setTransitionStyle} onPresetChange={setSelectedPreset} onApplyEffectsChange={setApplyEffects} onEffectTypeChange={setEffectType} disabled={isProcessing} availableUploadIds={uploadedVideos.map(v => v.id)} />
                        )}
                        {((mode === 'single' && uploadId && selectedFeatures.length > 0) || (mode === 'compilation' && selectedVideos.length >= 2 && selectedVideos.length <= 5)) && (
                            <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl shadow-xl p-6">
                                <button onClick={handleProcess} disabled={isProcessing} className="w-full bg-gradient-to-r from-purple-500 to-blue-500 text-white font-semibold py-3 px-6 rounded-xl transition-transform duration-200 hover:scale-105 hover:shadow-[0_0_15px_rgba(168,85,247,0.7)]">
                                    {isProcessing ? 'Processing...' : 'Process Video'}
                                </button>
                            </div>
                        )}
                        {processingStatus && (
                            <ProcessingStatus status={processingStatus} uploadId={uploadId} />
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}

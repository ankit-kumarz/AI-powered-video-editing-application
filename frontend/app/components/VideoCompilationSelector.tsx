'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

interface VideoCompilationSelectorProps {
    onUploadIdsChange: (uploadIds: string[]) => void
    onMaxDurationChange: (duration: number) => void
    onTransitionStyleChange: (style: string) => void
    onPresetChange: (preset: string) => void
    onApplyEffectsChange: (apply: boolean) => void
    onEffectTypeChange: (effectType: string) => void
    disabled?: boolean
    availableUploadIds?: string[]
}

interface CompilationPreset {
    id: string
    name: string
    max_duration: number
    description: string
    aspect_ratio: string
    features: string[]
    platform: string
    icon: string
}

interface TransitionStyle {
    id: string
    name: string
    description: string
    icon: string
}

interface PostCompilationEffect {
    id: string
    name: string
    description: string
    icon: string
    category: string
}

const COMPILATION_PRESETS: CompilationPreset[] = [
    {
        id: "youtube_shorts",
        name: "YouTube Shorts",
        max_duration: 60,
        description: "Vertical 9:16 format, perfect for mobile viewing",
        aspect_ratio: "9:16",
        features: ["Vertical format", "Mobile-optimized", "Trending content"],
        platform: "YouTube",
        icon: "📱"
    },
    {
        id: "youtube_standard",
        name: "YouTube Standard",
        max_duration: 300,
        description: "Traditional 16:9 format for desktop viewing",
        aspect_ratio: "16:9",
        features: ["Desktop optimized", "Longer content", "Detailed editing"],
        platform: "YouTube",
        icon: "📺"
    },
    {
        id: "instagram_reels",
        name: "Instagram Reels",
        max_duration: 90,
        description: "Vertical format with music and effects support",
        aspect_ratio: "9:16",
        features: ["Music integration", "Effects ready", "Story-friendly"],
        platform: "Instagram",
        icon: "📸"
    },
    {
        id: "instagram_stories",
        name: "Instagram Stories",
        max_duration: 15,
        description: "Quick 15-second stories for daily content",
        aspect_ratio: "9:16",
        features: ["Quick content", "Daily updates", "Engagement focused"],
        platform: "Instagram",
        icon: "📱"
    },
    {
        id: "tiktok",
        name: "TikTok",
        max_duration: 60,
        description: "Trending vertical format with viral potential",
        aspect_ratio: "9:16",
        features: ["Viral potential", "Trending format", "Music sync"],
        platform: "TikTok",
        icon: "🎵"
    },
    {
        id: "facebook_reels",
        name: "Facebook Reels",
        max_duration: 60,
        description: "Facebook's short-form video format",
        aspect_ratio: "9:16",
        features: ["Facebook optimized", "Community focused", "Shareable"],
        platform: "Facebook",
        icon: "👥"
    },
    {
        id: "twitter_video",
        name: "Twitter Video",
        max_duration: 140,
        description: "Twitter's video format with character limit awareness",
        aspect_ratio: "16:9",
        features: ["Twitter optimized", "Quick consumption", "Thread-friendly"],
        platform: "Twitter",
        icon: "🐦"
    },
    {
        id: "linkedin_video",
        name: "LinkedIn Video",
        max_duration: 600,
        description: "Professional format for business content",
        aspect_ratio: "16:9",
        features: ["Professional", "Business focused", "Educational"],
        platform: "LinkedIn",
        icon: "💼"
    },
    {
        id: "snapchat_spotlight",
        name: "Snapchat Spotlight",
        max_duration: 60,
        description: "Snapchat's vertical video format",
        aspect_ratio: "9:16",
        features: ["Snapchat native", "Youth audience", "Creative effects"],
        platform: "Snapchat",
        icon: "👻"
    },
    {
        id: "pinterest_video",
        name: "Pinterest Video",
        max_duration: 60,
        description: "Pinterest's visual discovery format",
        aspect_ratio: "2:3",
        features: ["Visual discovery", "Inspiration focused", "Pin-optimized"],
        platform: "Pinterest",
        icon: "📌"
    },
    {
        id: "custom",
        name: "Custom Format",
        max_duration: 300,
        description: "Customize your own format and settings",
        aspect_ratio: "16:9",
        features: ["Fully customizable", "Flexible duration", "Any platform"],
        platform: "Custom",
        icon: "⚙️"
    }
]

const TRANSITION_STYLES: TransitionStyle[] = [
    {
        id: "fade",
        name: "Fade In/Out",
        description: "Smooth fade transitions between clips",
        icon: "🌅"
    },
    {
        id: "crossfade",
        name: "Crossfade",
        description: "Overlapping fade transitions for seamless flow",
        icon: "🔄"
    },
    {
        id: "slide",
        name: "Slide",
        description: "Slide transitions between clips",
        icon: "➡️"
    },
    {
        id: "zoom",
        name: "Zoom",
        description: "Zoom in/out transitions for dynamic effect",
        icon: "🔍"
    },
    {
        id: "wipe",
        name: "Wipe",
        description: "Wipe transitions for modern look",
        icon: "🧹"
    },
    {
        id: "dissolve",
        name: "Dissolve",
        description: "Dissolve transitions for artistic effect",
        icon: "✨"
    }
]

const POST_COMPILATION_EFFECTS: PostCompilationEffect[] = [
    {
        id: "none",
        name: "No Effect",
        description: "Keep original video without any effects",
        icon: "🎬",
        category: "none"
    },
    {
        id: "vintage",
        name: "Vintage",
        description: "Classic film look with warm tones and grain",
        icon: "📷",
        category: "retro"
    },
    {
        id: "cinematic",
        name: "Cinematic",
        description: "Movie-like appearance with enhanced contrast",
        icon: "🎭",
        category: "professional"
    },
    {
        id: "warm",
        name: "Warm",
        description: "Cozy, golden-hour lighting effect",
        icon: "🌅",
        category: "color"
    },
    {
        id: "cool",
        name: "Cool",
        description: "Blue-tinted, modern aesthetic",
        icon: "❄️",
        category: "color"
    },
    {
        id: "dramatic",
        name: "Dramatic",
        description: "High contrast, bold colors for impact",
        icon: "⚡",
        category: "professional"
    },
    {
        id: "bright",
        name: "Bright",
        description: "Enhanced brightness and vibrant colors",
        icon: "☀️",
        category: "color"
    },
    {
        id: "moody",
        name: "Moody",
        description: "Dark, atmospheric mood with reduced brightness",
        icon: "🌙",
        category: "atmospheric"
    },
    {
        id: "vibrant",
        name: "Vibrant",
        description: "Saturated, eye-catching colors",
        icon: "🌈",
        category: "color"
    },
    {
        id: "monochrome",
        name: "Monochrome",
        description: "Classic black and white effect",
        icon: "⚫",
        category: "retro"
    },
    {
        id: "sepia",
        name: "Sepia",
        description: "Antique brown-tinted effect",
        icon: "📜",
        category: "retro"
    }
]

export default function VideoCompilationSelector({
    onUploadIdsChange,
    onMaxDurationChange,
    onTransitionStyleChange,
    onPresetChange,
    onApplyEffectsChange,
    onEffectTypeChange,
    disabled,
    availableUploadIds = []
}: VideoCompilationSelectorProps) {
    const [selectedUploadIds, setSelectedUploadIds] = useState<string[]>([])
    const [selectedPreset, setSelectedPreset] = useState<string>("youtube_shorts")
    const [selectedTransitionStyle, setSelectedTransitionStyle] = useState<string>("fade")
    const [customDuration, setCustomDuration] = useState<number>(60)
    const [applyEffects, setApplyEffects] = useState<boolean>(false)
    const [selectedEffect, setSelectedEffect] = useState<string>("none")

    useEffect(() => {
        onUploadIdsChange(selectedUploadIds)
    }, [selectedUploadIds, onUploadIdsChange])

    useEffect(() => {
        const preset = COMPILATION_PRESETS.find(p => p.id === selectedPreset)
        if (preset) {
            setCustomDuration(preset.max_duration)
            onMaxDurationChange(preset.max_duration)
            onPresetChange(selectedPreset)
        }
    }, [selectedPreset, onMaxDurationChange, onPresetChange])

    useEffect(() => {
        onTransitionStyleChange(selectedTransitionStyle)
    }, [selectedTransitionStyle, onTransitionStyleChange])

    useEffect(() => {
        onApplyEffectsChange(applyEffects)
    }, [applyEffects, onApplyEffectsChange])

    useEffect(() => {
        onEffectTypeChange(selectedEffect)
    }, [selectedEffect, onEffectTypeChange])

    const handleUploadIdToggle = (uploadId: string) => {
        if (selectedUploadIds.includes(uploadId)) {
            setSelectedUploadIds(selectedUploadIds.filter(id => id !== uploadId))
        } else {
            if (selectedUploadIds.length < 5) {
                setSelectedUploadIds([...selectedUploadIds, uploadId])
            }
        }
    }

    const handleCustomDurationChange = (duration: number) => {
        setCustomDuration(duration)
        onMaxDurationChange(duration)
    }

    const selectedPresetData = COMPILATION_PRESETS.find(p => p.id === selectedPreset)

    return (
        <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
            className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl shadow-xl p-6"
        >
            <h3 className="text-xl font-semibold neon-heading mb-4 flex items-center">
                <span className="mr-3 text-xl">🎬</span>
                Video Compilation Settings
            </h3>

            <div className="space-y-6">
                {/* Video Selection */}
                <div>
                    <label className="block text-sm font-medium text-purple-400 uppercase tracking-wide mb-3">
                        Select Videos
                    </label>

                    <div className="space-y-3">
                        {availableUploadIds.length === 0 ? (
                            <div className="p-3 rounded-xl border border-white/10 bg-transparent text-sm text-gray-400 italic">Upload videos first to create a compilation</div>
                        ) : (
                            availableUploadIds.map((uploadId) => (
                                <label
                                    key={uploadId}
                                    className={`flex items-start space-x-3 p-3 rounded-xl border cursor-pointer transition-all duration-200 backdrop-blur-md ${selectedUploadIds.includes(uploadId)
                                        ? 'border-purple-500 bg-purple-900/10 shadow-[0_0_24px_rgba(168,85,247,0.25)]'
                                        : 'border-white/10 hover:border-purple-500'}`}
                                >
                                    <input
                                        type="checkbox"
                                        checked={selectedUploadIds.includes(uploadId)}
                                        onChange={() => handleUploadIdToggle(uploadId)}
                                        disabled={disabled}
                                        className="mt-1 h-4 w-4 accent-purple-500 focus:ring-purple-500 border-purple-500 rounded"
                                    />

                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center space-x-2">
                                            <span className="text-sm font-medium text-gray-100">Video {uploadId.slice(0, 8)}...</span>
                                        </div>
                                        <p className="text-xs text-gray-400 mt-1">Selected: {selectedUploadIds.length}/5 videos</p>
                                    </div>
                                </label>
                            ))
                        )}
                    </div>
                </div>

                {/* Social Media Platform Preset */}
                <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                        Choose Social Media Platform
                    </label>
                    <div className="space-y-3 max-h-64 overflow-y-auto pr-2">
                        {COMPILATION_PRESETS.map((preset) => (
                            <label
                                key={preset.id}
                                className={`flex items-start space-x-3 p-3 rounded-xl border cursor-pointer transition-all duration-200 backdrop-blur-md ${selectedPreset === preset.id
                                    ? 'border-purple-500 bg-purple-900/10 shadow-[0_0_24px_rgba(168,85,247,0.25)]'
                                    : 'border-white/10 hover:border-purple-500'}`}
                            >
                                <input
                                    type="radio"
                                    name="preset"
                                    value={preset.id}
                                    checked={selectedPreset === preset.id}
                                    onChange={(e) => setSelectedPreset(e.target.value)}
                                    disabled={disabled}
                                    className="mt-1 h-4 w-4 accent-purple-500 focus:ring-purple-500 border-purple-500 rounded"
                                />
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between mb-1">
                                        <div className="flex items-center space-x-2">
                                            <span className="text-xl text-purple-400 drop-shadow-[0_0_8px_rgba(168,85,247,0.7)]">{preset.icon}</span>
                                            <span className="text-sm font-medium text-gray-100">{preset.name}</span>
                                        </div>
                                        <span className="text-xs bg-white/3 text-gray-200 px-2 py-1 rounded-full border border-white/6">{preset.platform}</span>
                                    </div>
                                    <p className="text-xs text-gray-400 mb-2">{preset.description}</p>
                                    <div className="flex items-center space-x-3 text-xs text-gray-400">
                                        <span>📐 {preset.aspect_ratio}</span>
                                        <span>⏱️ {preset.max_duration}s</span>
                                    </div>
                                </div>
                            </label>
                        ))}
                    </div>
                </div>



                {/* Custom Duration */}
                <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                        Max Duration (seconds)
                    </label>
                    <input
                        type="number"
                        min="15"
                        max="600"
                        value={customDuration}
                        onChange={(e) => handleCustomDurationChange(parseInt(e.target.value) || 60)}
                        disabled={disabled}
                        className="w-full px-3 py-2 border border-white/6 rounded-md bg-[#0b0c10]/40 text-gray-200 focus:outline-none focus:ring-2 focus:ring-indigo-400"
                    />
                </div>

                {/* Transition Style */}
                <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                        Transition Style
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                        {TRANSITION_STYLES.map((style) => (
                            <label key={style.id} className="flex items-center p-2 border border-white/6 rounded-md hover:bg-white/2 cursor-pointer">
                                <input
                                    type="radio"
                                    name="transitionStyle"
                                    value={style.id}
                                    checked={selectedTransitionStyle === style.id}
                                    onChange={(e) => setSelectedTransitionStyle(e.target.value)}
                                    disabled={disabled}
                                    className="h-4 w-4 text-indigo-300 focus:ring-indigo-300 border-gray-600 disabled:opacity-50"
                                />
                                <span className="ml-2 text-sm text-gray-300 flex items-center">
                                    <span className="mr-1">{style.icon}</span>
                                    {style.name}
                                </span>
                            </label>
                        ))}
                    </div>
                    <p className="text-xs text-gray-400 mt-1">
                        {TRANSITION_STYLES.find(s => s.id === selectedTransitionStyle)?.description}
                    </p>
                </div>

                {/* Post-Compilation Effects */}
                <div>
                    <div className="flex items-center justify-between mb-2">
                        <label className="block text-sm font-medium text-gray-300">
                            Apply Effects to Final Video
                        </label>
                        <label className="flex items-center">
                            <input
                                type="checkbox"
                                checked={applyEffects}
                                onChange={(e) => setApplyEffects(e.target.checked)}
                                disabled={disabled}
                                className="h-4 w-4 text-indigo-400 focus:ring-indigo-300 border-gray-600 rounded disabled:opacity-50"
                            />
                            <span className="ml-2 text-sm text-gray-300">Enable Effects</span>
                        </label>
                    </div>

                    {applyEffects && (
                        <div className="space-y-3">
                            <label className="block text-sm font-medium text-gray-300">
                                Select Effect
                            </label>
                            <div className="grid grid-cols-2 gap-2 max-h-48 overflow-y-auto">
                                {POST_COMPILATION_EFFECTS.map((effect) => (
                                    <label key={effect.id} className="flex items-center p-2 border border-white/6 rounded-md hover:bg-white/2 cursor-pointer">
                                        <input
                                            type="radio"
                                            name="effectType"
                                            value={effect.id}
                                            checked={selectedEffect === effect.id}
                                            onChange={(e) => setSelectedEffect(e.target.value)}
                                            disabled={disabled}
                                            className="h-4 w-4 text-indigo-300 focus:ring-indigo-300 border-gray-600 disabled:opacity-50"
                                        />
                                        <span className="ml-2 text-sm text-gray-300 flex items-center">
                                            <span className="mr-1">{effect.icon}</span>
                                            {effect.name}
                                        </span>
                                    </label>
                                ))}
                            </div>
                            <p className="text-xs text-gray-400">
                                {POST_COMPILATION_EFFECTS.find(e => e.id === selectedEffect)?.description}
                            </p>
                        </div>
                    )}
                </div>

                {/* Preview Info */}
                <div className="p-3 bg-purple-900/6 border border-purple-500 rounded-xl shadow-[0_0_24px_rgba(168,85,247,0.12)]">
                    <p className="text-sm text-gray-100">
                        <strong className="text-purple-300">🎯 AI Compilation Preview:</strong> Your <span className="font-semibold text-white">{selectedUploadIds.length}</span> videos will be analyzed for the best moments and compiled into a <span className="font-semibold text-white">{customDuration}-second</span> video optimized for <span className="font-semibold text-indigo-200">{selectedPresetData?.name}</span>.
                    </p>
                    <p className="text-sm text-gray-300 mt-3">
                        <strong className="text-purple-300">✨ Features:</strong> {TRANSITION_STYLES.find(s => s.id === selectedTransitionStyle)?.name.toLowerCase()} transitions, AI best parts detection, and {selectedPresetData?.aspect_ratio} aspect ratio for {selectedPresetData?.platform.toLowerCase()} optimization.
                    </p>
                </div>
            </div>
        </motion.div>
    )
}

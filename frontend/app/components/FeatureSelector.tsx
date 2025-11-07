'use client'

import StyleFilterSelector from './StyleFilterSelector'
import { motion } from 'framer-motion'

interface FeatureSelectorProps {
  selectedFeatures: string[]
  onFeatureToggle: (feature: string) => void
  selectedStyle?: string
  onStyleChange?: (style: string) => void
  disabled?: boolean
  mode?: 'single' | 'compilation'
}

const AI_FEATURES = [
  {
    id: 'ai-editing-suggestions',
    name: 'AI Film Editing Suggestions',
    description: 'Get intelligent cuts and transitions based on video analysis',
    icon: '🎬',
    category: 'Editing'
  },
  {
    id: 'auto-cut-silence',
    name: 'AI Auto-Cut & Transitions',
    description: 'Remove silence with AI scene analysis and smooth transitions',
    icon: '✂️',
    category: 'Editing'
  },
  {
    id: 'background-removal',
    name: 'AI Background Removal',
    description: 'Remove or replace video backgrounds',
    icon: '🎭',
    category: 'Visual'
  },
  {
    id: 'subtitles',
    name: 'AI Subtitle Generation',
    description: 'Generate and burn-in subtitles',
    icon: '💬',
    category: 'Audio'
  },
  {
    id: 'scene-detection',
    name: 'AI Scene Detection',
    description: 'Auto-split video into scenes',
    icon: '🎬',
    category: 'Analysis'
  },
  {
    id: 'voice-translate',
    name: 'Voice Translation & Dubbing',
    description: 'Translate speech and generate new audio',
    icon: '🌍',
    category: 'Audio'
  },
  {
    id: 'style',
    name: 'AI Style Filters',
    description: 'Apply artistic styles to video',
    icon: '🎨',
    category: 'Visual'
  },
  {
    id: 'object-remove',
    name: 'AI Object Removal',
    description: 'Remove unwanted objects from video',
    icon: '🧹',
    category: 'Visual'
  },
  {
    id: 'video-compilation',
    name: 'Video Compilation',
    description: 'AI-powered compilation from multiple videos with best parts detection',
    icon: '🎞️',
    category: 'Compilation'
  }
]

export default function FeatureSelector({
  selectedFeatures,
  onFeatureToggle,
  selectedStyle = 'cartoon',
  onStyleChange,
  disabled,
  mode = 'single'
}: FeatureSelectorProps) {
  // Filter features based on mode
  const filteredFeatures = mode === 'compilation'
    ? [] // No features needed in compilation mode since VideoCompilationSelector handles everything
    : AI_FEATURES.filter(feature => feature.id !== 'video-compilation')

  const groupedFeatures = filteredFeatures.reduce((acc, feature) => {
    if (!acc[feature.category]) {
      acc[feature.category] = []
    }
    acc[feature.category].push(feature)
    return acc
  }, {} as Record<string, typeof AI_FEATURES>)

  // Don't render anything in compilation mode since VideoCompilationSelector handles everything
  if (mode === 'compilation') {
    return null
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.03, boxShadow: '0 0 32px rgba(168,85,247,0.25)' }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl shadow-xl p-6"
    >
      <h2 className="text-xl font-semibold neon-heading mb-4">
        AI Features
      </h2>

      <div className="space-y-6">
        {Object.entries(groupedFeatures).map(([category, features]) => (
          <div key={category}>
            <h3 className="text-sm font-medium text-purple-400 uppercase tracking-wide mb-3">
              {category}
            </h3>

            <div className="space-y-3">
              {features.map((feature) => (
                <label
                  key={feature.id}
                  className={`flex items-start space-x-3 p-3 rounded-xl border cursor-pointer transition-all duration-200 shadow-lg backdrop-blur-md ${selectedFeatures.includes(feature.id)
                    ? 'border-purple-500 bg-purple-900/10 shadow-[0_0_24px_rgba(168,85,247,0.4)]'
                    : 'border-white/10 hover:border-purple-500'} ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <input
                    type="checkbox"
                    checked={selectedFeatures.includes(feature.id)}
                    onChange={() => onFeatureToggle(feature.id)}
                    disabled={disabled}
                    className="mt-1 h-4 w-4 accent-purple-500 focus:ring-purple-500 border-purple-500 rounded shadow-[0_0_8px_rgba(168,85,247,0.7)]"
                  />

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2">
                      <span className="text-xl text-purple-400 drop-shadow-[0_0_8px_rgba(168,85,247,0.7)]">{feature.icon}</span>
                      <span className="text-sm font-medium text-gray-100">
                        {feature.name}
                      </span>
                    </div>
                    <p className="text-xs text-gray-400 mt-1">
                      {feature.description}
                    </p>
                  </div>
                </label>
              ))}
            </div>
          </div>
        ))}
      </div>

      {selectedFeatures.length > 0 && (
        <div className="mt-6 p-3 bg-purple-900/10 border border-purple-500 rounded-xl shadow-[0_0_24px_rgba(168,85,247,0.4)]">
          <p className="text-sm text-purple-400">
            <span className="font-medium">{selectedFeatures.length}</span> feature{selectedFeatures.length !== 1 ? 's' : ''} selected
          </p>
        </div>
      )}

      {/* Style Filter Selector - Only in Single Mode */}
      {mode === 'single' && (
        <StyleFilterSelector
          isVisible={selectedFeatures.includes('style')}
          selectedStyle={selectedStyle}
          onStyleChange={onStyleChange || (() => { })}
          disabled={disabled}
        />
      )}
  </motion.div>
  )
}

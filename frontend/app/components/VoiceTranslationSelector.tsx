'use client'

import { useState, useEffect } from 'react'

interface VoiceTranslationSelectorProps {
  onLanguageChange: (language: string) => void
  onVoiceTypeChange: (voiceType: string) => void
  onSubtitleChange: (addSubtitles: boolean) => void
  disabled?: boolean
}

interface Language {
  code: string
  name: string
}

const SUPPORTED_LANGUAGES: Language[] = [
  { code: "es", name: "Spanish" },
  { code: "fr", name: "French" },
  { code: "de", name: "German" },
  { code: "it", name: "Italian" },
  { code: "pt", name: "Portuguese" },
  { code: "ru", name: "Russian" },
  { code: "ja", name: "Japanese" },
  { code: "ko", name: "Korean" },
  { code: "zh", name: "Chinese (Mandarin)" },
  { code: "ar", name: "Arabic" },
  { code: "hi", name: "Hindi" },
  { code: "nl", name: "Dutch" },
  { code: "pl", name: "Polish" },
  { code: "tr", name: "Turkish" }
]

const VOICE_TYPES = [
  { id: "female", name: "Female Voice" },
  { id: "male", name: "Male Voice" }
]

export default function VoiceTranslationSelector({
  onLanguageChange,
  onVoiceTypeChange,
  onSubtitleChange,
  disabled
}: VoiceTranslationSelectorProps) {
  const [selectedLanguage, setSelectedLanguage] = useState<string>("es")
  const [selectedVoiceType, setSelectedVoiceType] = useState<string>("female")
  const [addSubtitles, setAddSubtitles] = useState<boolean>(true)

  useEffect(() => {
    onLanguageChange(selectedLanguage)
  }, [selectedLanguage, onLanguageChange])

  useEffect(() => {
    onVoiceTypeChange(selectedVoiceType)
  }, [selectedVoiceType, onVoiceTypeChange])

  useEffect(() => {
    onSubtitleChange(addSubtitles)
  }, [addSubtitles, onSubtitleChange])

  return (
  <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl shadow-xl transition-all duration-300 hover:border-purple-500 hover:shadow-[0_0_24px_rgba(168,85,247,0.4)] p-5">
  <h3 className="text-lg font-semibold neon-heading text-gray-100 mb-4">Voice Translation Settings</h3>

      <div className="space-y-4">
        {/* Language Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-100 mb-2">
            Target Language
          </label>
          <select
            value={selectedLanguage}
            onChange={(e) => setSelectedLanguage(e.target.value)}
            disabled={disabled}
            className="w-full px-3 py-2 border border-white/6 bg-[#0b0c10]/40 text-gray-100 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400 disabled:bg-gray-800 disabled:cursor-not-allowed"
          >
            {SUPPORTED_LANGUAGES.map((language) => (
              <option key={language.code} value={language.code}>
                {language.name}
              </option>
            ))}
          </select>
        </div>

        {/* Voice Type Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-100 mb-2">
            Voice Type
          </label>
          <div className="space-y-2">
            {VOICE_TYPES.map((voiceType) => (
              <label key={voiceType.id} className="flex items-center">
                <input
                  type="radio"
                  name="voiceType"
                  value={voiceType.id}
                  checked={selectedVoiceType === voiceType.id}
                  onChange={(e) => setSelectedVoiceType(e.target.value)}
                  disabled={disabled}
                  className="h-4 w-4 text-indigo-300 focus:ring-indigo-300 border-gray-600 disabled:opacity-50"
                />
                <span className="ml-2 text-sm text-gray-100">
                  {voiceType.name}
                </span>
              </label>
            ))}
          </div>
        </div>

        {/* Subtitle Options */}
        <div>
          <label className="block text-sm font-medium text-gray-100 mb-2">
            Subtitle Options
          </label>
          <div className="space-y-2">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={addSubtitles}
                onChange={(e) => setAddSubtitles(e.target.checked)}
                disabled={disabled}
                className="h-4 w-4 text-indigo-300 focus:ring-indigo-300 border-gray-600 disabled:opacity-50"
              />
              <span className="ml-2 text-sm text-gray-100">
                Add translated subtitles to video
              </span>
            </label>
          </div>
        </div>

        {/* Preview Info */}
        <div className="p-3 bg-white/3 border border-white/6 rounded-md">
          <p className="text-sm text-gray-100">
            <strong className="text-indigo-300">Preview:</strong> Your video will be translated to{' '}
            <span className="font-semibold text-white">{SUPPORTED_LANGUAGES.find(l => l.code === selectedLanguage)?.name}</span>{' '}
            with a <span className="font-semibold text-indigo-200">{VOICE_TYPES.find(v => v.id === selectedVoiceType)?.name.toLowerCase()}</span>.
          </p>
          <p className="text-sm text-gray-300 mt-2">
            <strong className="text-indigo-300">Features:</strong> Optimized speech translation, AI voice generation, and {addSubtitles ? 'burned-in' : 'optional'} subtitle translation.
          </p>
        </div>
      </div>
    </div>
  )
}

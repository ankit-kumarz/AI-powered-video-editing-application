'use client'

interface StyleFilterSelectorProps {
  isVisible: boolean
  selectedStyle: string
  onStyleChange: (style: string) => void
  disabled?: boolean
}

const STYLE_OPTIONS = [
  {
    id: 'cartoon',
    name: 'Cartoon',
    description: 'Animated cartoon style with bold colors and edges',
    icon: '🎨',
    preview: 'cartoon-preview.jpg'
  },
  {
    id: 'sketch',
    name: 'Sketch',
    description: 'Pencil sketch effect with artistic lines',
    icon: '✏️',
    preview: 'sketch-preview.jpg'
  },
  {
    id: 'cinematic',
    name: 'Cinematic',
    description: 'Movie-like color grading and contrast',
    icon: '🎬',
    preview: 'cinematic-preview.jpg'
  },
  {
    id: 'vintage',
    name: 'Vintage',
    description: 'Old film look with sepia tones',
    icon: '📷',
    preview: 'vintage-preview.jpg'
  },
  {
    id: 'neon',
    name: 'Neon',
    description: 'Cyberpunk neon glow with edge effects',
    icon: '💫',
    preview: 'neon-preview.jpg'
  }
]

export default function StyleFilterSelector({ 
  isVisible, 
  selectedStyle, 
  onStyleChange, 
  disabled 
}: StyleFilterSelectorProps) {
  if (!isVisible) return null

  return (
    <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
      <h4 className="text-sm font-medium text-gray-700 mb-3">
        Choose Style Filter
      </h4>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {STYLE_OPTIONS.map((style) => (
          <label
            key={style.id}
            className={`relative flex flex-col items-center p-3 rounded-lg border-2 cursor-pointer transition-all ${
              selectedStyle === style.id
                ? 'border-purple-500 bg-purple-50'
                : 'border-gray-200 hover:border-gray-300 bg-white'
            } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <input
              type="radio"
              name="style-filter"
              value={style.id}
              checked={selectedStyle === style.id}
              onChange={(e) => onStyleChange(e.target.value)}
              disabled={disabled}
              className="sr-only"
            />
            
            <div className="text-2xl mb-2">{style.icon}</div>
            <div className="text-center">
              <div className="text-sm font-medium text-gray-900">
                {style.name}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {style.description}
              </div>
            </div>
            
            {selectedStyle === style.id && (
              <div className="absolute top-2 right-2">
                <div className="w-4 h-4 bg-purple-500 rounded-full flex items-center justify-center">
                  <div className="w-2 h-2 bg-white rounded-full"></div>
                </div>
              </div>
            )}
          </label>
        ))}
      </div>
      
      <div className="mt-3 text-xs text-gray-500">
        💡 Tip: Each style creates a unique artistic look for your video
      </div>
    </div>
  )
}

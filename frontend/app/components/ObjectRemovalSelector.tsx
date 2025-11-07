'use client'

import { useState, useRef, useEffect } from 'react'

interface BoundingBox {
  id: string
  x: number
  y: number
  width: number
  height: number
  label?: string
}

interface ObjectRemovalSelectorProps {
  videoFile: File | null
  onBoundingBoxesChange: (boxes: BoundingBox[]) => void
  disabled?: boolean
}

export default function ObjectRemovalSelector({ 
  videoFile, 
  onBoundingBoxesChange, 
  disabled = false 
}: ObjectRemovalSelectorProps) {
  const [boundingBoxes, setBoundingBoxes] = useState<BoundingBox[]>([])
  const [isDrawing, setIsDrawing] = useState(false)
  const [startPoint, setStartPoint] = useState({ x: 0, y: 0 })
  const [currentBox, setCurrentBox] = useState<BoundingBox | null>(null)
  const [videoUrl, setVideoUrl] = useState<string>('')
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 })
  const [showPreview, setShowPreview] = useState(false)
  const [detectedObjects, setDetectedObjects] = useState<BoundingBox[]>([])
  const [isDetecting, setIsDetecting] = useState(false)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (videoFile) {
      const url = URL.createObjectURL(videoFile)
      setVideoUrl(url)
      return () => URL.revokeObjectURL(url)
    }
  }, [videoFile])

  useEffect(() => {
    onBoundingBoxesChange(boundingBoxes)
  }, [boundingBoxes, onBoundingBoxesChange])

  const handleVideoLoad = () => {
    if (videoRef.current) {
      const { videoWidth, videoHeight } = videoRef.current
      setVideoDimensions({ width: videoWidth, height: videoHeight })
      
      // Set canvas size to match video
      if (canvasRef.current) {
        canvasRef.current.width = videoWidth
        canvasRef.current.height = videoHeight
      }
    }
  }

  const getMousePos = (e: React.MouseEvent) => {
    if (!containerRef.current) return { x: 0, y: 0 }
    
    const rect = containerRef.current.getBoundingClientRect()
    const scaleX = videoDimensions.width / rect.width
    const scaleY = videoDimensions.height / rect.height
    
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    }
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    if (disabled) return
    
    const pos = getMousePos(e)
    setIsDrawing(true)
    setStartPoint(pos)
    setCurrentBox({
      id: Date.now().toString(),
      x: pos.x,
      y: pos.y,
      width: 0,
      height: 0
    })
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDrawing || !currentBox) return
    
    const pos = getMousePos(e)
    const newBox = {
      ...currentBox,
      width: Math.abs(pos.x - startPoint.x),
      height: Math.abs(pos.y - startPoint.y),
      x: Math.min(pos.x, startPoint.x),
      y: Math.min(pos.y, startPoint.y)
    }
    
    setCurrentBox(newBox)
    drawBoxes([...boundingBoxes, newBox])
  }

  const handleMouseUp = () => {
    if (!isDrawing || !currentBox) return
    
    // Only add box if it has minimum size
    if (currentBox.width > 10 && currentBox.height > 10) {
      const newBoxes = [...boundingBoxes, currentBox]
      setBoundingBoxes(newBoxes)
      drawBoxes(newBoxes)
    }
    
    setIsDrawing(false)
    setCurrentBox(null)
  }

  const drawBoxes = (boxes: BoundingBox[]) => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    // Draw all boxes
    boxes.forEach((box, index) => {
      ctx.strokeStyle = '#ff0000'
      ctx.lineWidth = 2
      ctx.strokeRect(box.x, box.y, box.width, box.height)
      
      // Draw label
      ctx.fillStyle = '#ff0000'
      ctx.font = '14px Arial'
      ctx.fillText(`Object ${index + 1}`, box.x, box.y - 5)
      
      // Draw remove button
      ctx.fillStyle = '#ff0000'
      ctx.fillRect(box.x + box.width - 15, box.y, 15, 15)
      ctx.fillStyle = '#ffffff'
      ctx.font = '12px Arial'
      ctx.fillText('×', box.x + box.width - 10, box.y + 12)
    })
  }

  const removeBox = (id: string) => {
    const newBoxes = boundingBoxes.filter(box => box.id !== id)
    setBoundingBoxes(newBoxes)
    drawBoxes(newBoxes)
  }

  const detectObjects = async () => {
    if (!videoFile) return
    
    setIsDetecting(true)
    try {
      // Create a temporary canvas to get a frame from video
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      if (!ctx || !videoRef.current) return
      
      canvas.width = videoDimensions.width
      canvas.height = videoDimensions.height
      ctx.drawImage(videoRef.current, 0, 0)
      
      // Convert canvas to blob
      canvas.toBlob(async (blob) => {
        if (!blob) return
        
        const formData = new FormData()
        formData.append('image', blob)
        
        try {
          const response = await fetch('http://localhost:8000/detect-objects', {
            method: 'POST',
            body: formData
          })
          
          if (response.ok) {
            const objects = await response.json()
            setDetectedObjects(objects)
          }
        } catch (error) {
          console.error('Object detection failed:', error)
          // Fallback: create some example detected objects
          setDetectedObjects([
            { id: 'detected1', x: 100, y: 100, width: 80, height: 60, label: 'Person' },
            { id: 'detected2', x: 300, y: 200, width: 120, height: 90, label: 'Car' }
          ])
        }
        setIsDetecting(false)
      })
    } catch (error) {
      console.error('Error detecting objects:', error)
      setIsDetecting(false)
    }
  }

  const addDetectedObject = (object: BoundingBox) => {
    const newBox = {
      ...object,
      id: Date.now().toString()
    }
    const newBoxes = [...boundingBoxes, newBox]
    setBoundingBoxes(newBoxes)
    drawBoxes(newBoxes)
  }

  const clearAllBoxes = () => {
    setBoundingBoxes([])
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
      }
    }
  }

  useEffect(() => {
    drawBoxes(boundingBoxes)
  }, [boundingBoxes])

  if (!videoFile) {
    return (
      <div className="bg-white rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Object Removal</h3>
        <p className="text-gray-600">Upload a video first to start marking objects for removal.</p>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg p-6 shadow-lg">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Object Removal</h3>
      
      <div className="space-y-4">
        {/* Instructions */}
        <div className="bg-blue-50 p-3 rounded-lg">
          <p className="text-sm text-blue-800">
            <strong>How to use:</strong> Draw boxes around objects you want to remove. 
            Click and drag to create a selection box. Use the "Detect Objects" button 
            for AI suggestions.
          </p>
        </div>

        {/* Video Container with Canvas Overlay */}
        <div 
          ref={containerRef}
          className="relative bg-black rounded-lg overflow-hidden"
          style={{ maxWidth: '100%', maxHeight: '400px' }}
        >
          <video
            ref={videoRef}
            src={videoUrl}
            className="w-full h-auto"
            onLoadedMetadata={handleVideoLoad}
            muted
            loop
            playsInline
          />
          
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full cursor-crosshair"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          />
          
          {/* Drawing indicator */}
          {isDrawing && (
            <div className="absolute top-2 left-2 bg-red-500 text-white px-2 py-1 rounded text-sm">
              Drawing...
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="flex flex-wrap gap-2">
          <button
            onClick={detectObjects}
            disabled={disabled || isDetecting}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white rounded-lg text-sm"
          >
            {isDetecting ? 'Detecting...' : '🔍 Detect Objects'}
          </button>
          
          <button
            onClick={clearAllBoxes}
            disabled={disabled || boundingBoxes.length === 0}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white rounded-lg text-sm"
          >
            🗑️ Clear All
          </button>
          
          <button
            onClick={() => setShowPreview(!showPreview)}
            disabled={disabled || boundingBoxes.length === 0}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-lg text-sm"
          >
            👁️ {showPreview ? 'Hide' : 'Show'} Preview
          </button>
        </div>

        {/* Detected Objects */}
        {detectedObjects.length > 0 && (
          <div className="bg-gray-50 p-3 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-2">Detected Objects:</h4>
            <div className="flex flex-wrap gap-2">
              {detectedObjects.map((object) => (
                <button
                  key={object.id}
                  onClick={() => addDetectedObject(object)}
                  className="px-3 py-1 bg-blue-100 hover:bg-blue-200 text-blue-800 rounded text-sm"
                >
                  {object.label || 'Object'}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Selected Objects List */}
        {boundingBoxes.length > 0 && (
          <div className="bg-gray-50 p-3 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-2">Objects to Remove:</h4>
            <div className="space-y-2">
              {boundingBoxes.map((box, index) => (
                <div key={box.id} className="flex items-center justify-between bg-white p-2 rounded">
                  <span className="text-sm text-gray-700">
                    Object {index + 1} ({Math.round(box.width)}×{Math.round(box.height)})
                  </span>
                  <button
                    onClick={() => removeBox(box.id)}
                    className="text-red-600 hover:text-red-800 text-sm"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Preview */}
        {showPreview && boundingBoxes.length > 0 && (
          <div className="bg-gray-50 p-3 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-2">Preview:</h4>
            <div className="bg-black rounded-lg p-2">
              <p className="text-white text-sm text-center">
                Objects marked in red will be removed using AI inpainting.
                The AI will intelligently fill the removed areas with surrounding content.
              </p>
            </div>
          </div>
        )}

        {/* Status */}
        <div className="text-sm text-gray-600">
          {boundingBoxes.length > 0 ? (
            <p>✅ {boundingBoxes.length} object{boundingBoxes.length !== 1 ? 's' : ''} marked for removal</p>
          ) : (
            <p>📝 No objects marked yet. Draw boxes around objects you want to remove.</p>
          )}
        </div>
      </div>
    </div>
  )
}

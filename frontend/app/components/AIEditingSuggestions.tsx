'use client';

import React, { useState, useRef } from 'react';

interface Suggestion {
  timestamp: number;
  suggestion_type: string;
  description: string;
  confidence: number;
  reasoning: string;
  metadata?: any;
}

interface AIEditingSuggestionsProps {
  suggestions: Suggestion[];
  videoUrl?: string | null;
}

export default function AIEditingSuggestions({ suggestions, videoUrl }: AIEditingSuggestionsProps) {
  const [currentPage, setCurrentPage] = useState(1);
  const [filterType, setFilterType] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [compactMode, setCompactMode] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [hoveredSuggestion, setHoveredSuggestion] = useState<Suggestion | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const itemsPerPage = compactMode ? 20 : 10;

  // Calculate suggestion type counts
  const suggestionCounts = suggestions.reduce((counts, suggestion) => {
    const type = suggestion.suggestion_type;
    counts[type] = (counts[type] || 0) + 1;
    return counts;
  }, {} as Record<string, number>);

  // Filter suggestions
  const filteredSuggestions = suggestions.filter(suggestion => {
    const matchesType = filterType === 'all' || suggestion.suggestion_type === filterType;
    const matchesSearch = suggestion.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
      suggestion.reasoning.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesType && matchesSearch;
  });

  // Pagination
  const totalPages = Math.ceil(filteredSuggestions.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentSuggestions = filteredSuggestions.slice(startIndex, endIndex);

  const seekToTime = (timestamp: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = timestamp;
      setCurrentTime(timestamp);
      videoRef.current.play().catch(() => {
        // Auto-play might be blocked, that's okay
      });
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getSuggestionIcon = (type: string) => {
    switch (type) {
      case 'cut':
        return '✂️';
      case 'transition':
        return '➡️';
      case 'emphasis':
        return '⭐';
      case 'pace_change':
        return '⚡';
      default:
        return '⏰';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-100 text-green-800';
    if (confidence >= 0.6) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  const getSuggestionTypeColor = (type: string) => {
    switch (type) {
      case 'cut':
        return 'bg-red-500';
      case 'transition':
        return 'bg-blue-500';
      case 'emphasis':
        return 'bg-yellow-500';
      case 'pace_change':
        return 'bg-purple-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getSuggestionTypeBorderColor = (type: string) => {
    switch (type) {
      case 'cut':
        return 'border-red-600';
      case 'transition':
        return 'border-blue-600';
      case 'emphasis':
        return 'border-yellow-600';
      case 'pace_change':
        return 'border-purple-600';
      default:
        return 'border-gray-600';
    }
  };

  const handleVideoLoad = () => {
    if (videoRef.current) {
      setVideoDuration(videoRef.current.duration);
    }
  };

  const handleTimelineClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!videoRef.current || !videoDuration) return;

    const rect = event.currentTarget.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const percentage = clickX / rect.width;
    const newTime = percentage * videoDuration;

    seekToTime(newTime);
  };

  const getSuggestionPosition = (timestamp: number) => {
    if (!videoDuration) return 0;
    return (timestamp / videoDuration) * 100;
  };

  // Group suggestions by timestamp (within 2 seconds)
  const groupedSuggestions = suggestions.reduce((groups, suggestion) => {
    const key = Math.floor(suggestion.timestamp / 2) * 2;
    if (!groups[key]) groups[key] = [];
    groups[key].push(suggestion);
    return groups;
  }, {} as Record<number, Suggestion[]>);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
              ✂️ AI Editing Suggestions
            </h3>
            <p className="text-sm text-gray-600">
              Found {suggestions.length} suggestions • {filteredSuggestions.length} filtered
            </p>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setCompactMode(!compactMode)}
              className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors"
            >
              {compactMode ? '📖 Normal' : '📱 Compact'}
            </button>
          </div>
        </div>

        {/* Suggestion Type Summary */}
        <div className="mb-4 p-4 bg-gray-50 rounded-md">
          <h4 className="text-sm font-medium text-gray-900 mb-2">📊 Suggestion Breakdown</h4>
          <div className="flex flex-wrap gap-2">
            {Object.entries(suggestionCounts).map(([type, count]) => (
              <div
                key={type}
                className={`px-3 py-1 rounded-full text-xs font-medium ${getSuggestionTypeColor(type)} text-white`}
              >
                {getSuggestionIcon(type)} {type.charAt(0).toUpperCase() + type.slice(1)}: {count}
              </div>
            ))}
          </div>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-3 mb-4">
          <div className="flex-1 min-w-0">
            <input
              type="text"
              placeholder="Search suggestions..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Types ({suggestions.length})</option>
            {Object.entries(suggestionCounts).map(([type, count]) => (
              <option key={type} value={type}>
                {type.charAt(0).toUpperCase() + type.slice(1)} ({count})
              </option>
            ))}
          </select>
        </div>

        {/* Video Preview with Custom Timeline */}
        {videoUrl && (
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-gray-900">🎬 Video Preview</h4>
              <span className="text-xs text-gray-500">
                Click timeline dots to jump to suggestions
              </span>
            </div>

            <video
              ref={videoRef}
              src={videoUrl}
              controls
              className="w-full max-h-64 rounded-md mb-3"
              onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
              onLoadedMetadata={handleVideoLoad}
            />

            {/* Custom Timeline with Suggestion Markers */}
            <div className="relative">
              <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                <span>{formatTime(currentTime)}</span>
                <span>{formatTime(videoDuration)}</span>
              </div>

              <div
                className="relative h-8 bg-gray-200 rounded-md cursor-pointer"
                onClick={handleTimelineClick}
              >
                {/* Progress bar */}
                <div
                  className="absolute h-full bg-blue-500 rounded-md transition-all duration-100"
                  style={{ width: `${(currentTime / videoDuration) * 100}%` }}
                />

                {/* Suggestion markers */}
                {Object.entries(groupedSuggestions).map(([timestamp, suggestions]) => {
                  const position = getSuggestionPosition(parseFloat(timestamp));
                  const primarySuggestion = suggestions[0];
                  const suggestionCount = suggestions.length;

                  return (
                    <div
                      key={timestamp}
                      className="absolute top-1/2 transform -translate-y-1/2 -translate-x-1/2"
                      style={{ left: `${position}%` }}
                    >
                      <div className="relative">
                        {/* Main suggestion dot */}
                        <div
                          className={`w-4 h-4 rounded-full border-2 ${getSuggestionTypeColor(primarySuggestion.suggestion_type)} ${getSuggestionTypeBorderColor(primarySuggestion.suggestion_type)} cursor-pointer hover:scale-125 transition-transform`}
                          onClick={(e) => {
                            e.stopPropagation();
                            seekToTime(primarySuggestion.timestamp);
                          }}
                          onMouseEnter={() => setHoveredSuggestion(primarySuggestion)}
                          onMouseLeave={() => setHoveredSuggestion(null)}
                        />

                        {/* Multiple suggestions indicator */}
                        {suggestionCount > 1 && (
                          <div className="absolute -top-1 -right-1 w-3 h-3 bg-white border border-gray-300 rounded-full text-xs flex items-center justify-center text-gray-700 font-bold">
                            {suggestionCount}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Timeline legend */}
              <div className="flex items-center justify-center gap-4 mt-2 text-xs">
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <span>Cuts</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                  <span>Transitions</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                  <span>Emphasis</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                  <span>Pace Changes</span>
                </div>
              </div>
            </div>

            {/* Hover tooltip */}
            {hoveredSuggestion && (
              <div className="absolute bg-black text-white text-xs rounded-md p-2 z-10 pointer-events-none">
                <div className="font-medium">{hoveredSuggestion.suggestion_type.toUpperCase()}</div>
                <div>{hoveredSuggestion.description}</div>
                <div className="text-gray-300">{formatTime(hoveredSuggestion.timestamp)}</div>
              </div>
            )}
          </div>
        )}

        {/* Suggestions List */}
        <div className="space-y-3">
          {currentSuggestions.map((suggestion, index) => (
            <div
              key={index}
              className={`border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors cursor-pointer ${compactMode ? 'py-2' : ''}`}
              onClick={() => seekToTime(suggestion.timestamp)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-lg">{getSuggestionIcon(suggestion.suggestion_type)}</span>
                    <span className={`px-2 py-1 text-xs rounded-full font-medium ${getSuggestionTypeColor(suggestion.suggestion_type)} text-white`}>
                      {suggestion.suggestion_type.charAt(0).toUpperCase() + suggestion.suggestion_type.slice(1)}
                    </span>
                    <span className={`px-2 py-1 text-xs rounded-full ${getConfidenceColor(suggestion.confidence)}`}>
                      {Math.round(suggestion.confidence * 100)}%
                    </span>
                  </div>

                  <h4 className={`font-medium text-gray-900 ${compactMode ? 'text-sm' : ''}`}>
                    {suggestion.description}
                  </h4>

                  {!compactMode && (
                    <p className="text-sm text-gray-600 mt-1">
                      {suggestion.reasoning}
                    </p>
                  )}
                </div>

                <div className="flex items-center gap-2 ml-4">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      seekToTime(suggestion.timestamp);
                    }}
                    className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200 transition-colors"
                  >
                    {formatTime(suggestion.timestamp)}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between mt-6">
            <div className="text-sm text-gray-600">
              Showing {startIndex + 1}-{Math.min(endIndex, filteredSuggestions.length)} of {filteredSuggestions.length} suggestions
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 disabled:bg-gray-50 disabled:cursor-not-allowed transition-colors"
              >
                ← Previous
              </button>

              <span className="text-sm text-gray-600">
                Page {currentPage} of {totalPages}
              </span>

              <button
                onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 disabled:bg-gray-50 disabled:cursor-not-allowed transition-colors"
              >
                Next →
              </button>
            </div>
          </div>
        )}

        {/* No Results */}
        {currentSuggestions.length === 0 && (
          <div className="text-center py-8">
            <p className="text-gray-500">No suggestions found matching your criteria.</p>
          </div>
        )}
      </div>
    </div>
  );
}

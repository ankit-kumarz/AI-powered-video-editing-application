'use client';

import React, { useState, useEffect } from 'react';
import ScriptInput from './ScriptInput';
import AIEditingSuggestions from './AIEditingSuggestions';
import ProcessingStatus from './ProcessingStatus';

interface AIEditingSelectorProps {
    uploadId: string;
    videoFile: File;
}

interface EditingSuggestion {
    timestamp: number;
    suggestion_type: string;
    description: string;
    confidence: number;
    reasoning: string;
    metadata?: any;
}

interface VideoFeature {
    timestamp: number;
    feature_type: string;
    confidence: number;
    metadata?: any;
}

interface AnalysisResult {
    suggestions: EditingSuggestion[];
    video_features: VideoFeature[];
    video_duration: number;
    total_suggestions: number;
    cut_suggestions: number;
    transition_suggestions: number;
}

const AIEditingSelector: React.FC<AIEditingSelectorProps> = ({ uploadId, videoFile }) => {
    const [currentStep, setCurrentStep] = useState<'script' | 'analysis' | 'results'>('script');
    const [scriptContent, setScriptContent] = useState<string>('');
    const [isProcessing, setIsProcessing] = useState(false);
    const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
    const [error, setError] = useState<string>('');
    const [videoUrl, setVideoUrl] = useState<string>('');

    // Create video URL for preview
    useEffect(() => {
        if (videoFile) {
            const url = URL.createObjectURL(videoFile);
            setVideoUrl(url);
            return () => URL.revokeObjectURL(url);
        }
    }, [videoFile]);

    const handleScriptSubmit = async (content: string) => {
        if (!uploadId) {
            setError('No upload ID provided. Please upload a video first.');
            return;
        }

        setScriptContent(content);
        setCurrentStep('analysis');
        setIsProcessing(true);
        setError('');

        try {
            // Call the AI editing suggestions API
            const response = await fetch('/api/process-ai-editing', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    uploadId,
                    features: ['ai_editing_suggestions', 'script_analysis'],
                    scriptContent: content,
                }),
            });

            if (!response.ok) {
                const errorText = await response.text();
                try {
                    const errorData = JSON.parse(errorText);
                    throw new Error(errorData.error || 'Failed to analyze video and script');
                } catch {
                    throw new Error('Failed to analyze video and script');
                }
            }

            const result = await response.json();
            setAnalysisResult(result);
            setCurrentStep('results');
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setIsProcessing(false);
        }
    };

    const handleAnalyzeWithoutScript = async () => {
        setCurrentStep('analysis');
        setIsProcessing(true);
        setError('');

        try {
            const response = await fetch('/api/process-ai-editing', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    uploadId,
                    features: ['ai_editing_suggestions'],
                    scriptContent: null,
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to analyze video');
            }

            const result = await response.json();
            setAnalysisResult(result);
            setCurrentStep('results');
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setIsProcessing(false);
        }
    };

    const handleBackToScript = () => {
        setCurrentStep('script');
        setAnalysisResult(null);
        setError('');
    };

    const handleNewAnalysis = () => {
        setCurrentStep('script');
        setScriptContent('');
        setAnalysisResult(null);
        setError('');
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-bold neon-heading text-gray-100">AI Film Editing</h2>
                    <p className="text-gray-400">Get AI-powered suggestions for cuts and transitions based on script structure and video features</p>
                </div>
            </div>

            {/* Progress Steps */}
            <div className="flex items-center space-x-4 mb-6">
                <div className={`flex items-center ${currentStep === 'script' ? 'text-indigo-400' : 'text-gray-500'}`}>
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${currentStep === 'script' ? 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white' : 'bg-white/6 text-gray-300'}`}>
                        1
                    </div>
                    <span className="ml-2 text-gray-300">Script Input</span>
                </div>
                <div className="flex-1 h-1 bg-white/6"></div>
                <div className={`flex items-center ${currentStep === 'analysis' ? 'text-indigo-400' : currentStep === 'results' ? 'text-green-400' : 'text-gray-500'}`}>
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${currentStep === 'analysis' ? 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white' : currentStep === 'results' ? 'bg-gradient-to-r from-green-600 to-emerald-500 text-white' : 'bg-white/6 text-gray-300'}`}>
                        2
                    </div>
                    <span className="ml-2 text-gray-300">Analysis</span>
                </div>
                <div className="flex-1 h-1 bg-white/6"></div>
                <div className={`flex items-center ${currentStep === 'results' ? 'text-green-400' : 'text-gray-500'}`}>
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${currentStep === 'results' ? 'bg-gradient-to-r from-green-600 to-emerald-500 text-white' : 'bg-white/6 text-gray-300'}`}>
                        3
                    </div>
                    <span className="ml-2 text-gray-300">Results</span>
                </div>
            </div>

            {/* Error Display */}
            {error && (
                <div className="bg-red-900/8 border border-red-600/20 rounded-lg p-4">
                    <div className="flex">
                        <div className="flex-shrink-0">
                            <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                            </svg>
                        </div>
                        <div className="ml-3">
                            <h3 className="text-sm font-medium text-red-300">Analysis failed</h3>
                            <div className="mt-2 text-sm text-red-200">{error}</div>
                        </div>
                    </div>
                </div>
            )}

            {/* Content */}
            {currentStep === 'script' && (
                <div className="space-y-6">
                    <ScriptInput
                        onScriptSubmit={handleScriptSubmit}
                        isLoading={isProcessing}
                    />

                    <div className="text-center">
                            <div className="text-gray-500 mb-2">or</div>
                            <button
                                onClick={handleAnalyzeWithoutScript}
                                disabled={isProcessing}
                                className="px-6 py-3 bg-gradient-to-r from-purple-500 to-indigo-500 text-white rounded-md hover:from-purple-600 hover:to-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            >
                                Analyze Video Only (No Script)
                            </button>
                        </div>
                </div>
            )}

            {currentStep === 'analysis' && (
                <div className="space-y-6">
                    <ProcessingStatus
                        status={{ status: 'processing', progress: 50 }}
                        uploadId={uploadId}
                    />

                    <div className="bg-white/3 border border-white/6 rounded-md p-4">
                        <div className="flex">
                            <div className="flex-shrink-0">
                                <svg className="h-5 w-5 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
                                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                                </svg>
                            </div>
                            <div className="ml-3">
                                <h3 className="text-sm font-medium text-gray-100">Analysis in Progress</h3>
                                <div className="mt-2 text-sm text-gray-300">
                                    {scriptContent ?
                                        'Analyzing video features and script content to generate editing suggestions...' :
                                        'Analyzing video features to generate editing suggestions...'
                                    }
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {currentStep === 'results' && analysisResult && (
                <div className="space-y-6">
                    {/* Success Message */}
                    <div className="bg-green-50 border border-green-200 rounded-md p-4">
                        <div className="flex">
                            <div className="flex-shrink-0">
                                <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                </svg>
                            </div>
                            <div className="ml-3">
                                <h3 className="text-sm font-medium text-green-800">Analysis Complete</h3>
                                <div className="mt-2 text-sm text-green-700">
                                    Found {analysisResult.total_suggestions} editing suggestions based on video analysis
                                    {scriptContent && ' and script content'}.
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Results */}
                    <AIEditingSuggestions
                        suggestions={analysisResult.suggestions}
                        videoUrl={videoUrl}
                    />

                    {/* Action Buttons */}
                    <div className="flex space-x-4">
                        <button
                            onClick={handleNewAnalysis}
                            className="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                        >
                            New Analysis
                        </button>
                        <button
                            onClick={handleBackToScript}
                            className="px-6 py-3 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors"
                        >
                            Back to Script
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default AIEditingSelector;

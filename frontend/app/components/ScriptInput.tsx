'use client';

import React, { useState } from 'react';

interface ScriptInputProps {
    onScriptSubmit: (script: string) => void;
    isLoading?: boolean;
}

const ScriptInput: React.FC<ScriptInputProps> = ({ onScriptSubmit, isLoading = false }) => {
    const [scriptContent, setScriptContent] = useState<string>('');
    const [showExamples, setShowExamples] = useState(false);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (scriptContent.trim()) {
            onScriptSubmit(scriptContent.trim());
        }
    };

    const insertExample = (example: string) => {
        setScriptContent(example);
        setShowExamples(false);
    };

    const examples = [
        {
            title: "Basic Script",
            content: `SPEAKER: Hello everyone, welcome to our amazing video!
SPEAKER: Today we're going to show you something incredible.
SPEAKER: This is going to be absolutely thrilling!`
        },
        {
            title: "Interview Style",
            content: `INTERVIEWER: So tell us about your experience.
GUEST: Well, it was absolutely amazing.
INTERVIEWER: What was the most challenging part?
GUEST: Definitely the beginning, but it got easier.`
        },
        {
            title: "Narrative with Transitions",
            content: `NARRATOR: Our story begins in a small town.
NARRATOR: Meanwhile, in another location...
NARRATOR: Finally, the moment we've been waiting for!
NARRATOR: This dramatic turn changed everything.`
        }
    ];

    return (
        <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-6 shadow-xl">
            <div className="mb-4">
                <h3 className="text-lg font-medium neon-heading mb-2 text-gray-100">Script Content (Optional)</h3>
                <p className="text-sm text-gray-400">
                    Add your script to get more accurate editing suggestions. The AI will analyze both video features and script content.
                </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                    <label htmlFor="script" className="block text-sm font-medium text-gray-700 mb-2">
                        Script Text
                    </label>
                    <textarea
                        id="script"
                        value={scriptContent}
                        onChange={(e) => setScriptContent(e.target.value)}
                        placeholder={"Enter your script here...\n\nExample:\nSPEAKER: Hello everyone, welcome to our amazing video!\nSPEAKER: Today we&apos;re going to show you something incredible.\nSPEAKER: This is going to be absolutely thrilling!"}
                        className="w-full h-48 px-3 py-2 border border-white/6 rounded-md bg-[#0b0c10]/40 text-gray-200 focus:outline-none focus:ring-2 focus:ring-indigo-400 resize-none"
                        disabled={isLoading}
                    />
                </div>

                <div className="flex items-center justify-between">
                    <div className="flex space-x-3">
                        <button
                            type="submit"
                            disabled={isLoading || !scriptContent.trim()}
                            className="px-6 py-2 bg-gradient-to-r from-purple-500 to-indigo-500 text-white rounded-md hover:from-purple-600 hover:to-indigo-600 disabled:bg-gray-500 disabled:cursor-not-allowed transition-colors"
                        >
                            {isLoading ? 'Processing...' : 'Analyze with Script'}
                        </button>

                        <button
                            type="button"
                            onClick={() => setShowExamples(!showExamples)}
                            className="px-4 py-2 text-gray-300 hover:text-gray-100 transition-colors"
                        >
                            {showExamples ? 'Hide' : 'Show'} Examples
                        </button>
                    </div>

                    <div className="text-sm text-gray-500">
                        {scriptContent.length} characters
                    </div>
                </div>
            </form>

            {/* Examples Section */}
            {showExamples && (
                <div className="mt-6 p-4 bg-white/3 border border-white/6 rounded-xl">
                    <h4 className="text-sm font-medium text-gray-100 mb-3">Script Examples</h4>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        {examples.map((example, index) => (
                            <div key={index} className="border border-white/6 rounded-md p-3 bg-transparent">
                                <h5 className="text-sm font-medium text-gray-100 mb-2">{example.title}</h5>
                                <p className="text-xs text-gray-400 mb-2 line-clamp-3">{example.content.substring(0, 100)}...</p>
                                <button
                                    type="button"
                                    onClick={() => insertExample(example.content)}
                                    className="text-xs text-indigo-300 hover:text-indigo-200 transition-colors"
                                >
                                    Use this example
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Tips Section */}
            <div className="mt-6 p-4 bg-purple-900/6 border border-purple-500 rounded-xl shadow-[0_0_24px_rgba(168,85,247,0.08)]">
                <h4 className="text-sm font-medium text-purple-300 mb-2">💡 Tips for Better Results</h4>
                <ul className="text-sm text-gray-300 space-y-1">
                    <li>• Use &quot;SPEAKER:&quot; or character names to identify different speakers</li>
                    <li>• Include transition words like &quot;meanwhile&quot;, &quot;finally&quot;, &quot;suddenly&quot; for better timing</li>
                    <li>• Add emotional keywords like &quot;dramatic&quot;, &quot;exciting&quot;, &quot;calm&quot; for mood-based cuts</li>
                    <li>• Keep lines concise for more accurate timing estimation</li>
                </ul>
            </div>
        </div>
    );
};

export default ScriptInput;

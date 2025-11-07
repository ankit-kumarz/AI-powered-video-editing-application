import { NextRequest, NextResponse } from 'next/server';

interface AIEditingRequest {
    uploadId: string;
    features: string[];
    scriptContent?: string;
}

export async function POST(request: NextRequest) {
    try {
        const body: AIEditingRequest = await request.json();
        const { uploadId, features, scriptContent } = body;

        // Validate upload_id format (should be a UUID)
        const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
        if (!uuidRegex.test(uploadId)) {
            return NextResponse.json(
                { error: 'Invalid upload ID format' },
                { status: 400 }
            );
        }

        // Validate features
        if (!features || features.length === 0) {
            return NextResponse.json(
                { error: 'No features selected' },
                { status: 400 }
            );
        }

        // Call backend API
        const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
        const response = await fetch(`${backendUrl}/process-ai-editing`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                upload_id: uploadId,
                features: features,
                script_content: scriptContent
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Backend API error:', response.status, errorText);

            return NextResponse.json(
                { error: 'Failed to process AI editing' },
                { status: response.status }
            );
        }

        const data = await response.json();

        return NextResponse.json({
            success: true,
            suggestions: data.suggestions || [],
            video_features: data.video_features || [],
            total_suggestions: data.total_suggestions || 0,
            cut_suggestions: data.cut_suggestions || 0,
            transition_suggestions: data.transition_suggestions || 0
        });

    } catch (error) {
        console.error('Error processing AI editing:', error);

        return NextResponse.json(
            { error: 'Internal server error' },
            { status: 500 }
        );
    }
}

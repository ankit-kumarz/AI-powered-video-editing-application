import { motion } from 'framer-motion';
import { useRef, useEffect, useState } from 'react';

interface VideoPreviewProps {
    file: File;
    uploadId: string | null;
    processingStatus: any;
}

export default function VideoPreview({ file, uploadId, processingStatus }: VideoPreviewProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const [videoUrl, setVideoUrl] = useState<string>('');

    useEffect(() => {
        if (file) {
            const url = URL.createObjectURL(file);
            setVideoUrl(url);
            return () => URL.revokeObjectURL(url);
        }
    }, [file]);

    return (
        <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            whileHover={{ scale: 1.03, boxShadow: '0 0 32px rgba(168,85,247,0.25)' }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
            className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl shadow-xl transition-all duration-300 hover:border-purple-500 hover:shadow-[0_0_24px_rgba(168,85,247,0.4)]"
        >
            <h2 className="text-xl font-semibold text-gray-100 mb-4">Video Preview</h2>
            <div className="relative bg-black rounded-lg overflow-hidden mb-6">
                <video
                    ref={videoRef}
                    src={videoUrl}
                    controls
                    className="w-full h-auto max-h-96"
                    preload="metadata"
                >
                    Your browser does not support the video tag.
                </video>
            </div>
        </motion.div>
    );
}
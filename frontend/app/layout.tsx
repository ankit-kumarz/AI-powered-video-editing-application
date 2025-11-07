import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
    title: 'AI Video Editor MVP',
    description: 'AI-powered video editing with background removal, subtitles, scene detection, and more',
}

export default function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <html lang="en">
            <body className={inter.className + " min-h-screen"} style={{ background: 'linear-gradient(135deg, #0b0c10 0%, #121212 100%)', color: '#c5c6c7', fontFamily: 'Inter, Poppins, system-ui, sans-serif' }}>
                <div>
                    {children}
                </div>
            </body>
        </html>
    )
}

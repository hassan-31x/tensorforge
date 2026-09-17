import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = { title: 'Tensorforge | LLM Training Studio', description: 'Design a language model, simulate training, and understand memory, throughput, time, and cost.' };
export default function RootLayout({ children }: { children: React.ReactNode }) { return <html lang="en"><body>{children}</body></html> }

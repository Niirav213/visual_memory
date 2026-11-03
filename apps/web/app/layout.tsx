import './globals.css';
import Navbar from '../components/Navbar';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Visual Memory AI — From Seeing to Remembering',
  description: 'Production visual memory AI architecture for tracking and remembering everyday objects with evidence-backed spatial and temporal retrieval.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen text-white antialiased">
        <Navbar />
        <main className="mx-auto w-full max-w-[1800px] px-4 sm:px-6 lg:px-8 py-7 sm:py-9">
          {children}
        </main>
      </body>
    </html>
  );
}

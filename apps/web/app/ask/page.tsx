'use client';

import React, { FormEvent, KeyboardEvent, useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  ArrowDown,
  Camera,
  Clock,
  ExternalLink,
  ImageOff,
  MapPin,
  MessageSquare,
  Send,
  ShieldCheck,
  Sparkles,
  Volume2,
  VolumeX,
} from 'lucide-react';
import { api } from '../../lib/api';
import { QueryResponse, MemoryEvent } from '../../lib/types';
import EvidenceModal from '../../components/EvidenceModal';

interface ChatTurn {
  id: number;
  question: string;
  response?: QueryResponse;
  error?: string;
}

export default function AskPage() {
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState<ChatTurn[]>([]);
  const [loading, setLoading] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState<MemoryEvent | null>(null);
  const conversationRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const presetQuestions = [
    'Where is my laptop?',
    'When did I last see my phone?',
    'How many times have you seen a bottle?',
    'Find the red cup near the window',
    'When did the laptop disappear and return?',
    'What objects have you seen?',
  ];

  useEffect(() => {
    const conversation = conversationRef.current;
    if (conversation) {
      conversation.scrollTo({ top: conversation.scrollHeight, behavior: 'smooth' });
    }
  }, [messages, loading]);

  const handleAsk = async (event?: FormEvent<HTMLFormElement>, prompt?: string) => {
    event?.preventDefault();
    const askedQuestion = (prompt ?? question).trim();
    if (!askedQuestion || loading) return;

    const id = Date.now();
    setQuestion('');
    setMessages((current) => [...current, { id, question: askedQuestion }]);
    setLoading(true);

    try {
      const response = await api.askQuery(askedQuestion);
      setMessages((current) =>
        current.map((turn) => (turn.id === id ? { ...turn, response } : turn))
      );

      if (typeof window !== 'undefined' && 'speechSynthesis' in window && speaking) {
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(
          new SpeechSynthesisUtterance(response.answer.replace(/\*\*/g, ''))
        );
      }
    } catch (error) {
      console.error('Query execution error:', error);
      setMessages((current) =>
        current.map((turn) =>
          turn.id === id
            ? { ...turn, error: 'I could not reach Visual Memory. Check that the API is running, then try again.' }
            : turn
        )
      );
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const toggleSpeech = (answer?: string) => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    if (speaking) {
      window.speechSynthesis.cancel();
      setSpeaking(false);
    } else {
      setSpeaking(true);
      if (answer) {
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(new SpeechSynthesisUtterance(answer.replace(/\*\*/g, '')));
      }
    }
  };

  const openEvidence = (response: QueryResponse) => {
    if (!response.snapshot_uri) return;
    setSelectedEvent({
      event_id: response.event_id || 'evt_query',
      camera_id: response.camera_id || 'camera_01',
      zone_id: response.zone_id || 'study_desk',
      track_id: '101',
      class_name: response.object_class || 'object',
      confidence: response.confidence || 0.94,
      timestamp_utc: response.timestamp_utc || new Date().toISOString(),
      bbox_xyxy: [210, 140, 470, 360],
      snapshot_uri: response.snapshot_uri,
      event_label: 'observed',
    });
  };

  const handleInputKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void handleAsk();
    }
  };

  return (
    <div className="mx-auto flex h-[calc(100dvh-170px)] min-h-[560px] max-w-5xl flex-col overflow-hidden rounded-2xl border border-slate-700/80 bg-slate-950/70 shadow-2xl shadow-black/20 animate-fade-in">
      <header className="flex items-center justify-between border-b border-slate-700/70 bg-slate-900/80 px-4 py-3 sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-slate-800 text-amber-100 ring-1 ring-amber-100/15">
            <Sparkles className="h-5 w-5" />
          </div>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-semibold text-white sm:text-base">Visual Memory</h1>
            <div className="mt-0.5 flex items-center gap-1.5 text-[11px] text-slate-400">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              Evidence-backed assistant
            </div>
          </div>
        </div>
        <div className="hidden items-center gap-2 rounded-full border border-slate-700 bg-slate-900 px-3 py-1.5 text-[10px] font-medium uppercase tracking-wide text-slate-300 sm:flex">
          <ShieldCheck className="h-3.5 w-3.5 text-amber-200" />
          Object memory
        </div>
      </header>

      <div
        ref={conversationRef}
        className="flex-1 space-y-6 overflow-y-auto overscroll-contain px-4 py-5 sm:px-8 sm:py-7"
        aria-live="polite"
        aria-label="Conversation"
      >
        {messages.length === 0 ? (
          <div className="flex min-h-full flex-col items-center justify-center py-8 text-center">
            <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-slate-700 bg-slate-900 text-amber-100">
              <MessageSquare className="h-6 w-6" />
            </div>
            <h2 className="text-xl font-semibold tracking-tight text-white sm:text-2xl">
              What would you like to find?
            </h2>
            <p className="mt-2 max-w-md text-sm leading-6 text-slate-400">
              Ask about an object you have seen. I’ll look through your visual memories and show the evidence.
            </p>
            <div className="mt-7 flex max-w-2xl flex-wrap justify-center gap-2">
              {presetQuestions.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  disabled={loading}
                  onClick={() => void handleAsk(undefined, prompt)}
                  className="rounded-full border border-slate-700 bg-slate-900/80 px-3.5 py-2 text-left text-xs text-slate-300 transition-colors hover:border-amber-200/50 hover:bg-slate-800 hover:text-white disabled:opacity-50"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((turn) => (
              <div key={turn.id} className="space-y-5">
                <div className="flex justify-end">
                  <div className="max-w-[88%] rounded-2xl rounded-br-md border border-amber-100/20 bg-amber-100 px-4 py-3 text-sm leading-6 text-slate-950 shadow-md sm:max-w-[75%]">
                    {turn.question}
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-slate-700 bg-slate-800 text-amber-100">
                    <Sparkles className="h-4 w-4" />
                  </div>
                  <div className="min-w-0 max-w-[92%] space-y-3 sm:max-w-[82%]">
                    {turn.response ? (
                      <>
                        <div className="rounded-2xl rounded-tl-md border border-slate-700 bg-slate-900 px-4 py-3.5 text-sm leading-6 text-slate-100">
                          <div className="mb-2 flex items-center justify-between gap-4">
                            <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">
                              Memory response
                            </span>
                            <button
                              type="button"
                              onClick={() => toggleSpeech(turn.response?.answer)}
                              aria-label={speaking ? 'Stop speaking' : 'Read answer aloud'}
                              className="rounded-md p-1 text-slate-400 transition-colors hover:bg-slate-800 hover:text-amber-100"
                            >
                              {speaking ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
                            </button>
                          </div>
                          <p>{turn.response.answer}</p>

                          {turn.response.zone_id && (
                            <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
                              <EvidenceFact icon={MapPin} label="Observed zone" value={turn.response.zone_id} />
                              <EvidenceFact icon={Clock} label="Last seen" value={turn.response.time_ago || 'Recent'} />
                              <EvidenceFact icon={Camera} label="Camera" value={turn.response.camera_id || 'camera_01'} />
                              <EvidenceFact
                                icon={ShieldCheck}
                                label="Confidence"
                                value={turn.response.confidence ? `${(turn.response.confidence * 100).toFixed(0)}%` : '—'}
                              />
                            </div>
                          )}
                        </div>

                        {turn.response.snapshot_uri && (
                          <button
                            type="button"
                            onClick={() => openEvidence(turn.response!)}
                            className="group flex w-full items-center gap-3 rounded-xl border border-slate-700 bg-slate-900/80 p-2.5 text-left transition-colors hover:border-amber-100/40"
                          >
                            <div className="relative flex h-20 w-28 shrink-0 items-center justify-center overflow-hidden rounded-lg border border-slate-700 bg-slate-950 text-slate-500">
                              <ImageOff className="h-5 w-5" />
                              <img
                                src={turn.response.snapshot_uri}
                                alt={`Evidence for ${turn.response.object_class || 'object'}`}
                                className="absolute inset-0 h-full w-full object-cover"
                                onError={(event) => {
                                  (event.currentTarget as HTMLImageElement).style.display = 'none';
                                }}
                              />
                            </div>
                            <span className="min-w-0 flex-1">
                              <span className="flex items-center gap-1.5 text-xs font-semibold text-white">
                                Visual evidence
                                <ExternalLink className="h-3 w-3 text-amber-100" />
                              </span>
                              <span className="mt-1 block text-xs text-slate-400">
                                {turn.response.zone_id || 'Recorded sighting'}
                                {turn.response.time_ago ? ` · ${turn.response.time_ago}` : ''}
                              </span>
                            </span>
                          </button>
                        )}

                        {turn.response.disclaimer && (
                          <div className="flex items-start gap-2 px-1 text-xs leading-5 text-slate-400">
                            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-200" />
                            <span>{turn.response.disclaimer}</span>
                          </div>
                        )}
                      </>
                    ) : turn.error ? (
                      <div className="rounded-2xl rounded-tl-md border border-orange-400/30 bg-slate-900 px-4 py-3.5 text-sm leading-6 text-slate-200">
                        <div className="mb-1 flex items-center gap-2 text-xs font-semibold text-orange-200">
                          <AlertTriangle className="h-4 w-4" />
                          Could not get an answer
                        </div>
                        {turn.error}
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 rounded-2xl rounded-tl-md border border-slate-700 bg-slate-900 px-4 py-3.5 text-sm text-slate-400">
                        <span className="flex gap-1" aria-hidden="true">
                          <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-amber-100 [animation-delay:-0.2s]" />
                          <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-amber-100 [animation-delay:-0.1s]" />
                          <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-amber-100" />
                        </span>
                        Searching visual memories…
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
            {messages.length > 1 && (
              <div className="flex justify-center">
                <button
                  type="button"
                  onClick={() => conversationRef.current?.scrollTo({ top: conversationRef.current.scrollHeight, behavior: 'smooth' })}
                  className="inline-flex items-center gap-1.5 rounded-full border border-slate-700 bg-slate-900/90 px-3 py-1.5 text-[11px] text-slate-300 shadow-lg hover:text-white"
                >
                  <ArrowDown className="h-3 w-3" />
                  Latest message
                </button>
              </div>
            )}
          </>
        )}
      </div>

      <div className="border-t border-slate-700/80 bg-slate-900/90 p-3 sm:px-5 sm:py-4">
        {messages.length > 0 && (
          <div className="mb-2 flex gap-2 overflow-x-auto pb-1">
            {presetQuestions.slice(0, 3).map((prompt) => (
              <button
                key={prompt}
                type="button"
                disabled={loading}
                onClick={() => void handleAsk(undefined, prompt)}
                className="shrink-0 rounded-full border border-slate-700 px-3 py-1.5 text-[11px] text-slate-400 transition-colors hover:border-amber-100/40 hover:text-white disabled:opacity-50"
              >
                {prompt}
              </button>
            ))}
          </div>
        )}
        <form
          onSubmit={(event) => void handleAsk(event)}
          className="flex items-end gap-2 rounded-2xl border border-slate-600 bg-slate-950 p-2 shadow-inner focus-within:border-amber-100/60"
        >
          <textarea
            ref={inputRef}
            rows={1}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={handleInputKeyDown}
            disabled={loading}
            aria-label="Message Visual Memory"
            placeholder="Ask about an object you’ve seen…"
            className="max-h-32 min-h-10 flex-1 resize-none bg-transparent px-3 py-2.5 text-sm leading-5 text-white placeholder:text-slate-500 focus:outline-none disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={loading || !question.trim()}
            aria-label="Send message"
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-amber-100 text-slate-950 transition-colors hover:bg-amber-200 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Send className="h-4 w-4" />
          </button>
        </form>
        <p className="mt-2 text-center text-[10px] text-slate-500">
          Enter to send · Shift + Enter for a new line · Answers use recorded sightings only
        </p>
      </div>

      <EvidenceModal
        event={selectedEvent}
        onClose={() => setSelectedEvent(null)}
      />
    </div>
  );
}

function EvidenceFact({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof MapPin;
  label: string;
  value: string;
}) {
  return (
    <div className="min-w-0 rounded-lg border border-slate-700/80 bg-slate-950/70 px-2.5 py-2">
      <div className="mb-1 flex items-center gap-1.5 text-[10px] text-slate-400">
        <Icon className="h-3 w-3 shrink-0 text-amber-200" />
        <span>{label}</span>
      </div>
      <div className="truncate text-[11px] font-medium text-white">{value}</div>
    </div>
  );
}

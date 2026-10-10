import { useEffect, useRef, useState } from "react";
import {
  ArrowDown, ArrowRight, ArrowUp, BookOpen, ChevronDown, Clock3, FileText,
  BookOpenCheck, GraduationCap, Menu, MessageCircle, Plus, Search, ShieldCheck, Sparkles,
  Trash2, X,
} from "lucide-react";
import CurriculumPage from "./CurriculumPage.jsx";

const STORAGE_KEY = "cucumber-conversations-v1";
const suggestions = [
  { icon: GraduationCap, title: "Study plans", text: "What courses are in year 1?" },
  { icon: BookOpen, title: "Course details", text: "What are the prerequisites for calculus?" },
  { icon: FileText, title: "Credits", text: "How many credits are required to graduate?" },
  { icon: Search, title: "Explore topics", text: "Which courses cover data science?" },
];

function newConversation() {
  return { id: crypto.randomUUID(), title: "New conversation", updatedAt: Date.now(), messages: [] };
}

function readConversations() {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    return Array.isArray(value) ? value.filter(item => item && typeof item.id === "string" && Array.isArray(item.messages))
      .map(item => ({
        ...item,
        title: typeof item.title === "string" && item.title.trim() ? item.title : "New conversation",
        updatedAt: Number.isFinite(item.updatedAt) ? item.updatedAt : Date.now(),
        messages: item.messages.filter(message => message && typeof message === "object")
          .map(message => ({
            ...message,
            role: message.role === "user" ? "user" : "assistant",
            content: typeof message.content === "string" ? message.content : "",
            sources: Array.isArray(message.sources) ? message.sources : [],
          })),
      })) : [];
  } catch { return []; }
}

function timeLabel(timestamp) {
  const now = new Date();
  const date = new Date(timestamp);
  if (date.toDateString() === now.toDateString()) return "Today";
  const yesterday = new Date(now);
  yesterday.setDate(yesterday.getDate() - 1);
  return date.toDateString() === yesterday.toDateString() ? "Yesterday" : date.toLocaleDateString();
}

function databaseSearchStatus(message) {
  if (typeof message.database_searched === "boolean") return message.database_searched;
  // Older browser-saved turns predate this field but have an unambiguous status.
  if (message.status === "general" || message.status === "clarification_required") return false;
  if (message.sources?.length) return true;
  return null;
}

function CitationList({ sources }) {
  if (!sources?.length) return null;
  return <details className="citations">
    <summary><BookOpen size={15} /> Sources <span>{sources.length}</span><ChevronDown size={15} /></summary>
    <div className="citation-list">{sources.map((source, index) => <a
      className="citation-item" key={`${source.chunk_id}-${index}`} href={source.url || "#sources"}
      target={source.url ? "_blank" : undefined} rel="noreferrer">
      <span className="citation-number">{index + 1}</span>
      <span><strong>{source.title || `${source.program || "Curriculum"} · ${source.course_code || "Course information"}`}</strong>
        <small>{[source.program, source.academic_year, source.plan, source.source_page ? `Page ${source.source_page}` : null].filter(Boolean).join(" · ")}</small>
        {source.text && <em>{source.text}</em>}
      </span>
    </a>)}</div>
  </details>;
}

function Message({ message }) {
  const user = message.role === "user";
  const databaseSearched = databaseSearchStatus(message);
  return <article className={`message-row ${user ? "from-user" : "from-assistant"}`}>
    {!user && <div className="assistant-avatar"><span>C</span></div>}
    <div className="message-column">
      <div className={`message-bubble ${user ? "user-bubble" : "assistant-bubble"}`}>
        {!user && <div className="message-brand"><Sparkles size={14} /> Cucumber</div>}
        {!user && <div className={`retrieval-badge ${databaseSearched === true ? "retrieval-done" : databaseSearched === false ? "retrieval-skipped" : "retrieval-unknown"}`}>
          <Search size={12} />
          {databaseSearched === true ? "Curriculum database searched" : databaseSearched === false ? "No database search" : "Search status unavailable"}
        </div>}
        <p style={{ whiteSpace: "pre-line" }}>{message.content}</p>
        {message.status && message.status !== "answer" && <div className={`response-state state-${message.status}`}>
          <span />{message.status === "insufficient_evidence" ? "Not enough verified information" : message.status === "clarification_required" ? "More detail needed" : message.status === "general" ? "General conversation" : "Response status"}
        </div>}
      </div>
      {!user && <CitationList sources={message.sources} />}
      <time>{new Date(message.createdAt || Date.now()).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}</time>
    </div>
  </article>;
}

export default function App() {
  const [conversations, setConversations] = useState(readConversations);
  const [activeId, setActiveId] = useState(() => readConversations()[0]?.id || null);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [searchText, setSearchText] = useState("");
  const [scrollDown, setScrollDown] = useState(false);
  const threadRef = useRef(null);
  const textareaRef = useRef(null);
  const active = conversations.find(item => item.id === activeId) || null;

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations)); }
    catch { /* Keep the chat usable when browser storage is unavailable. */ }
  }, [conversations]);

  useEffect(() => {
    fetch("/api/health").then(response => response.json()).then(setHealth).catch(() => setHealth({ status: "offline" }));
  }, []);

  useEffect(() => {
    const node = threadRef.current;
    if (!node) return;
    const onScroll = () => setScrollDown(node.scrollHeight - node.scrollTop - node.clientHeight > 120);
    node.addEventListener("scroll", onScroll);
    node.scrollTo({ top: node.scrollHeight, behavior: "smooth" });
    return () => node.removeEventListener("scroll", onScroll);
  }, [active?.messages.length, sending]);

  function updateConversation(id, updater) {
    setConversations(previous => previous.map(item => item.id === id ? updater(item) : item));
  }

  function closeSidebarOnMobile() {
    if (window.matchMedia("(max-width: 600px)").matches) setSidebarOpen(false);
  }

  function createChat() {
    const conversation = newConversation();
    setConversations(previous => [conversation, ...previous]);
    setActiveId(conversation.id);
    setDraft("");
    setError("");
    closeSidebarOnMobile();
    requestAnimationFrame(() => textareaRef.current?.focus());
  }

  function removeChat(id, event) {
    event?.stopPropagation();
    const remaining = conversations.filter(item => item.id !== id);
    setConversations(remaining);
    if (activeId === id) setActiveId(remaining[0]?.id || null);
  }

  async function sendMessage(value = draft, retrying = false) {
    const question = value.trim();
    if (!question || sending) return;
    let conversation = active;
    if (!conversation) {
      conversation = newConversation();
      setConversations(previous => [conversation, ...previous]);
      setActiveId(conversation.id);
    }
    const lastUser = [...conversation.messages].reverse().find(message => message.role === "user");
    const retryExisting = retrying && lastUser?.content === question;
    const userMessage = retryExisting ? lastUser : { id: crypto.randomUUID(), role: "user", content: question, createdAt: Date.now() };
    const conversationId = conversation.id;
    const previousMessages = retryExisting
      ? conversation.messages.filter(message => message.id !== userMessage.id)
      : conversation.messages;
    if (!retryExisting) updateConversation(conversationId, item => ({
      ...item, title: item.messages.length ? item.title : question.slice(0, 48),
      updatedAt: Date.now(), messages: [...item.messages, userMessage],
    }));
    setDraft("");
    setError("");
    setSending(true);
    try {
      const response = await fetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history: previousMessages.slice(-8).map(({ role, content }) => ({ role, content })) }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || `Unable to answer (${response.status})`);
      const answer = { id: crypto.randomUUID(), role: "assistant", content: data.answer,
        status: data.status, use_rag: data.use_rag, database_searched: data.database_searched,
        sources: data.sources || [], createdAt: Date.now() };
      updateConversation(conversationId, item => ({ ...item, updatedAt: Date.now(), messages: [...item.messages, answer] }));
      setHealth(current => data.index_ready ? { ...current, database_ready: true } : current);
    } catch (requestError) {
      setError(requestError.message || "Something went wrong. Try again.");
    } finally { setSending(false); }
  }

  function retryLast() {
    const question = [...(active?.messages || [])].reverse().find(message => message.role === "user")?.content;
    if (question) sendMessage(question, true);
  }

  function onComposerKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      sendMessage();
    }
  }

  const grouped = conversations.filter(item => item.messages.length > 0 && item.title.toLowerCase().includes(searchText.toLowerCase()))
    .reduce((groups, conversation) => {
      const group = timeLabel(conversation.updatedAt);
      (groups[group] ||= []).push(conversation);
      return groups;
    }, {});
  const hasMessages = Boolean(active?.messages.length);

  if (window.location.pathname === "/curriculum") return <CurriculumPage />;

  return <main className={`app-shell ${sidebarOpen ? "sidebar-visible" : "sidebar-hidden"}`}>
    <aside className="sidebar">
      <div className="sidebar-top">
        <a className="brand" href="/" aria-label="Cucumber home"><span className="brand-mark">C</span><span><strong>cucumber</strong><small>CURRICULUM ASSISTANT</small></span></a>
        <button className="icon-button sidebar-close" onClick={() => setSidebarOpen(false)} aria-label="Close sidebar"><X size={18} /></button>
      </div>
      <button className="new-chat-button" onClick={createChat}><Plus size={17} /> New conversation <span>⌘ K</span></button>
      <a className="sidebar-page-link" href="/curriculum"><BookOpenCheck size={16} /><span>Curriculum document</span><ArrowRight size={14} /></a>
      <label className="history-search"><Search size={16} /><input placeholder="Search conversations" value={searchText} onChange={event => setSearchText(event.target.value)} /></label>
      <div className="history-list">
        {Object.entries(grouped).map(([group, items]) => <section className="history-group" key={group}>
          <h2>{group}</h2>{items.map(item => <div key={item.id} className={`history-item ${item.id === activeId ? "selected" : ""}`}>
            <button onClick={() => { setActiveId(item.id); closeSidebarOnMobile(); setError(""); }} title={item.title}><MessageCircle size={15} /><span>{item.title}</span></button>
            <button className="history-delete" onClick={event => removeChat(item.id, event)} aria-label={`Delete ${item.title}`}><Trash2 size={14} /></button>
          </div>)}
        </section>)}
        {Object.keys(grouped).length === 0 && <p className="history-empty">Your conversations will appear here.</p>}
      </div>
      <div className="sidebar-bottom">
        <div className="source-status"><span className={`status-light ${health?.database_ready ? "ready" : ""}`} /><span><strong>{health?.database_ready ? "Curriculum index ready" : "Curriculum index"}</strong><small>{health?.database_ready ? "Verified course sources" : health?.status === "offline" ? "API unavailable" : "Build index to enable answers"}</small></span></div>
        <div className="privacy-note"><ShieldCheck size={15} /> Conversations are saved in this browser</div>
      </div>
    </aside>

    <section className="workspace">
      <header className="topbar">
        <button className="icon-button menu-open" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar"><Menu size={19} /></button>
        <div className="breadcrumb"><span>Workspace</span><span className="crumb-divider">/</span><strong>Ask Cucumber</strong></div>
        <div className="topbar-right"><span className="secure-label"><ShieldCheck size={14} /> Source-grounded answers</span><button className="top-new" onClick={createChat}><Plus size={15} /> New chat</button></div>
      </header>

      <div className={`conversation-area ${hasMessages ? "has-messages" : "welcome-area"}`} ref={threadRef}>
        {!hasMessages ? <div className="welcome-content">
          <div className="welcome-kicker"><Sparkles size={14} /> YOUR CURRICULUM, MADE CLEAR</div>
          <h1>Good questions.<br /><span>Grounded answers.</span></h1>
          <p className="welcome-copy">Ask about courses, study plans, credits, and curriculum requirements. Answers are based on indexed curriculum sources.</p>
          <div className="prompt-grid">{suggestions.map(({ icon: Icon, title, text }) => <button key={title} className="prompt-card" onClick={() => sendMessage(text)}>
            <span className="prompt-icon"><Icon size={17} /></span><strong>{title}</strong><span>{text}</span><ArrowUp size={15} className="prompt-arrow" />
          </button>)}</div>
          <div className="welcome-note"><span className="mini-brand">C</span><span>Course facts are checked against the curriculum index.</span><span className="note-dot">·</span><span>Sources included when available</span></div>
        </div> : <div className="message-thread">
          <div className="thread-heading"><span className="thread-marker"><MessageCircle size={15} /></span><div><strong>{active.title}</strong><small>Curriculum conversation</small></div></div>
          {active.messages.map(message => <Message key={message.id} message={message} />)}
          {sending && <div className="message-row from-assistant"><div className="assistant-avatar"><span>C</span></div><div className="message-column"><div className="message-bubble assistant-bubble pending-bubble"><div className="message-brand"><Sparkles size={14} /> Cucumber</div><div className="thinking"><i /><i /><i /><span>Checking curriculum sources</span></div></div></div></div>}
        </div>}
      </div>
      {scrollDown && <button className="scroll-bottom" aria-label="Scroll to latest" onClick={() => threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight, behavior: "smooth" })}><ArrowDown size={16} /></button>}

      <footer className="composer-wrap">
        <div className={`composer ${error ? "composer-error" : ""}`}>
          {error && <div className="composer-error-text">{error}<button onClick={retryLast}>Retry last question</button></div>}
          <textarea ref={textareaRef} value={draft} onChange={event => setDraft(event.target.value)} onKeyDown={onComposerKeyDown}
            placeholder="Ask about a course, plan, or requirement..." rows={2} maxLength={3000} aria-label="Ask a curriculum question" disabled={sending} />
          <div className="composer-tools"><span><span className="keyboard-key">↵</span> to send <span className="keyboard-key">⇧ ↵</span> for a new line</span>
            <div className="composer-actions"><span className="character-count">{draft.length > 0 ? `${draft.length}/3000` : ""}</span>
              <button className="send-button" aria-label="Send question" onClick={() => sendMessage()} disabled={!draft.trim() || sending}><ArrowUp size={18} /></button>
            </div>
          </div>
        </div>
        <div className="footer-caption"><span><ShieldCheck size={13} /> Grounded in curriculum documents</span><span><Clock3 size={13} /> Check the cited source for official details</span></div>
      </footer>
      {!sidebarOpen && <div className="mobile-backdrop" onClick={() => setSidebarOpen(true)} />}
    </section>
  </main>;
}

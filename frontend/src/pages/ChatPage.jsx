import { useEffect, useMemo, useRef, useState } from "react";
import { askQuestion, fetchPrograms } from "../api";
import {
  buildConversationContext,
  defaultCatalogKey,
  resetContextForEdition,
} from "../chatScope";

const STORAGE_KEY = "cucumber-chat-sessions-v1";
const HARD_STATUS_LABELS = {
  answer: "ตอบแล้ว",
  satisfied: "เป็นไปตามเงื่อนไข",
  violation: "ไม่เป็นไปตามลำดับที่กำหนด",
  incomplete_evidence: "ข้อมูลยังไม่เพียงพอที่จะยืนยันทั้งหมด",
  invalid_scope: "ขอบเขตหลักสูตรไม่ถูกต้อง",
  deficit: "พบข้อขาดตามข้อมูลที่ตรวจสอบได้",
  infeasible: "จัดลำดับไม่ได้ตามข้อจำกัดที่ตรวจสอบได้",
  clarification_required: "ต้องระบุข้อมูลเพิ่มเติม",
  unsupported: "ยังไม่รองรับคำถามนี้",
  error: "เกิดข้อผิดพลาดในการตรวจสอบ",
};

function loadStored() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed.sessions)) return null;
    return parsed;
  } catch {
    return null;
  }
}

function newSession(program, catalogKey = "") {
  return {
    id: `s-${Date.now()}-${Math.floor(Math.random() * 1e6)}`,
    program: program || "",
    catalogKey,
    title: "New chat",
    messages: [],
    context: null,
    createdAt: Date.now(),
  };
}

export default function ChatPage() {
  const [programs, setPrograms] = useState([]);
  const [sessions, setSessions] = useState(() => {
    const stored = loadStored();
    if (stored && stored.sessions.length > 0) return stored.sessions;
    return [newSession("IT")];
  });
  const [activeId, setActiveId] = useState(() => {
    const stored = loadStored();
    if (stored && stored.sessions.length > 0) {
      return stored.activeId || stored.sessions[0].id;
    }
    return null;
  });
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const threadRef = useRef(null);

  const active = useMemo(
    () => sessions.find((s) => s.id === activeId) || sessions[0],
    [sessions, activeId]
  );

  useEffect(() => {
    fetchPrograms()
      .then((res) => setPrograms(res.programs || []))
      .catch(() => setPrograms([]));
  }, []);

  useEffect(() => {
    if (programs.length === 0) return;
    setSessions((previous) =>
      previous.map((session) => {
        const program = programs.find(
          (item) => item.program_code === session.program
        );
        const editions = program?.editions || [];
        const existing = session.catalogKey || session.context?.catalog_key || "";
        const isAvailable = editions.some(
          (edition) => edition.catalog_key === existing
        );
        const catalogKey = isAvailable
          ? existing
          : defaultCatalogKey(session.program, programs);
        if (catalogKey === session.catalogKey) return session;
        const changedEdition = Boolean(existing && existing !== catalogKey);
        return {
          ...session,
          catalogKey,
          context: changedEdition
            ? resetContextForEdition(session.program, catalogKey)
            : session.context,
        };
      })
    );
  }, [programs]);

  useEffect(() => {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ sessions, activeId: active?.id || null })
      );
    } catch {
      // storage is best-effort only
    }
  }, [sessions, active]);

  useEffect(() => {
    threadRef.current?.scrollTo({
      top: threadRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [active?.messages.length, loading]);

  function updateActive(patch) {
    setSessions((prev) =>
      prev.map((s) => (s.id === active.id ? { ...s, ...patch } : s))
    );
  }

  function handleNewChat() {
    const program = active?.program || "IT";
    const session = newSession(program, defaultCatalogKey(program, programs));
    setSessions((prev) => [session, ...prev]);
    setActiveId(session.id);
    setQuestion("");
    setError("");
  }

  function handleDeleteSession(id) {
    setSessions((prev) => {
      const next = prev.filter((s) => s.id !== id);
      if (next.length === 0) {
        const program = active?.program || "IT";
        const fresh = newSession(program, defaultCatalogKey(program, programs));
        setActiveId(fresh.id);
        return [fresh];
      }
      if (id === active?.id) setActiveId(next[0].id);
      return next;
    });
  }

  function handleProgramChange(program) {
    // Keep the current session and its history; only the scope changes.
    // Context is cleared so the next question is seeded with the new
    // program instead of chaining the previous program's scope.
    updateActive({
      program,
      catalogKey: defaultCatalogKey(program, programs),
      context: null,
    });
    setQuestion("");
    setError("");
  }

  function handleCatalogChange(catalogKey) {
    updateActive({
      catalogKey,
      context: resetContextForEdition(active.program, catalogKey),
    });
    setQuestion("");
    setError("");
  }

  async function handleAsk() {
    const q = question.trim();
    setError("");
    if (!active) return;
    if (q.length < 2) {
      setError("กรุณาพิมพ์คำถามอย่างน้อย 2 ตัวอักษร");
      return;
    }
    const selectedProgram = programs.find(
      (item) => item.program_code === active.program
    );
    if ((selectedProgram?.editions?.length || 0) > 1 && !active.catalogKey) {
      setError("กรุณาเลือกปีหลักสูตรก่อนส่งคำถาม");
      return;
    }
    setLoading(true);
    try {
      // The selected catalog is authoritative over any stale follow-up context.
      const seed = buildConversationContext(active);
      const data = await askQuestion(q, seed);
      const entry = {
        id: Date.now(),
        question: q,
        answer: data.answer || "ไม่พบคำตอบ",
        status: data.status,
        route: data.route,
        provenance: data.provenance || [],
      };
      updateActive({
        messages: [...active.messages, entry],
        context: buildConversationContext({ ...active, context: data.next_context }),
        title:
          active.messages.length === 0
            ? q.length > 42
              ? `${q.slice(0, 42)}…`
              : q
            : active.title,
      });
      setQuestion("");
    } catch (err) {
      setError(err.message || "เกิดข้อผิดพลาด");
    } finally {
      setLoading(false);
    }
  }

  if (!active) return null;

  const selectedProgram = programs.find(
    (item) => item.program_code === active.program
  );
  const editions = selectedProgram?.editions || [];

  return (
    <div className="page">
      <header className="page-header">
        <h1>Chat</h1>
        <p>ระบบถาม–ตอบข้อมูลหลักสูตรจากฐานข้อมูล CUCUMBER</p>
      </header>

      <div className="chat-layout">
        <aside className="card session-panel">
          <button type="button" className="session-new" onClick={handleNewChat}>
            + New chat
          </button>
          <ul className="session-list">
            {sessions.map((s) => (
              <li key={s.id}>
                <button
                  type="button"
                  className={`session-item${s.id === active.id ? " active" : ""}`}
                  onClick={() => {
                    setActiveId(s.id);
                    setQuestion("");
                    setError("");
                  }}
                  title={s.title}
                >
                  <span className="session-program">{s.program || "All"}</span>
                  <span className="session-title">{s.title}</span>
                  <span className="session-count">{s.messages.length}</span>
                </button>
                {sessions.length > 1 && (
                  <button
                    type="button"
                    className="session-del"
                    aria-label="Delete chat"
                    onClick={() => handleDeleteSession(s.id)}
                  >
                    ✕
                  </button>
                )}
              </li>
            ))}
          </ul>
        </aside>

        <div className="chat-main">
          <section className="card program-bar">
            <div className="field field-inline">
              <label htmlFor="chatProgram">Program</label>
              <select
                id="chatProgram"
                value={active.program}
                disabled={loading}
                onChange={(e) => handleProgramChange(e.target.value)}
              >
                <option value="">All programs</option>
                {programs.map((p) => (
                  <option key={p.program_code} value={p.program_code}>
                    {p.program_code}
                  </option>
                ))}
              </select>
            </div>
            {editions.length > 0 && (
              <div className="field field-inline">
                <label htmlFor="chatCatalog">Curriculum year</label>
                <select
                  id="chatCatalog"
                  value={active.catalogKey || ""}
                  disabled={loading || editions.length === 1}
                  onChange={(e) => handleCatalogChange(e.target.value)}
                >
                  {editions.length > 1 && !active.catalogKey && (
                    <option value="" disabled>
                      Select a year
                    </option>
                  )}
                  {editions.map((edition) => (
                    <option key={edition.catalog_key} value={edition.catalog_key}>
                      {edition.academic_year || edition.catalog_key}
                      {editions.filter(
                        (item) => item.academic_year === edition.academic_year
                      ).length > 1
                        ? ` (${edition.catalog_key})`
                        : ""}
                    </option>
                  ))}
                </select>
              </div>
            )}
            <span className="hint">
              คำถามในห้องนี้จะใช้ {active.program || "ทุกหลักสูตร"} เป็นขอบเขต
              ไม่ต้องพิมพ์ชื่อหลักสูตรทุกครั้ง
            </span>
          </section>

          <section className="card thread-card">
            <div className="thread" ref={threadRef}>
              {active.messages.length === 0 && (
                <div className="empty">
                  ยังไม่มีบทสนทนาในห้องนี้ เริ่มถามได้เลย
                </div>
              )}
              {active.messages.map((m) => (
                <div key={m.id}>
                  <div className="msg-row user">
                    <div className="bubble bubble-user">{m.question}</div>
                  </div>
                  <div className="msg-row assistant">
                    <div className="bubble bubble-assistant">{m.answer}</div>
                  </div>
                  <div className="msg-row assistant">
                    <div className="meta">
                      <span className={`badge status-${m.status}`}>
                        {m.route === "hard"
                          ? HARD_STATUS_LABELS[m.status] || "ผลการตรวจสอบหลักสูตร"
                          : m.status}
                      </span>
                      {m.route !== "hard" && m.provenance.length > 0 && (
                        <span className="hint">
                          {m.provenance
                            .map(
                              (p) =>
                                `${p.program || "-"} p.${p.source_page ?? "?"}`
                            )
                            .join(" · ")}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              {loading && (
                <div className="msg-row assistant">
                  <div className="bubble bubble-assistant typing">
                    กำลังค้นหา...
                  </div>
                </div>
              )}
            </div>

            <div className="composer">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") handleAsk();
                }}
                placeholder="พิมพ์คำถามได้เลย เช่น วิชา 06016454 กี่หน่วยกิต"
              />
              <div className="actions">
                <button
                  type="button"
                  onClick={handleAsk}
                  disabled={loading}
                >
                  {loading ? "กำลังค้นหา..." : "ส่ง"}
                </button>
                <span className="hint">กด Ctrl + Enter เพื่อส่งคำถาม</span>
              </div>
              {error && <div className="error visible">{error}</div>}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

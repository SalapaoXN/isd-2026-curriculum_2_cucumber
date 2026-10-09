import { useEffect, useMemo, useRef, useState } from "react";
import { askQuestion, fetchPrograms } from "../api";
import PlanSelector from "../components/PlanSelector";
import { createChatSubmissionManager } from "../chatSubmission";
import {
  buildConversationContext,
  newSession,
  normalizeStoredSession,
  selectProgram,
  selectProgramForRetry,
  applyChatResponse,
  defaultCatalogKey,
  resetContextForEdition,
  availablePlans,
  displayablePlans,
  selectPlan,
  selectPlanForRetry,
  clarificationPlans,
  clarificationCatalogs,
  catalogOptionLabel,
  selectCatalog,
  selectCatalogForRetry,
  normalCatalogTarget,
  updateMessageClarification,
  planLabel,
  formatElapsedTime,
  provenanceLabel,
  classifyAnswerLine,
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
const CHAT_STATUS_LABELS = {
  answer: "ตอบแล้ว",
  insufficient_evidence: "หลักฐานยังไม่เพียงพอ",
  clarification_required: "ต้องระบุข้อมูลเพิ่มเติม",
  error: "เกิดข้อผิดพลาด",
  provider_unavailable: "ระบบยังไม่พร้อมใช้งาน กรุณาลองใหม่",
  incomplete_evidence: "ข้อมูลยังไม่ครบถ้วน",
};

function loadStored() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed.sessions)) return null;
    return {...parsed, sessions: parsed.sessions.map(normalizeStoredSession).filter(Boolean)};
  } catch {
    return null;
  }
}

function AnswerText({ text }) {
  return (
    <div className="chat-answer-text">
      {String(text || "").split(/\r?\n/).map((line, index) => {
        const kind = classifyAnswerLine(line);
        return (
          <div className={`chat-answer-line answer-line-${kind}`} key={index}>
            {line || "\u00a0"}
          </div>
        );
      })}
    </div>
  );
}

function clarificationForMessage(session, message) {
  if (message?.clarification) {
    return {...message.clarification, question: message.question, messageId: message.id};
  }
  const legacy = session.pendingClarification;
  return legacy?.messageId === message?.id ? legacy : null;
}

function sessionForMessageClarification(session, message, clarification) {
  const requestScope = clarification?.request_scope;
  const hasHomeProgram = requestScope
    && Object.prototype.hasOwnProperty.call(requestScope, "home_program");
  const context = clarification?.next_context
    ?? requestScope?.conversation_context
    ?? session.context;
  const target = clarification?.clarification_target;
  const normalCatalog = clarification?.action === "catalog_required"
    && (target == null || normalCatalogTarget(target));
  const program = hasHomeProgram
    ? requestScope.home_program || (normalCatalog ? "" : context?.program || "")
    : context?.program || session.program;
  return {
    ...session,
    program,
    catalogKey: context?.catalog_key ?? (hasHomeProgram ? "" : session.catalogKey),
    plan: context?.plan ?? (hasHomeProgram ? null : session.plan),
    context: context ?? null,
    pendingClarification: {
      ...clarification,
      question: message.question,
      messageId: message.id,
    },
  };
}

function retryScopeForMessageClarification(clarification, retrySession) {
  const requestScope = clarification?.request_scope;
  const homeProgram = requestScope
    && Object.prototype.hasOwnProperty.call(requestScope, "home_program")
    ? requestScope.home_program
    : retrySession.program || null;
  const conversationContext = buildConversationContext(retrySession)
    ?? requestScope?.conversation_context
    ?? clarification?.next_context
    ?? null;
  return {home_program: homeProgram, conversation_context: conversationContext};
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
  const [inFlightCount, setInFlightCount] = useState(0);
  const loading = inFlightCount > 0;
  const [error, setError] = useState("");
  const threadRef = useRef(null);
  const submissionManagerRef = useRef(null);
  if (submissionManagerRef.current === null) {
    submissionManagerRef.current = createChatSubmissionManager();
  }
  const cooldownTimersRef = useRef(new Map());
  const inFlightClarificationIdsRef = useRef(new Set());
  const [inFlightClarificationIds, setInFlightClarificationIds] = useState(() => new Set());
  const [cooldownNow, setCooldownNow] = useState(() => Date.now());

  const active = useMemo(
    () => sessions.find((s) => s.id === activeId) || sessions[0],
    [sessions, activeId]
  );
  const activeSessionRef = useRef(null);
  activeSessionRef.current = active?.id;
  const cooldownRemainingMs = active
    ? submissionManagerRef.current.remaining(active.id, cooldownNow)
    : 0;

  useEffect(() => () => {
    for (const timer of cooldownTimersRef.current.values()) clearTimeout(timer);
    cooldownTimersRef.current.clear();
  }, []);

  useEffect(() => {
    fetchPrograms()
      .then((res) => setPrograms(res.programs || []))
      .catch(() => setPrograms([]));
  }, []);

  useEffect(() => {
    if (programs.length === 0) return;
    setSessions((previous) =>
      previous.map((session) => {
        if (!session.program) return session;
        const program = programs.find(
          (item) => item.program_code === session.program
        );
        const editions = program?.editions || [];
        const existing = session.catalogKey || "";
        const isAvailable = editions.some(
          (edition) => edition.catalog_key === existing
        );
        const catalogKey = isAvailable
          ? existing
          : defaultCatalogKey(session.program, programs);
        const scoped = {...session, catalogKey};
        const plans = availablePlans(scoped, programs);
        const priorPlan = session.plan;
        const plan = plans.some(p => p.plan_key === priorPlan)
          ? priorPlan : plans.length === 1 ? plans[0].plan_key : null;
        if (catalogKey === session.catalogKey && plan === session.plan) return session;
        const changedEdition = Boolean(existing && existing !== catalogKey);
        return {
          ...session,
          catalogKey,
          plan,
          pendingClarification: changedEdition || priorPlan !== plan ? null : session.pendingClarification,
          context: changedEdition || priorPlan !== plan
            ? resetContextForEdition(session.program, catalogKey, plan)
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
    const program = active?.program ?? "IT";
    const session = newSession(program, defaultCatalogKey(program, programs), programs);
    setSessions((prev) => [session, ...prev]);
    setActiveId(session.id);
    setQuestion("");
    setError("");
  }

  function handleDeleteSession(id) {
    setSessions((prev) => {
      const next = prev.filter((s) => s.id !== id);
      if (next.length === 0) {
        const program = active?.program ?? "IT";
        const fresh = newSession(program, defaultCatalogKey(program, programs), programs);
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
    updateActive(selectProgram(active, program, programs));
    setQuestion("");
    setError("");
  }

  function handleProgramClarificationChange(program, clarificationMessageId) {
    const message = active.messages.find(item => item.id === clarificationMessageId);
    const clarification = message ? clarificationForMessage(active, message) : null;
    if (!message || clarification?.action !== "program_required") return;
    if (inFlightClarificationIdsRef.current.has(message.id)) return;

    let retry;
    try {
      const selectionSession = sessionForMessageClarification(
        active, message, clarification,
      );
      retry = selectProgramForRetry(
        selectionSession, program, programs, clarification,
      );
    } catch (err) {
      setError(err.message);
      return;
    }
    setError("");
    setSessions(previous => updateMessageClarification(
      previous, active.id, message.id, retry.clarification,
    ));
    handleAsk(
      message.question,
      retry.session,
      message.id,
      null,
      retry.clarification.clarification_resolutions,
      retry.retryScope,
    );
  }

  function handleCatalogChange(catalogKey, clarificationMessageId = null) {
    const message = clarificationMessageId == null
      ? null : active.messages.find(item => item.id === clarificationMessageId);
    const clarification = message ? clarificationForMessage(active, message) : null;
    if (clarificationMessageId != null && (!message || !clarification)) return;
    if (message && inFlightClarificationIdsRef.current.has(message.id)) return;
    if (!message && loading) return;

    const selectionSession = message
      ? sessionForMessageClarification(active, message, clarification)
      : active;
    let retry;
    try {
      retry = message
        ? selectCatalogForRetry(selectionSession, catalogKey, programs, clarification)
        : {session: selectCatalog(active, catalogKey, programs), clarificationResolution: null};
    } catch (err) {
      setError(err.message);
      return;
    }
    setError("");
    if (!message) {
      updateActive(retry.session);
      setQuestion("");
      return;
    }

    const resolutions = retry.clarificationResolutions
      ?? clarification.clarification_resolutions
      ?? [];
    const updatedClarification = retry.clarification
      ?? {...clarification, clarification_resolutions: resolutions};
    const retryScope = retryScopeForMessageClarification(updatedClarification, retry.session);
    const retryClarification = {
      ...updatedClarification,
      next_context: retryScope.conversation_context,
      request_scope: retryScope,
      clarification_resolutions: resolutions,
    };
    if (!clarification.clarification_target?.operand) {
      setSessions(previous => previous.map(item => item.id === active.id
        ? {
          ...item,
          ...(item.program ? {catalogKey: retry.session.catalogKey, plan: retry.session.plan} : {}),
          context: retry.session.context ?? item.context,
        }
        : item));
    }
    setSessions(previous => updateMessageClarification(
      previous, active.id, message.id, retryClarification,
    ));
    handleAsk(message.question, retry.session, message.id,
      retry.clarificationResolution ?? null,
      retry.clarificationResolutions ?? resolutions,
      retryScope);
  }

  function handlePlanChange(plan, clarificationMessageId = null) {
    const message = clarificationMessageId == null
      ? null : active.messages.find(item => item.id === clarificationMessageId);
    const clarification = message ? clarificationForMessage(active, message) : null;
    if (clarificationMessageId != null && (!message || !clarification)) return;
    if (message && inFlightClarificationIdsRef.current.has(message.id)) return;
    if (!message && loading) return;

    const selectionSession = message
      ? sessionForMessageClarification(active, message, clarification)
      : active;
    let retry;
    try {
      retry = message
        ? selectPlanForRetry(selectionSession, plan, programs, clarification)
        : {session: selectPlan(active, plan, programs), clarificationResolution: null};
    } catch (err) {
      setError(err.message);
      return;
    }
    setError("");
    if (!message) {
      updateActive(retry.session);
      return;
    }

    const resolutions = retry.clarificationResolutions
      ?? clarification.clarification_resolutions
      ?? [];
    const updatedClarification = retry.clarification
      ?? {...clarification, clarification_resolutions: resolutions};
    const retryScope = retryScopeForMessageClarification(updatedClarification, retry.session);
    const retryClarification = {
      ...updatedClarification,
      next_context: retryScope.conversation_context,
      request_scope: retryScope,
      clarification_resolutions: resolutions,
    };
    if (!clarification.clarification_target?.operand) {
      setSessions(previous => previous.map(item => item.id === active.id
        ? {
          ...item,
          ...(item.program ? {plan: retry.session.plan, catalogKey: retry.session.catalogKey} : {}),
          context: retry.session.context ?? item.context,
        }
        : item));
    }
    setSessions(previous => updateMessageClarification(
      previous, active.id, message.id, retryClarification,
    ));
    handleAsk(message.question, retry.session, message.id,
      retry.clarificationResolution ?? null,
      retry.clarificationResolutions ?? resolutions,
      retryScope);
  }

  async function handleAsk(retryQuestion, selectedSession, retryId, clarificationResolution = null, clarificationResolutions = null, retryScopeOverride = null) {
    const session = selectedSession || active;
    const q = typeof retryQuestion === "string" ? retryQuestion : question.trim();
    if (!active) return;
    if (retryId != null && inFlightClarificationIdsRef.current.has(retryId)) return;
    if (q.length < 2) {
      setError("กรุณาพิมพ์คำถามอย่างน้อย 2 ตัวอักษร");
      return;
    }
    const requestContext = retryScopeOverride?.conversation_context;
    if (!retryScopeOverride) {
      const selectedProgram = programs.find(
        (item) => item.program_code === session.program
      );
      if ((selectedProgram?.editions?.length || 0) > 1 && !session.catalogKey) {
        setError("กรุณาเลือกปีหลักสูตรก่อนส่งคำถาม");
        return;
      }
    }

    const seed = retryScopeOverride
      ? requestContext
      : buildConversationContext(session);
    const homeProgram = retryScopeOverride
      ? retryScopeOverride.home_program
      : session.program || null;
    const submission = submissionManagerRef.current.begin(
      session.id,
      () => askQuestion(q, seed, homeProgram, clarificationResolution, clarificationResolutions),
      {bypassCooldown: Boolean(retryId)},
    );
    if (!submission.accepted) return;

    setError("");
    setInFlightCount(count => count + 1);
    if (retryId != null) {
      inFlightClarificationIdsRef.current.add(retryId);
      setInFlightClarificationIds(new Set(inFlightClarificationIdsRef.current));
    }
    if (!retryId) {
      setQuestion(current => current.trim() === q ? "" : current);
      setCooldownNow(Date.now());
      const previousTimer = cooldownTimersRef.current.get(session.id);
      if (previousTimer) clearTimeout(previousTimer);
      const timer = setTimeout(() => {
        cooldownTimersRef.current.delete(session.id);
        setCooldownNow(Date.now());
      }, Math.max(0, submission.availableAt - Date.now()));
      cooldownTimersRef.current.set(session.id, timer);
    }

    try {
      const requestStarted = performance.now();
      const data = await submission.promise;
      const elapsedMs = performance.now() - requestStarted;
      const entry = {
        id: retryId || submission.requestId,
        question: q,
        answer: data.answer || "ไม่พบคำตอบ",
        status: data.status,
        route: data.route,
        elapsedMs,
        provenance: data.provenance || [],
        planClarification: data.status === "clarification_required" && data.action === "plan_required",
        catalogClarification: data.status === "clarification_required" && data.action === "catalog_required",
        programClarification: data.status === "clarification_required" && data.action === "program_required",
        request_scope: {
          home_program: homeProgram,
          conversation_context: seed,
        },
      };
      setSessions(previous => applyChatResponse(previous, session.id, entry, data, retryId,
        clarificationResolutions || (clarificationResolution ? [clarificationResolution] : null)));
    } catch (err) {
      if (activeSessionRef.current === session.id) setError(err.message || "เกิดข้อผิดพลาด");
    } finally {
      setInFlightCount(count => Math.max(0, count - 1));
      if (retryId != null) {
        inFlightClarificationIdsRef.current.delete(retryId);
        setInFlightClarificationIds(new Set(inFlightClarificationIdsRef.current));
      }
    }
  }

  if (!active) return null;

  const selectedProgram = programs.find(
    (item) => item.program_code === active.program
  );
  const editions = selectedProgram?.editions || [];
  const selectedEdition = editions.find(
    (edition) => edition.catalog_key === active.catalogKey
  );
  const plans = availablePlans(active, programs);
  const visiblePlans = displayablePlans(plans);
  const planSegment = planLabel(active.plan);
  const scopeLabel = active.program
    ? `${active.program}${
        selectedEdition
          ? ` · ${selectedEdition.academic_year || selectedEdition.catalog_key}`
          : active.catalogKey
            ? ` · ${active.catalogKey}`
            : editions.length > 1
              ? " · เลือกฉบับหลักสูตร"
              : ""
      }${planSegment ? ` · ${planSegment}` : ""}`
    : "ทุกหลักสูตร";

  return (
    <div className="page chat-page">
      <div className="chat-layout">
        <aside className="card chat-context-panel" aria-label="ขอบเขตการค้นหา">
          <div className="chat-context-intro">
            <h1>ขอบเขตการค้นหา</h1>
            <p>คำตอบอ้างอิงข้อมูลตามรายการที่เลือก</p>
          </div>

          <div className="field chat-scope-field">
            <label htmlFor="chatProgram">หลักสูตร</label>
            <select
              id="chatProgram"
              value={active.program}
              disabled={loading}
              onChange={(e) => handleProgramChange(e.target.value)}
            >
              <option value="">ทุกหลักสูตร</option>
              {programs.map((p) => (
                <option key={p.program_code} value={p.program_code}>
                  {p.program_code}
                </option>
              ))}
            </select>
          </div>

          {editions.length > 0 && (
            <div className="field chat-scope-field">
              <label htmlFor="chatCatalog">ฉบับหลักสูตร</label>
              <select
                id="chatCatalog"
                value={active.catalogKey || ""}
                disabled={loading || editions.length === 1}
                onChange={(e) => handleCatalogChange(e.target.value)}
              >
                {editions.length > 1 && !active.catalogKey && (
                  <option value="" disabled>
                    เลือกฉบับหลักสูตร
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

          {visiblePlans.length > 0 && (
            <div className="field chat-scope-field">
              <label>แผนการเรียน</label>
              <PlanSelector plans={visiblePlans} value={active.plan} onChange={handlePlanChange} disabled={loading} />
            </div>
          )}

          <div className="chat-current-scope">
            <span>ขอบเขตปัจจุบัน</span>
            <strong>{scopeLabel}</strong>
          </div>

          <div className="chat-context-divider" />

          <button type="button" className="chat-new-session" onClick={handleNewChat}>
            <span aria-hidden="true">＋</span> แชตใหม่
          </button>
          <h2 className="chat-history-heading">ประวัติการสนทนา</h2>
          <ul className="chat-session-list">
            {sessions.map((s) => (
              <li key={s.id}>
                <button
                  type="button"
                  className={`chat-session-item${s.id === active.id ? " active" : ""}`}
                  onClick={() => {
                    setActiveId(s.id);
                    setQuestion("");
                    setError("");
                  }}
                  title={s.title}
                  aria-current={s.id === active.id ? "page" : undefined}
                >
                  <span className="chat-session-program">{s.program || "All"}</span>
                  <span className="chat-session-title">{s.title}</span>
                  <span className="chat-session-count" aria-label={`${s.messages.length} ข้อความ`}>
                    {s.messages.length}
                  </span>
                </button>
                {sessions.length > 1 && (
                  <button
                    type="button"
                    className="chat-session-delete"
                    aria-label={`ลบบทสนทนา ${s.title}`}
                    onClick={() => handleDeleteSession(s.id)}
                  >
                    ×
                  </button>
                )}
              </li>
            ))}
          </ul>
        </aside>

        <div className="chat-main">
          <header className="chat-product-header">
            <div>
              <h1>Curriculum Assistant</h1>
              <p>ถามเกี่ยวกับรายวิชา หลักสูตร และแผนการเรียน</p>
            </div>
            <div className="chat-header-scope">
              <span>ขอบเขตปัจจุบัน</span>
              <strong>{scopeLabel}</strong>
            </div>
          </header>

          <section className="card chat-thread-card" aria-label="บทสนทนา">
            <div className="chat-thread" ref={threadRef}>
              {active.messages.length === 0 && (
                <div className="chat-empty-state">
                  <span className="chat-empty-mark" aria-hidden="true">C</span>
                  <strong>เริ่มถามเกี่ยวกับหลักสูตร</strong>
                  <span>คำตอบจะแสดงพร้อมแหล่งอ้างอิงเมื่อมีข้อมูลรองรับ</span>
                </div>
              )}
              {active.messages.map((m) => {
                const clarification = clarificationForMessage(active, m);
                const clarificationSession = clarification
                  ? sessionForMessageClarification(active, m, clarification)
                  : active;
                const messagePlans = clarification
                  ? displayablePlans(clarificationPlans(clarificationSession, programs, clarification))
                  : [];
                const messageCatalogs = clarification
                  ? clarificationCatalogs(clarificationSession, programs, clarification)
                  : [];
                const retryInFlight = inFlightClarificationIds.has(m.id);
                const showPlanClarification = clarification?.action === "plan_required";
                const showCatalogClarification = clarification?.action === "catalog_required";
                const showProgramClarification = clarification?.action === "program_required";
                return (
                <article className="chat-turn" key={m.id}>
                  <div className="chat-user-row">
                    <div className="chat-user-message">{m.question}</div>
                  </div>
                  <div className="chat-assistant-row">
                    <div className="chat-answer-card">
                       <div className="chat-answer-label">
                         CUCUMBER
                         {formatElapsedTime(m.elapsedMs) && (
                           <span className="chat-answer-timing"> · {formatElapsedTime(m.elapsedMs)}</span>
                         )}
                       </div>
                       <AnswerText text={
                         typeof m.answer === "string" && m.answer.trim() && m.answer !== "ไม่พบคำตอบ"
                           ? m.answer
                           : m.planClarification ? "คำถามนี้ต้องระบุแผนการเรียนก่อน" : "ไม่พบคำตอบ"
                       } />
                            {showPlanClarification && (
                               <PlanSelector plans={messagePlans} value={clarification.clarification_target ? null : clarificationSession.plan} onChange={plan => handlePlanChange(plan, m.id)} disabled={retryInFlight} />
                            )}
                        {showCatalogClarification && messageCatalogs.length > 0 && (
                              <div className="plan-selector" role="group" aria-label="ฉบับหลักสูตรสำหรับคำถามนี้">
                                {messageCatalogs.map(edition => (
                                  <button key={edition.catalog_key} type="button" disabled={retryInFlight}
                                    onClick={() => handleCatalogChange(edition.catalog_key, m.id)}>
                                    {catalogOptionLabel(edition, messageCatalogs)}
                                  </button>
                                ))}
                          </div>
                        )}
                        {showProgramClarification && programs.length > 0 && (
                          <div className="plan-selector" role="group" aria-label="หลักสูตรสำหรับคำถามนี้">
                            {programs.map(program => (
                              <button key={program.program_code} type="button" disabled={retryInFlight}
                                onClick={() => handleProgramClarificationChange(program.program_code, m.id)}>
                                {program.program_code}
                              </button>
                            ))}
                          </div>
                        )}
                    </div>
                  </div>
                  <div className="chat-message-meta">
                    <span className={`chat-status-badge status-${m.status || "unknown"}`}>
                      {m.route === "hard"
                        ? HARD_STATUS_LABELS[m.status] || "ผลการตรวจสอบหลักสูตร"
                        : CHAT_STATUS_LABELS[m.status] || m.status}
                    </span>
                  </div>
                  {m.provenance.length > 0 && (
                    <details className="chat-provenance">
                      <summary>แหล่งอ้างอิง {m.provenance.length} รายการ</summary>
                      <ul>
                        {m.provenance.map((source, index) => (
                          <li key={`${source.source_filename || source.source_page || source.program || "source"}-${index}`}>
                             <span>{provenanceLabel(source)}</span>
                          </li>
                        ))}
                      </ul>
                    </details>
                  )}
                </article>
                );
              })}
              {loading && (
                <div className="chat-assistant-row" role="status" aria-live="polite">
                  <div className="chat-loading-card">
                    <span className="chat-loading-indicator" aria-hidden="true" />
                    <div>
                      <strong>กำลังค้นหาข้อมูลหลักสูตร...</strong>
                      <span>กำลังตรวจสอบข้อมูลและแหล่งอ้างอิง</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="chat-composer">
              <textarea
                aria-label="พิมพ์คำถาม"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") handleAsk();
                }}
                placeholder="พิมพ์คำถามได้เลย เช่น วิชา 06016454 กี่หน่วยกิต"
              />
              <div className="chat-composer-actions">
                <span className="chat-key-hint">กด Ctrl + Enter เพื่อส่งคำถาม</span>
                <button
                  type="button"
                  className="chat-send-button"
                  onClick={handleAsk}
                  disabled={cooldownRemainingMs > 0}
                >
                  {cooldownRemainingMs > 0
                    ? `ส่งได้ใน ${Math.ceil(cooldownRemainingMs / 1000)} วินาที`
                    : "ส่ง"}
                </button>
              </div>
              {error && <div className="chat-error" role="alert">{error}</div>}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

export function defaultCatalogKey(program, programs) {
  const selectedProgram = programs.find((item) => item.program_code === program);
  const editions = Array.isArray(selectedProgram?.editions)
    ? selectedProgram.editions.filter(
        (edition) => typeof edition.catalog_key === "string" && edition.catalog_key
      )
    : [];

  if (editions.length === 1) return editions[0].catalog_key;
  if (editions.length < 1) return "";

  const years = editions.map((edition) => Number(edition.academic_year));
  if (years.some((year) => !Number.isInteger(year))) return "";
  if (new Set(years).size !== years.length) return "";

  return editions[years.indexOf(Math.max(...years))].catalog_key;
}

export function availablePlans(session, programs) {
  return programs.find(p => p.program_code === session.program)?.editions
    ?.find(e => e.catalog_key === session.catalogKey)?.plans || [];
}

export function planLabel(plan) {
  // Internal-only plan keys (e.g. "default") have no user-facing label.
  // Canonical keys are unchanged; only display is affected.
  const labels = {coop: "สหกิจ", no_coop: "ไม่สหกิจ"};
  return labels[plan] ?? null;
}

export function displayablePlans(plans) {
  return (plans || []).filter((plan) => planLabel(plan?.plan_key) != null);
}

export function formatElapsedTime(elapsedMs) {
  if (!Number.isFinite(elapsedMs) || elapsedMs < 0) return null;
  const seconds = elapsedMs / 1000;
  return `${seconds.toFixed(seconds < 1 ? 2 : 1)} วินาที`;
}

export function provenanceLabel(source) {
  return [
    source.program ? `หลักสูตร ${source.program}` : null,
    source.source_page != null ? `หน้า ${source.source_page}` : null,
  ].filter(Boolean).join(" · ");
}

export function classifyAnswerLine(line) {
  const text = String(line).trim();
  if (!text) return "blank";
  if (/^(?:ช่องวิชาเลือกที่ยังไม่ระบุวิชาจริง|ช่องวิชาเลือก)/.test(text)) return "elective-slot";
  if (/^อ้างอิง\s*:/.test(text)) return "reference-note";
  if (/^หมายเหตุ\s*:/.test(text)) return "note";
  if (/^วิชาบังคับก่อน\s*:/.test(text)) return "prerequisite";
  if (/^(?:-|•)\s?/.test(text)) return "list-item";
  if (/^(?:ปี\s*\d+\s*เทอม\s*\d+|(?:ราย)?วิชา[^\n]*|พบรายวิชา[^\n]*):$/.test(text)) {
    return "section-heading";
  }
  return "paragraph";
}

export function selectPlan(session, plan, programs) {
  if (!availablePlans(session, programs).some(p => p.plan_key === plan)) {
    throw new Error("แผนการเรียนไม่ตรงกับฉบับหลักสูตรที่เลือก");
  }
  return {...session, plan, context: resetContextForEdition(session.program, session.catalogKey, plan), pendingClarification: null};
}

export function resetContextForEdition(program, catalogKey, plan = null) {
  return {
    ...(program ? { program } : {}),
    ...(catalogKey ? { catalog_key: catalogKey } : {}),
    ...(typeof plan === "string" && plan ? {plan} : {}),
  };
}

export function buildConversationContext(session) {
  if (!session) return null;
  const context = { ...(session.context || {}) };
  if (session.program && context.program && session.program !== context.program) {
    return resetContextForEdition(session.program, session.catalogKey, session.plan);
  }
  if (session.plan && context.plan && session.plan !== context.plan) {
    return resetContextForEdition(session.program, session.catalogKey, session.plan);
  }
  if (session.program) context.program = session.program;

  const selectedCatalog = session.catalogKey;
  if (selectedCatalog) {
    const resultHasOtherEdition = (context.result_courses || []).some(
      (course) =>
        course.catalog_key && course.catalog_key !== selectedCatalog
    );
    if (
      (context.catalog_key && context.catalog_key !== selectedCatalog) ||
      resultHasOtherEdition
    ) {
      return resetContextForEdition(session.program, selectedCatalog, session.plan);
    }
    context.catalog_key = selectedCatalog;
  }

  if (session.plan) context.plan = session.plan;

  return Object.keys(context).length > 0 ? context : null;
}

export function newSession(program = "", catalogKey = "", programs = []) {
  const plans = availablePlans({program, catalogKey}, programs);
  return {
    id: `s-${Date.now()}-${Math.floor(Math.random() * 1e6)}`,
    program,
    catalogKey: program ? catalogKey : "",
    plan: program && plans.length === 1 ? plans[0].plan_key : null,
    pendingClarification: null,
    title: "New chat",
    messages: [],
    context: null,
    createdAt: Date.now(),
  };
}

export function normalizeStoredSession(session) {
  if (!session || typeof session !== "object" || typeof session.id !== "string" || !session.id) return null;
  const validProgram = typeof session.program === "string";
  const program = validProgram ? session.program : "";
  return {
    ...session,
    program,
    catalogKey: program && typeof session.catalogKey === "string" ? session.catalogKey : "",
    plan: program && typeof session.plan === "string" ? session.plan : null,
    messages: Array.isArray(session.messages) ? session.messages : [],
    title: typeof session.title === "string" ? session.title : "New chat",
    context: validProgram && session.context && typeof session.context === "object" && !Array.isArray(session.context)
      ? session.context : null,
    pendingClarification: validProgram ? session.pendingClarification || null : null,
  };
}

export function selectProgram(session, program, programs) {
  const catalogKey = defaultCatalogKey(program, programs);
  const plans = availablePlans({program, catalogKey}, programs);
  const plan = plans.length === 1 ? plans[0].plan_key : null;
  return {...session, program, catalogKey, plan, pendingClarification: null,
    context: resetContextForEdition(program, catalogKey, plan)};
}

export function applyChatResponse(sessions, originId, entry, data, retryId) {
  return sessions.map(session => session.id !== originId ? session : ({
    ...session,
    messages: retryId ? session.messages.map(m => m.id === retryId ? entry : m) : [...session.messages, entry],
    pendingClarification: entry.planClarification ? {question: entry.question, messageId: entry.id} : null,
    context: buildConversationContext({...session, context: data.next_context}),
    title: session.messages.length === 0
      ? entry.question.length > 42 ? `${entry.question.slice(0, 42)}…` : entry.question
      : session.title,
  }));
}

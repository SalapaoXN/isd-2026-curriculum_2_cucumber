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

function operandTarget(target, dimension) {
  return target && typeof target === "object" && !Array.isArray(target)
    && Object.keys(target).length === 3
    && target.dimension === dimension
    && typeof target.program === "string" && target.program.trim()
    && ["left", "right"].includes(target.operand);
}

export function normalCatalogTarget(target) {
  return target && typeof target === "object" && !Array.isArray(target)
    && Object.keys(target).length === 3 && target.dimension === "catalog"
    && typeof target.program === "string" && target.program.trim()
    && target.operand === null;
}

export function clarificationPlans(session, programs) {
  const target = session.pendingClarification?.clarification_target;
  if (target == null) return availablePlans(session, programs);
  if (!operandTarget(target, "plan")) return [];
  const program = programs.find(p => p.program_code === target.program);
  const plans = program?.plans?.length ? program.plans
    : (program?.editions || []).flatMap(edition => edition.plans || []);
  return [...new Map(plans.map(plan => [plan.plan_key, plan])).values()];
}

export function selectPlanForRetry(session, plan, programs) {
  const target = session.pendingClarification?.clarification_target;
  if (target == null) return {session: selectPlan(session, plan, programs), clarificationResolution: null};
  if (!operandTarget(target, "plan") || !clarificationPlans(session, programs).some(p => p.plan_key === plan)) {
    throw new Error("ไม่สามารถระบุแผนของฝั่งเปรียบเทียบได้อย่างปลอดภัย");
  }
  return accumulatedRetry(session, {...target, value: plan});
}

function accumulatedRetry(session, resolution) {
  const previous = session.pendingClarification?.clarification_resolutions ?? [];
  if (!Array.isArray(previous) || previous.length > 4) {
    throw new Error("ข้อมูลการระบุขอบเขตต่อเนื่องไม่ถูกต้อง");
  }
  const unique = new Map();
  for (const item of [...previous, resolution]) {
    if (!item || typeof item !== "object" || Array.isArray(item)
      || Object.keys(item).length !== 4 || !["catalog", "plan"].includes(item.dimension)
      || !["left", "right"].includes(item.operand)
      || ![item.program, item.value].every(value => typeof value === "string" && value.trim() && value.length <= 80)) {
      throw new Error("ข้อมูลการระบุขอบเขตต่อเนื่องไม่ถูกต้อง");
    }
    const key = `${item.operand}:${item.dimension}`;
    const prior = unique.get(key);
    if (prior && (prior.program !== item.program || prior.value !== item.value)) {
      throw new Error("ข้อมูลการระบุขอบเขตต่อเนื่องขัดแย้งกัน");
    }
    unique.set(key, {...item});
  }
  if (unique.size > 4) throw new Error("ข้อมูลการระบุขอบเขตเกินขอบเขตที่รองรับ");
  const history = [...unique.values()];
  return {session: {...session, pendingClarification: {
      ...session.pendingClarification, clarification_resolutions: history.map(item => ({...item})),
    }}, clarificationResolution: resolution, clarificationResolutions: history};
}

export function clarificationCatalogs(session, programs) {
  const target = session.pendingClarification?.clarification_target;
  if (target != null && !operandTarget(target, "catalog") && !normalCatalogTarget(target)) return [];
  const program = target?.program || session.program;
  return (programs.find(p => p.program_code === program)?.editions || [])
    .filter(edition => typeof edition.catalog_key === "string" && edition.catalog_key);
}

export function catalogOptionLabel(edition, editions) {
  const label = edition.academic_year || edition.catalog_key;
  return editions.filter(item => item.academic_year === edition.academic_year).length > 1
    ? `${label} (${edition.catalog_key})` : label;
}

export function selectCatalog(session, catalogKey, programs) {
  const editions = programs.find(p => p.program_code === session.program)?.editions || [];
  if (!editions.some(edition => edition.catalog_key === catalogKey)) {
    throw new Error("ฉบับหลักสูตรไม่ตรงกับหลักสูตรที่เลือก");
  }
  const plans = availablePlans({...session, catalogKey}, programs);
  const plan = plans.some(p => p.plan_key === session.plan)
    ? session.plan : plans.length === 1 ? plans[0].plan_key : null;
  return {...session, catalogKey, plan, pendingClarification: null,
    context: resetContextForEdition(session.program, catalogKey, plan)};
}

export function selectCatalogForRetry(session, catalogKey, programs) {
  const target = session.pendingClarification?.clarification_target;
  if (target == null) return {session: selectCatalog(session, catalogKey, programs), clarificationResolution: null};
  if (normalCatalogTarget(target)) {
    if (session.program && session.program !== target.program) {
      throw new Error("หลักสูตรที่ต้องการระบุฉบับไม่ตรงกับหลักสูตรประจำแชท");
    }
    if (session.program) {
      return {session: selectCatalog(session, catalogKey, programs), clarificationResolution: null};
    }
    // Unscoped home stays unscoped; the explicit question owns normal context.
    const sameProgram = session.context?.program === target.program;
    const normal = selectCatalog({...session, program: target.program,
      catalogKey: sameProgram ? session.context.catalog_key || "" : "",
      plan: sameProgram ? session.context.plan || null : null,
    }, catalogKey, programs);
    return {session: {...session, context: normal.context, pendingClarification: null},
      clarificationResolution: null};
  }
  if (!operandTarget(target, "catalog") || !clarificationCatalogs(session, programs).some(e => e.catalog_key === catalogKey)) {
    throw new Error("ไม่สามารถระบุฉบับหลักสูตรของฝั่งเปรียบเทียบได้อย่างปลอดภัย");
  }
  return accumulatedRetry(session, {...target, value: catalogKey});
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

export function applyChatResponse(sessions, originId, entry, data, retryId, clarificationResolutions = null) {
  return sessions.map(session => {
    if (session.id !== originId) return session;
    const sameQuestion = retryId && entry.id === retryId
      && session.messages.some(message => message.id === retryId && message.question === entry.question);
    const previous = session.pendingClarification;
    const history = sameQuestion
      ? clarificationResolutions ?? (previous?.messageId === retryId && previous.question === entry.question
        ? previous.clarification_resolutions : null)
      : null;
    return {
    ...session,
    messages: retryId ? session.messages.map(m => m.id === retryId ? entry : m) : [...session.messages, entry],
    pendingClarification: entry.planClarification || entry.catalogClarification ? {
      question: entry.question, messageId: entry.id,
      ...(data.clarification_target != null ? {clarification_target: data.clarification_target} : {}),
      ...(data.clarification_target != null && Array.isArray(history) && history.length
        ? {clarification_resolutions: history.map(item => ({...item}))} : {}),
    } : sameQuestion && previous?.messageId === retryId && (data.status || entry.status) !== "answer"
      && Array.isArray(history) && history.length ? previous : null,
    context: buildConversationContext({...session, context: data.next_context}),
    title: session.messages.length === 0
      ? entry.question.length > 42 ? `${entry.question.slice(0, 42)}…` : entry.question
      : session.title,
    };
  });
}

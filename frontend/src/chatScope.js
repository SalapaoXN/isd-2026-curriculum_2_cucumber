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

const STORED_CONTEXT_STRINGS = {
  program: 80, catalog_key: 128, plan: 80, category: 80, course_code: 80,
  result_scope_program: 80, focus_catalog_key: 128, semantic_topic: 80,
};

function cleanStoredString(value, maxLength) {
  return typeof value === "string" && value.trim() && value.length <= maxLength
    ? value : null;
}

function cleanStoredIntList(value, maxValue, maxLength) {
  if (!Array.isArray(value)) return null;
  const cleaned = value.filter(item => Number.isInteger(item) && item >= 1 && item <= maxValue).slice(0, maxLength);
  return cleaned;
}

function cleanStoredCourseRef(item) {
  if (!item || typeof item !== "object" || Array.isArray(item)) return null;
  const code = cleanStoredString(item.course_code, 64);
  if (!code) return null;
  const ref = {course_code: code};
  const program = cleanStoredString(item.program, 80);
  if (program) ref.program = program;
  const catalogKey = cleanStoredString(item.catalog_key, 128);
  if (catalogKey) ref.catalog_key = catalogKey;
  const name = cleanStoredString(item.course_name, 160);
  if (name) ref.course_name = name;
  return ref;
}

export function normalizeStoredContext(context) {
  if (!context || typeof context !== "object" || Array.isArray(context)) return null;
  const cleaned = {};
  for (const [key, maxLength] of Object.entries(STORED_CONTEXT_STRINGS)) {
    const value = cleanStoredString(context[key], maxLength);
    if (value) cleaned[key] = value;
  }
  for (const key of ["years", "semesters"]) {
    if (context[key] !== undefined) cleaned[key] = cleanStoredIntList(context[key], key === "years" ? 5 : 2, key === "years" ? 6 : 3) || [];
  }
  for (const key of ["year", "semester"]) {
    if (Number.isInteger(context[key]) && context[key] >= 1 && context[key] <= (key === "year" ? 5 : 2)) cleaned[key] = context[key];
  }
  if (context.plans !== undefined) {
    const plans = Array.isArray(context.plans) ? context.plans : [context.plans];
    const cleanedPlans = plans.filter(item => cleanStoredString(item, 80));
    if (cleanedPlans.length) cleaned.plans = cleanedPlans.slice(0, 1);
  }
  if (context.operations !== undefined && Array.isArray(context.operations)) {
    const operations = context.operations.filter(item => cleanStoredString(item, 80));
    if (operations.length) cleaned.operations = operations.slice(0, 8);
  }
  const focus = cleanStoredCourseRef(context.focus_course);
  if (focus) cleaned.focus_course = focus;
  if (Array.isArray(context.result_courses)) {
    const courses = context.result_courses.slice(0, 50).map(cleanStoredCourseRef).filter(Boolean);
    if (courses.length || context.result_set_empty === true) cleaned.result_courses = courses;
  }
  if (typeof context.result_set_empty === "boolean" && context.result_set_empty) {
    cleaned.result_set_empty = true;
  }
  if (typeof context.pending_catalog_selection === "boolean") {
    cleaned.pending_catalog_selection = context.pending_catalog_selection;
  }
  const operation = context.last_normal_operation;
  if (operation && typeof operation === "object" && !Array.isArray(operation)
      && operation.kind === "list_courses" && cleanStoredString(operation.program, 80)) {
    const normalized = {kind: "list_courses", program: operation.program, catalog_key: null, plan: null};
    const catalogKey = cleanStoredString(operation.catalog_key, 128);
    if (catalogKey) normalized.catalog_key = catalogKey;
    const plan = cleanStoredString(operation.plan, 80);
    if (plan) normalized.plan = plan;
    normalized.years = cleanStoredIntList(operation.years, 5, 6) || [];
    normalized.semesters = cleanStoredIntList(operation.semesters, 2, 3) || [];
    cleaned.last_normal_operation = normalized;
  }
  const studyPlan = context.study_plan_context;
  const last = context.last_answer;
  if (last && typeof last === "object" && !Array.isArray(last)) {
    const ref = {};
    for (const [key, bound] of Object.entries({program: 80, catalog_key: 128})) {
      const value = cleanStoredString(last[key], bound);
      if (value) ref[key] = value;
    }
    if (last.route === "course" && cleanStoredString(last.course_code, 80) && Array.isArray(last.operations)) {
      const operations = last.operations.filter(op => ["sum_credits", "placement", "describe", "prerequisite", "existence"].includes(op)).slice(0, 8);
      if (operations.length) cleaned.last_answer = {...ref, route: "course", course_code: last.course_code, operations};
    } else if (last.route === "policy" && cleanStoredString(last.policy_kind, 80)) {
      const policy = {...ref, route: "policy", policy_kind: last.policy_kind};
      if (cleanStoredString(last.plan, 80)) policy.plan = last.plan;
      if (Number.isSafeInteger(last.amount) && last.amount >= 0) policy.amount = last.amount;
      if (Array.isArray(last.evidence_ids)) policy.evidence_ids = last.evidence_ids.filter(id => cleanStoredString(id, 80)).slice(0, 50);
      cleaned.last_answer = policy;
    }
  }
  if (studyPlan && typeof studyPlan === "object" && !Array.isArray(studyPlan)
      && studyPlan.kind === "seven_term_plan"
      && cleanStoredString(studyPlan.program, 80)
      && cleanStoredString(studyPlan.catalog_key, 128)
      && cleanStoredString(studyPlan.plan, 80)) {
    cleaned.study_plan_context = {
      kind: "seven_term_plan", program: studyPlan.program,
      catalog_key: studyPlan.catalog_key, plan: studyPlan.plan,
    };
  }
  return Object.keys(cleaned).length > 0 ? cleaned : null;
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
      ? normalizeStoredContext(session.context) : null,
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

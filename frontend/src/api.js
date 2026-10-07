export async function fetchHealth() {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return res.json();
}

export async function askQuestion(question, conversationContext = null, homeProgram = null) {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      conversation_context: conversationContext,
      home_program: homeProgram,
    }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Request failed (${res.status})`);
  }
  return data;
}

export async function fetchPrograms() {
  const res = await fetch("/api/programs");
  if (!res.ok) throw new Error(`Failed to load programs (${res.status})`);
  return res.json();
}

export function buildCurriculumQuery(params) {
  const q = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== null && value !== undefined && value !== "") {
      q.set(key, String(value));
    }
  }
  return q.toString();
}

export async function fetchCurriculum(params) {
  const qs = buildCurriculumQuery(params);
  const res = await fetch(`/api/curriculum?${qs}`);
  if (!res.ok) throw new Error(`Failed to load curriculum (${res.status})`);
  return res.json();
}

export async function fetchCourseDetail(
  courseCode,
  program = null,
  catalogKey = null
) {
  const params = new URLSearchParams();
  if (program) params.set("program", program);
  if (catalogKey) params.set("catalog_key", catalogKey);
  const query = params.toString();
  const qs = query ? `?${query}` : "";
  const res = await fetch(
    `/api/courses/${encodeURIComponent(courseCode)}${qs}`
  );
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `Course not found (${res.status})`);
  }
  return res.json();
}

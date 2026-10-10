import { useEffect, useState } from "react";
import {
  ArrowLeft, ArrowRight, BookOpen, BookOpenCheck, ChevronDown, ChevronLeft,
  ChevronRight, Layers3, LoaderCircle, Search, X,
} from "lucide-react";

const PAGE_SIZE = 50;
const typeLabels = { plan: "แผนการเรียน", description: "คำอธิบายรายวิชา" };

function SelectFilter({ label, value, options, onChange, placeholder, formatOption = option => option }) {
  return <label className="curriculum-filter">
    <span>{label}</span>
    <span className="curriculum-select-wrap"><select value={value} onChange={event => onChange(event.target.value)}>
      <option value="">{placeholder}</option>
      {options.map(option => <option key={option} value={option}>{formatOption(option)}</option>)}
    </select><ChevronDown size={14} /></span>
  </label>;
}

export default function CurriculumPage() {
  const [options, setOptions] = useState({ programs: [], academic_years: [], plans: [], categories: [], course_types: [], study_years: [], semesters: [] });
  const [filters, setFilters] = useState({ search: "", program: "", academic_year: "", plan: "", category: "", course_type: "", study_year: "", semester: "" });
  const [offset, setOffset] = useState(0);
  const [result, setResult] = useState({ items: [], total: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reload, setReload] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const params = new URLSearchParams(Object.entries(filters).filter(([, value]) => value));
    fetch(`/api/curriculum/filters?${params}`).then(response => {
      if (!response.ok) throw new Error("Could not load curriculum filters");
      return response.json();
    }).then(data => { if (!cancelled) setOptions(data); })
      .catch(reason => { if (!cancelled) setError(reason.message); });
    return () => { cancelled = true; };
  }, [filters]);

  useEffect(() => {
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      setLoading(true);
      setError("");
      const params = new URLSearchParams();
      for (const [key, value] of Object.entries(filters)) if (value) params.set(key, value);
      params.set("limit", String(PAGE_SIZE));
      params.set("offset", String(offset));
      try {
        const response = await fetch(`/api/curriculum?${params}`, { signal: controller.signal });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || "Could not load curriculum subjects");
        setResult(data);
      } catch (reason) {
        if (reason.name !== "AbortError") setError(reason.message);
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }, filters.search ? 220 : 0);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [filters, offset, reload]);

  const page = Math.floor(offset / PAGE_SIZE) + 1;
  const pageCount = Math.max(1, Math.ceil(result.total / PAGE_SIZE));
  const hasFilters = Object.entries(filters).some(([key, value]) => key !== "search" && value);
  const plans = options.plans || [];

  function setFilter(key, value) {
    const reset = {
      program: ["academic_year", "plan", "category", "course_type", "study_year", "semester"],
      academic_year: ["plan", "category", "course_type", "study_year", "semester"],
      plan: ["study_year", "semester"],
      study_year: ["semester"],
    }[key] || [];
    setFilters(previous => ({ ...previous, [key]: value, ...Object.fromEntries(reset.map(name => [name, ""])) }));
    setOffset(0);
  }

  function clearFilters() {
    setFilters({ search: "", program: "", academic_year: "", plan: "", category: "", course_type: "", study_year: "", semester: "" });
    setOffset(0);
  }

  return <main className="curriculum-page">
    <header className="curriculum-topbar">
      <a className="curriculum-brand" href="/"><span className="brand-mark">C</span><span><strong>cucumber</strong><small>CURRICULUM ASSISTANT</small></span></a>
      <nav className="curriculum-topnav" aria-label="Main navigation">
        <a href="/">Ask Cucumber</a><a className="active" href="/curriculum">Curriculum document</a>
      </nav>
      <a className="back-to-chat" href="/"><ArrowLeft size={15} /> Back to chat</a>
    </header>

    <section className="curriculum-main">
      <div className="curriculum-title-row">
        <div><div className="curriculum-eyebrow"><BookOpenCheck size={14} /> CURRICULUM LIBRARY</div>
          <h1>Curriculum document</h1>
          <p>Browse indexed subjects, course details, and study-plan placements.</p>
        </div>
        <div className="curriculum-stat"><span><Layers3 size={16} /></span><div><strong>{result.total.toLocaleString()}</strong><small>subjects in index</small></div></div>
      </div>

      <section className="curriculum-filter-panel" aria-label="Subject filters">
        <div className="curriculum-filter-topline"><strong>Filter subjects</strong>
          {hasFilters && <button className="clear-filters" onClick={clearFilters}><X size={13} /> Clear filters</button>}
        </div>
        <div className="curriculum-filter-grid">
          <label className="curriculum-filter search-filter"><span>Search</span><span className="curriculum-search-wrap"><Search size={16} /><input value={filters.search} onChange={event => setFilter("search", event.target.value)} placeholder="Code, course name, description" /></span></label>
          <SelectFilter label="Program" value={filters.program} options={options.programs || []} placeholder="All programs" onChange={value => setFilter("program", value)} />
          <SelectFilter label="Curriculum edition year" value={filters.academic_year} options={options.academic_years || []} placeholder="All edition years" onChange={value => setFilter("academic_year", value)} />
          <SelectFilter label="Study plan" value={filters.plan} options={plans} placeholder="All plans" onChange={value => setFilter("plan", value)} />
          <SelectFilter label="Study year" value={filters.study_year} options={options.study_years || []} placeholder="All study years" formatOption={year => `Year ${year}`} onChange={value => setFilter("study_year", value)} />
          <SelectFilter label="Subject category" value={filters.category} options={options.categories || []} placeholder="All categories" onChange={value => setFilter("category", value)} />
          <SelectFilter label="Course type" value={filters.course_type} options={options.course_types || []} placeholder="All types" onChange={value => setFilter("course_type", value)} />
          <SelectFilter label="Semester" value={filters.semester} options={options.semesters || []} placeholder="All semesters" onChange={value => setFilter("semester", value)} />
        </div>
      </section>

      <div className="curriculum-results-head"><div><strong>Subjects</strong><span>{loading ? "Loading…" : `${result.total.toLocaleString()} results`}</span></div>
        <span className="results-page-label">Page {page} of {pageCount}</span></div>
      {error && <div className="curriculum-error" role="alert">{error}<button onClick={() => setReload(value => value + 1)}>Retry</button></div>}
      {loading ? <div className="curriculum-loading"><LoaderCircle size={20} /> Loading indexed subjects…</div>
        : result.items.length === 0 ? <div className="curriculum-empty"><BookOpen size={24} /><strong>No subjects found</strong><span>Try changing your filters or search terms.</span></div>
          : <div className="subject-list">{result.items.map(item => <article className="subject-card" key={`${item.course_id}-${item.program_code}`}>
            <div className="subject-card-top"><div className="subject-code">{item.course_code}</div><div className="subject-tags">
              <span>{item.program_code}</span><span>{item.academic_year || item.catalog_key}</span>
              {item.course_type && <span className="tag-type">{item.course_type}</span>}
            </div></div>
            <div className="subject-title-row"><div><h2>{item.name_en || item.name_th || "Untitled course"}</h2>{item.name_th && item.name_en && <p className="thai-course-name">{item.name_th}</p>}</div>
              <strong className="subject-credits">{item.credits || "—"}<small>credits</small></strong>
            </div>
            <div className="subject-meta">
              {item.category && <span><small>Category</small>{item.category}</span>}
              {item.plans?.length > 0 && <span><small>Study plan</small>{item.plans.join(" · ")}</span>}
              {item.placements?.length > 0 && <span><small>Placement</small>{item.placements.slice(0, 3).join(" · ")}{item.placements.length > 3 ? ` +${item.placements.length - 3}` : ""}</span>}
            </div>
            {(item.description_en || item.description_th || item.prerequisite_text || item.sources?.length > 0) && <details className="subject-details">
              <summary>Course details <ChevronDown size={14} /></summary>
              {item.prerequisite_text && <p><strong>Prerequisites:</strong> {item.prerequisite_text}</p>}
              {item.description_en && <p>{item.description_en}</p>}
              {item.description_th && <p>{item.description_th}</p>}
              {item.sources?.length > 0 && <small className="subject-source">Source: {item.sources.join("; ")}</small>}
            </details>}
            {item.document_types?.length > 0 && <div className="source-type-tags">{item.document_types.map(type => <span key={type}>{typeLabels[type] || type}</span>)}</div>}
          </article>)}</div>}

      <div className="curriculum-pagination"><span>Showing {result.items.length ? offset + 1 : 0}–{Math.min(offset + PAGE_SIZE, result.total)} of {result.total.toLocaleString()}</span>
        <div><button disabled={offset === 0 || loading} onClick={() => setOffset(value => Math.max(0, value - PAGE_SIZE))}><ChevronLeft size={16} /> Previous</button>
          <button disabled={offset + PAGE_SIZE >= result.total || loading} onClick={() => setOffset(value => value + PAGE_SIZE)}>Next <ChevronRight size={16} /></button></div>
      </div>
      <footer className="curriculum-footer"><span>Course data from the indexed curriculum documents</span><a href="/"><ArrowRight size={14} /> Ask Cucumber</a></footer>
    </section>
  </main>;
}

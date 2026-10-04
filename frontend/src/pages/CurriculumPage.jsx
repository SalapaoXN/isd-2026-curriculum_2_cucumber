import { useEffect, useMemo, useRef, useState } from "react";
import { fetchCourseDetail, fetchCurriculum, fetchPrograms } from "../api";

const PAGE_SIZE = 50;

export default function CurriculumPage() {
  const [programs, setPrograms] = useState([]);
  const [filters, setFilters] = useState({
    program: "IT",
    catalogKey: "",
    plan: "",
    year: "",
    semester: "",
    search: "",
  });
  const [applied, setApplied] = useState({
    program: "IT",
    limit: PAGE_SIZE,
    offset: 0,
  });
  const [data, setData] = useState({ items: [], total: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const detailRequestId = useRef(0);
  const [programsReady, setProgramsReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchPrograms()
      .then((res) => {
        if (cancelled) return;
        const availablePrograms = res.programs || [];
        setPrograms(availablePrograms);

        const initialProgram = availablePrograms.find(
          (program) => program.program_code === filters.program
        );
        const initialEditions = initialProgram?.editions || [];
        const latestEdition = initialEditions[initialEditions.length - 1];
        const catalogKey = latestEdition?.catalog_key || "";
        const params = {
          program: filters.program || undefined,
          catalog_key: catalogKey || undefined,
          limit: PAGE_SIZE,
          offset: 0,
        };

        setFilters((current) => ({ ...current, catalogKey }));
        setApplied(params);
        setProgramsReady(true);
        load(params);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err.message || "โหลดรายการหลักสูตรไม่สำเร็จ");
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // The initial request is deliberately scoped after program editions load.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const plans = useMemo(() => {
    const found = programs.find((p) => p.program_code === filters.program);
    const edition = found?.editions?.find(
      (item) => item.catalog_key === filters.catalogKey
    );
    return edition?.plans || found?.plans || [];
  }, [programs, filters.program, filters.catalogKey]);

  const editions = useMemo(() => {
    const found = programs.find((p) => p.program_code === filters.program);
    return found?.editions || [];
  }, [programs, filters.program]);

  function clearSelection() {
    detailRequestId.current += 1;
    setSelected(null);
    setDetail(null);
    setDetailLoading(false);
  }

  function handleProgramChange(program) {
    const nextProgram = programs.find((item) => item.program_code === program);
    const nextEditions = nextProgram?.editions || [];
    const defaultEdition = nextEditions[nextEditions.length - 1];
    const nextPlans = defaultEdition?.plans || nextProgram?.plans || [];
    const plan = nextPlans.some((item) => item.plan_key === filters.plan)
      ? filters.plan
      : "";

    setFilters({
      ...filters,
      program,
      catalogKey: defaultEdition?.catalog_key || "",
      plan,
    });
    clearSelection();
  }

  function handleEditionChange(catalogKey) {
    const edition = editions.find((item) => item.catalog_key === catalogKey);
    if (!edition) return;
    const selectedProgram = programs.find(
      (item) => item.program_code === filters.program
    );
    const nextPlans = edition.plans || selectedProgram?.plans || [];
    const plan = nextPlans.some((item) => item.plan_key === filters.plan)
      ? filters.plan
      : "";
    setFilters({ ...filters, catalogKey, plan });
    clearSelection();
  }

  async function load(params) {
    setLoading(true);
    setError("");
    try {
      const res = await fetchCurriculum(params);
      setData(res);
    } catch (err) {
      setError(err.message || "โหลดข้อมูลไม่สำเร็จ");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load(applied);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleSearch(e) {
    e?.preventDefault();
    const params = {
      program: filters.program || undefined,
      catalog_key: filters.catalogKey || undefined,
      plan: filters.plan || undefined,
      year: filters.year || undefined,
      semester: filters.semester || undefined,
      search: filters.search || undefined,
      limit: PAGE_SIZE,
      offset: 0,
    };
    setApplied(params);
    load(params);
  }

  function handlePage(delta) {
    const offset = Math.max(0, (applied.offset || 0) + delta * PAGE_SIZE);
    const params = { ...applied, offset };
    setApplied(params);
    load(params);
  }

  async function handleSelect(item) {
    const requestId = detailRequestId.current + 1;
    detailRequestId.current = requestId;
    const catalogKey = item.catalog_key || applied.catalog_key || null;
    setSelected(item.course_code);
    setDetail(null);
    setDetailLoading(true);
    try {
      const res = await fetchCourseDetail(
        item.course_code,
        item.program,
        catalogKey
      );
      if (detailRequestId.current === requestId) setDetail(res);
    } catch {
      if (detailRequestId.current === requestId) setDetail(null);
    } finally {
      if (detailRequestId.current === requestId) setDetailLoading(false);
    }
  }

  const page = Math.floor((applied.offset || 0) / PAGE_SIZE) + 1;
  const pageCount = Math.max(1, Math.ceil((data.total || 0) / PAGE_SIZE));

  return (
    <div className="page curriculum-page">
      <header className="page-header curriculum-page-header">
        <h1>Curriculum document</h1>
        <p>ค้นหารายวิชาตามหลักสูตร ปี และเทอม จากฐานข้อมูลทางการ</p>
      </header>

      <form className="card filter-card curriculum-filter-card" onSubmit={handleSearch}>
        <div className="filter-grid curriculum-filter-grid">
          <div className="field">
            <label htmlFor="curriculumProgram">หลักสูตร</label>
            <select
              id="curriculumProgram"
              value={filters.program}
              disabled={!programsReady}
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
          <div className="field">
            <label htmlFor="curriculumEdition">Curriculum Edition</label>
            <select
              id="curriculumEdition"
              value={filters.catalogKey}
              disabled={!programsReady || !filters.program || editions.length <= 1}
              onChange={(e) => handleEditionChange(e.target.value)}
            >
              {!filters.program && (
                <option value="">เลือกหลักสูตรก่อน</option>
              )}
              {filters.program && editions.length === 0 && (
                <option value="">ไม่มีฉบับหลักสูตร</option>
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
          <div className="field">
            <label htmlFor="curriculumPlan">แผนการเรียน</label>
            <select
              id="curriculumPlan"
              value={filters.plan}
              disabled={!programsReady || plans.length === 0}
              onChange={(e) => setFilters({ ...filters, plan: e.target.value })}
            >
              <option value="">All</option>
              {plans.map((pl) => (
                <option key={pl.plan_key} value={pl.plan_key}>
                  {pl.plan_name || pl.plan_key}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="curriculumYear">ชั้นปี</label>
            <select
              id="curriculumYear"
              value={filters.year}
              onChange={(e) => setFilters({ ...filters, year: e.target.value })}
            >
              <option value="">All</option>
              {[1, 2, 3, 4].map((y) => (
                <option key={y} value={y}>
                  Year {y}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="curriculumSemester">ภาคเรียน</label>
            <select
              id="curriculumSemester"
              value={filters.semester}
              onChange={(e) =>
                setFilters({ ...filters, semester: e.target.value })
              }
            >
              <option value="">All</option>
              {[1, 2].map((s) => (
                <option key={s} value={s}>
                  Semester {s}
                </option>
              ))}
            </select>
          </div>
          <div className="field field-wide">
            <label htmlFor="curriculumSearch">ค้นหารายวิชา</label>
            <input
              id="curriculumSearch"
              type="search"
              value={filters.search}
              onChange={(e) =>
                setFilters({ ...filters, search: e.target.value })
              }
              placeholder="รหัสวิชา หรือชื่อวิชา เช่น 06016454"
            />
          </div>
        </div>
        <div className="actions curriculum-filter-actions">
          <button type="submit" disabled={loading || !programsReady}>
            {loading ? "กำลังโหลด..." : "ค้นหา"}
          </button>
          <div className="curriculum-result-summary" aria-live="polite">
            <span>พบ <strong>{data.total || 0}</strong> รายวิชา</span>
            <span>หน้า <strong>{page}</strong> / {pageCount}</span>
          </div>
        </div>
      </form>

      {error && <div className="error visible curriculum-error" role="alert">{error}</div>}

      <div className="curriculum-layout">
        <section className="card table-card curriculum-table-card" aria-label="ผลการค้นหารายวิชา">
          <div className="table-wrap">
            <table className="rows-table curriculum-table">
              <thead>
                <tr>
                  <th>หลักสูตร</th>
                  <th>รหัสวิชา</th>
                  <th>ชื่อวิชา</th>
                  <th>หน่วยกิต</th>
                  <th>Y/S</th>
                  <th>แผน</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((item, idx) => (
                  <tr
                    key={`${item.course_code}-${item.plan_key}-${idx}`}
                    className={selected === item.course_code ? "selected" : ""}
                    onClick={() => handleSelect(item)}
                  >
                    <td className="mono">{item.program || "-"}</td>
                    <td className="mono curriculum-course-code">{item.course_code}</td>
                    <td className="curriculum-course-name">{item.name_en || item.name_th || "-"}</td>
                    <td className="curriculum-compact-cell">{item.credits || item.credit_units || "-"}</td>
                    <td className="curriculum-compact-cell">
                      {item.year ?? "-"} / {item.semester ?? "-"}
                    </td>
                    <td className="curriculum-plan-cell">{item.plan_key || "-"}</td>
                  </tr>
                ))}
                {data.items.length === 0 && loading && (
                  <tr>
                    <td colSpan={6} className="empty curriculum-table-state">
                      กำลังโหลด...
                    </td>
                  </tr>
                )}
                {data.items.length === 0 && !loading && (
                  <tr>
                    <td colSpan={6} className="empty curriculum-table-state">
                      ไม่พบรายวิชาตามเงื่อนไข
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="actions pager curriculum-pager">
            <button
              type="button"
              className="ghost"
              disabled={(applied.offset || 0) === 0 || loading}
              onClick={() => handlePage(-1)}
            >
              ← Prev
            </button>
            <button
              type="button"
              className="ghost"
              disabled={
                (applied.offset || 0) + PAGE_SIZE >= (data.total || 0) ||
                loading
              }
              onClick={() => handlePage(1)}
            >
              Next →
            </button>
            <span className="curriculum-pager-label">หน้า {page} / {pageCount}</span>
          </div>
        </section>

        <aside className="card detail-card curriculum-detail-card" aria-label="รายละเอียดรายวิชา">
          <h2 className="section-title">รายละเอียดรายวิชา</h2>
          {detailLoading && <div className="empty curriculum-detail-state">กำลังโหลด...</div>}
          {!detailLoading && !detail && (
            <div className="empty curriculum-detail-state">เลือกรายวิชาจากตารางเพื่อดูรายละเอียด</div>
          )}
          {!detailLoading && detail && (
            <>
              <div className="curriculum-detail-heading">
                <h3 className="mono curriculum-detail-code">{detail.course.course_code || "-"}</h3>
                <p className="detail-name curriculum-detail-name">
                  {detail.course.name_en || detail.course.name_th || "-"}
                </p>
              </div>
              <dl className="detail-list">
                <div>
                  <dt>หน่วยกิต</dt>
                  <dd>
                    {detail.course.credits ||
                      detail.course.credit_units ||
                      "-"}
                  </dd>
                </div>
                <div>
                  <dt>หมวดวิชา</dt>
                  <dd>{detail.course.category || "-"}</dd>
                </div>
                <div>
                  <dt>วิชาที่ต้องเรียนก่อน</dt>
                  <dd>{detail.course.prerequisite_text || "-"}</dd>
                </div>
              </dl>
              <section className="curriculum-detail-section">
                <h4>คำอธิบายรายวิชา</h4>
                <p className="detail-desc">
                  {detail.course.description_en || "-"}
                </p>
              </section>
              <section className="curriculum-detail-section">
                <h4>ตำแหน่งในแผนการเรียน</h4>
                {detail.placements?.length > 0 ? (
                  <ul className="detail-ul curriculum-placement-list">
                    {detail.placements.map((pl, i) => (
                      <li key={i}>
                        {pl.program} · {pl.plan_key} · Y{pl.year ?? "-"} S
                        {pl.semester ?? "-"}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="curriculum-detail-missing">-</p>
                )}
              </section>
            </>
          )}
        </aside>
      </div>
    </div>
  );
}

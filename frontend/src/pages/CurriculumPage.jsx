import { useEffect, useMemo, useState } from "react";
import { fetchCourseDetail, fetchCurriculum, fetchPrograms } from "../api";

const PAGE_SIZE = 50;

export default function CurriculumPage() {
  const [programs, setPrograms] = useState([]);
  const [filters, setFilters] = useState({
    program: "IT",
    plan: "",
    year: "",
    semester: "",
    search: "",
  });
  const [applied, setApplied] = useState({ program: "IT", limit: PAGE_SIZE, offset: 0 });
  const [data, setData] = useState({ items: [], total: 0 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    fetchPrograms()
      .then((res) => setPrograms(res.programs || []))
      .catch(() => setPrograms([]));
  }, []);

  const plans = useMemo(() => {
    const found = programs.find((p) => p.program_code === filters.program);
    return found ? found.plans : [];
  }, [programs, filters.program]);

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
    setSelected(item.course_code);
    setDetail(null);
    setDetailLoading(true);
    try {
      const res = await fetchCourseDetail(item.course_code, item.program);
      setDetail(res);
    } catch {
      setDetail(null);
    } finally {
      setDetailLoading(false);
    }
  }

  const page = Math.floor((applied.offset || 0) / PAGE_SIZE) + 1;
  const pageCount = Math.max(1, Math.ceil((data.total || 0) / PAGE_SIZE));

  return (
    <div className="page">
      <header className="page-header">
        <h1>Curriculum document</h1>
        <p>ค้นหารายวิชาตามหลักสูตร ปี และเทอม จากฐานข้อมูลทางการ</p>
      </header>

      <form className="card filter-card" onSubmit={handleSearch}>
        <div className="filter-grid">
          <div className="field">
            <label>Program</label>
            <select
              value={filters.program}
              onChange={(e) =>
                setFilters({ ...filters, program: e.target.value, plan: "" })
              }
            >
              <option value="">All</option>
              {programs.map((p) => (
                <option key={p.program_code} value={p.program_code}>
                  {p.program_code}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Plan</label>
            <select
              value={filters.plan}
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
            <label>Year</label>
            <select
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
            <label>Semester</label>
            <select
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
            <label>Search</label>
            <input
              type="search"
              value={filters.search}
              onChange={(e) =>
                setFilters({ ...filters, search: e.target.value })
              }
              placeholder="รหัสวิชา หรือชื่อวิชา เช่น 06016454"
            />
          </div>
        </div>
        <div className="actions">
          <button type="submit" disabled={loading}>
            {loading ? "กำลังโหลด..." : "ค้นหา"}
          </button>
          <span className="hint">
            พบ {data.total || 0} รายวิชา · หน้า {page}/{pageCount}
          </span>
        </div>
      </form>

      {error && <div className="error visible">{error}</div>}

      <div className="curriculum-layout">
        <div className="card table-card">
          <div className="table-wrap">
            <table className="rows-table">
              <thead>
                <tr>
                  <th>Program</th>
                  <th>Code</th>
                  <th>Name</th>
                  <th>Credits</th>
                  <th>Y/S</th>
                  <th>Plan</th>
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
                    <td className="mono">{item.course_code}</td>
                    <td>{item.name_en || item.name_th || "-"}</td>
                    <td>{item.credits || item.credit_units || "-"}</td>
                    <td>
                      {item.year ?? "-"} / {item.semester ?? "-"}
                    </td>
                    <td>{item.plan_key || "-"}</td>
                  </tr>
                ))}
                {data.items.length === 0 && !loading && (
                  <tr>
                    <td colSpan={6} className="empty">
                      ไม่พบรายวิชาตามเงื่อนไข
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="actions pager">
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
          </div>
        </div>

        <div className="card detail-card">
          <h2 className="section-title">รายละเอียดวิชา</h2>
          {detailLoading && <div className="empty">กำลังโหลด...</div>}
          {!detailLoading && !detail && (
            <div className="empty">เลือกรายวิชาจากตารางเพื่อดูรายละเอียด</div>
          )}
          {!detailLoading && detail && (
            <>
              <h3 className="mono">{detail.course.course_code}</h3>
              <p className="detail-name">
                {detail.course.name_en || detail.course.name_th}
              </p>
              <dl className="detail-list">
                <div>
                  <dt>Credits</dt>
                  <dd>
                    {detail.course.credits ||
                      detail.course.credit_units ||
                      "-"}
                  </dd>
                </div>
                <div>
                  <dt>Category</dt>
                  <dd>{detail.course.category || "-"}</dd>
                </div>
                <div>
                  <dt>Prerequisite</dt>
                  <dd>{detail.course.prerequisite_text || "-"}</dd>
                </div>
              </dl>
              {detail.course.description_en && (
                <p className="detail-desc">{detail.course.description_en}</p>
              )}
              {detail.placements?.length > 0 && (
                <>
                  <h4>Placements</h4>
                  <ul className="detail-ul">
                    {detail.placements.map((pl, i) => (
                      <li key={i}>
                        {pl.program} · {pl.plan_key} · Y{pl.year ?? "-"} S
                        {pl.semester ?? "-"}
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

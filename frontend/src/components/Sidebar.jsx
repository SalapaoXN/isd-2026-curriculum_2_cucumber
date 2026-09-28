import { NavLink } from "react-router-dom";

export default function Sidebar({ health, open, onClose }) {
  return (
    <>
      <div
        className={`sidebar-backdrop${open ? " visible" : ""}`}
        onClick={onClose}
      />
      <aside className={`sidebar${open ? " open" : ""}`}>
        <div className="sidebar-brand">
          <div className="sidebar-logo">C</div>
          <div>
            <div className="sidebar-title">CUCUMBER</div>
            <div className="sidebar-subtitle">Curriculum Assistant</div>
          </div>
        </div>

        <nav className="sidebar-nav">
          <NavLink
            to="/chat"
            className={({ isActive }) =>
              `sidebar-link${isActive ? " active" : ""}`
            }
            onClick={onClose}
          >
            <span className="sidebar-icon">💬</span>
            Chat
          </NavLink>
          <NavLink
            to="/curriculum"
            className={({ isActive }) =>
              `sidebar-link${isActive ? " active" : ""}`
            }
            onClick={onClose}
          >
            <span className="sidebar-icon">📚</span>
            Curriculum document
          </NavLink>
        </nav>

        <div className="sidebar-footer">
          <span
            className={`status-dot${
              health?.database_ready ? " ok" : ""
            }`}
          />
          <span className="sidebar-health">
            {health
              ? health.database_ready
                ? "ระบบพร้อมใช้งาน"
                : "ระบบยังไม่พร้อม"
              : "กำลังตรวจสอบระบบ..."}
          </span>
        </div>
      </aside>
    </>
  );
}

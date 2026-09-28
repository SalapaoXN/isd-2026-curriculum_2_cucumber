import { useEffect, useState } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import ChatPage from "./pages/ChatPage";
import CurriculumPage from "./pages/CurriculumPage";
import { fetchHealth } from "./api";

export default function App() {
  const [health, setHealth] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => setHealth(null));
  }, [location.pathname]);

  return (
    <div className="app-shell">
      <Sidebar
        health={health}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <div className="main">
        <button
          className="menu-button"
          type="button"
          onClick={() => setSidebarOpen(true)}
          aria-label="Open navigation"
        >
          ☰
        </button>
        <Routes>
          <Route path="/" element={<Navigate to="/chat" replace />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/curriculum" element={<CurriculumPage />} />
          <Route path="*" element={<Navigate to="/chat" replace />} />
        </Routes>
      </div>
    </div>
  );
}

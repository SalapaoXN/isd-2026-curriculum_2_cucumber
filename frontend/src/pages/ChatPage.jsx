import { useState } from "react";
import { askQuestion } from "../api";

export default function ChatPage() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [context, setContext] = useState(null);
  const [followUp, setFollowUp] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleAsk() {
    const q = question.trim();
    setError("");
    if (q.length < 2) {
      setError("กรุณาพิมพ์คำถามอย่างน้อย 2 ตัวอักษร");
      return;
    }
    setLoading(true);
    try {
      const data = await askQuestion(q, followUp ? context : null);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          question: q,
          answer: data.answer || "ไม่พบคำตอบ",
          status: data.status,
          provenance: data.provenance || [],
        },
      ]);
      setContext(data.next_context || null);
      setQuestion("");
    } catch (err) {
      setError(err.message || "เกิดข้อผิดพลาด");
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setMessages([]);
    setContext(null);
    setQuestion("");
    setError("");
  }

  return (
    <div className="page">
      <header className="page-header">
        <h1>Chat</h1>
        <p>ระบบถาม–ตอบข้อมูลหลักสูตรจากฐานข้อมูล CUCUMBER</p>
      </header>

      <section className="card ask-card">
        <label htmlFor="question">คำถาม</label>
        <textarea
          id="question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.ctrlKey && e.key === "Enter") handleAsk();
          }}
          placeholder="เช่น IT วิชา 06016454 กี่หน่วยกิต"
        />
        <div className="actions">
          <button type="button" onClick={handleAsk} disabled={loading}>
            {loading ? "กำลังค้นหา..." : "ถาม"}
          </button>
          <button
            type="button"
            className="ghost"
            onClick={handleReset}
            disabled={loading && messages.length === 0}
          >
            เริ่มใหม่
          </button>
          <label className="check">
            <input
              type="checkbox"
              checked={followUp}
              onChange={(e) => setFollowUp(e.target.checked)}
            />
            ต่อบทสนทนา (multi-turn)
          </label>
          <span className="hint">กด Ctrl + Enter เพื่อส่งคำถาม</span>
        </div>
        {error && <div className="error visible">{error}</div>}
      </section>

      <section className="thread">
        {messages.length === 0 && (
          <div className="empty">ยังไม่มีบทสนทนา เริ่มถามได้เลย</div>
        )}
        {messages.map((m) => (
          <article key={m.id} className="card thread-item">
            <div className="bubble user">{m.question}</div>
            <div className="bubble assistant">{m.answer}</div>
            <div className="meta">
              <span className={`badge status-${m.status}`}>{m.status}</span>
              {m.provenance.length > 0 && (
                <span className="hint">
                  provenance:{" "}
                  {m.provenance
                    .map(
                      (p) =>
                        `${p.program || "-"} p.${p.source_page ?? "?"}`
                    )
                    .join(" · ")}
                </span>
              )}
            </div>
          </article>
        ))}
      </section>
    </div>
  );
}

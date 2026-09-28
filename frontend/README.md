# CUCUMBER Frontend (React)

React + Vite single-page app. Every page uses a shared left navigation pane with:

- **Chat** (`/chat`) — ask questions via `POST /api/ask` (multi-turn supported)
- **Curriculum document** (`/curriculum`) — browse courses via `GET /api/programs`, `GET /api/curriculum`, `GET /api/courses/{code}`

## Develop

```powershell
cd frontend
npm install
npm run dev      # http://127.0.0.1:5173 (proxies /api to :8000)
```

Run the backend separately in another terminal:

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

## Production build

```powershell
cd frontend
npm run build    # outputs frontend/dist/
```

The backend prefers `frontend/dist/index.html` when present (serves `/assets`
too) and falls back to `frontend/index.html`. SPA routes `/chat` and
`/curriculum` return the same bundle. Rebuild after changing `frontend/src`.

# schedulingclassroom

Python Firebase Functions backend for two related concerns:

1. **EVO planner / coaching LLM API** — the HTTP endpoints that the EVO mobile
   app and the Node.js Cloud Functions in `../backend/functions` call into for
   AI-generated planner content, coaching messages, and progress summaries.
2. **School timetable optimizer** — an OR-Tools / PuLP CP-SAT solver that
   produces conflict-free weekly timetables for a primary school.

The two pieces share this codebase because they were originally prototyped
together; they could be split later without disrupting the EVO app.

## Layout

```
schedulingclassroom/
├── firebase.json              # Functions deploy config (runtime: python311)
├── functions/                 # <-- the deploy entry. firebase auto-discovers main.py
│   ├── main.py                # All @https_fn.on_request endpoints
│   ├── chatgpt_wrapper.py     # OpenAI/Chat client wrapper
│   ├── config.py              # API config, model names, prompts
│   ├── planner_utils.py       # Helpers for the planner endpoints
│   ├── generate_planner_content.py
│   ├── todo_generator.py
│   ├── school_scheduler.py    # Solver used by /generate_schedule
│   └── ...                    # Backups, demos, deploy scripts
├── school_scheduler.py        # Standalone solver (notebook / local use)
├── main.py                    # Standalone solver wrapper (notebook / local use)
└── workbook.ipynb             # Solver development notebook
```

The top-level `school_scheduler.py` and `main.py` are the standalone notebook
versions of the solver and are **not** part of the deploy. Firebase deploys
only what is under `functions/`.

## Endpoints (excerpt)

All endpoints are HTTP-triggered Cloud Functions defined in `functions/main.py`.
Notable groups:

- `generate_schedule`, `get_schedule_info` — school timetable solver.
- `generate_planner_content`, `generate_planner_content_async`,
  `process_planner_job`, `summarize_planner` — AI planner generation.
- `progress`, `coach`, `encourage_in_the_morning`,
  `summarize_end_of_the_week`, `summarize_next_week` — coaching messages.
- `summary_this_year_todos`, `summary_this_month_todos`, `analyze_user_todos`,
  `todo_fate_prediction` — todo analytics consumed by the Node functions in
  `../backend/functions`.
- `create_rag_todo_users`, `add_user_memory`, `embed_user_todos`,
  `delete_user_todo_memories` — RAG memory store for personalization.
- `track_user_intent_signal`, `get_user_intent_profile` — intent profiling
  used by the EVO concept's identity / pattern-detection layer.

## Environment

Secrets are loaded via `python-dotenv` at cold start. `functions/.env` is
loaded for local emulator runs; in production set them as Firebase secrets:

```
firebase functions:secrets:set OPENAI_API_KEY
```

`functions/.env` is gitignored and **must not** be committed.

## Deploy

```
cd schedulingclassroom
firebase deploy --only functions
```

To deploy a single endpoint:

```
firebase deploy --only functions:summary_this_month_todos
```

Helper scripts in `functions/`:

- `deploy.sh` — full deploy with logging.
- `deploy_optimized.sh` — deploys only the changed functions.
- `allow_public_access.sh` — sets the IAM allUsers invoker role on the
  HTTP endpoints (used because EVO's Node functions invoke them without
  service-account auth).

## Local development

```
cd schedulingclassroom/functions
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
firebase emulators:start --only functions
```

The Plotly-based scheduler visualization and the OR-Tools solver run on
the Python 3.11 runtime in production. Keep your local interpreter on
3.11 to match (`pyenv install 3.11.x` recommended).

## Companion code

- `../EVOforluanching/` — Expo React Native client.
- `../backend/functions/` — Node Cloud Functions for notifications,
  newsfeed triggers, and scheduled summaries (which call into the
  endpoints listed above).

from __future__ import annotations

DEFAULT_ROUTES = {
    "simple": ["main", "summarizer"],
    "planning": ["main", "head_decision", "planner", "reviewer", "summarizer"],
    "analyze": ["main", "head_decision", "research", "planner", "reviewer", "summarizer"],
    "coding": ["main", "head_decision", "planner", "coding", "tool", "verifier", "reviewer", "summarizer"],
    "tool": ["main", "head_decision", "tool", "verifier", "summarizer"],
}

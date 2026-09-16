# Oviva Patient Message Triage Agent

A small AI agent that classifies incoming patient messages into operational
categories and drafts a suggested reply, built as prep for the Working
Student – AI & Operations role.

## Problem
Operational teams handling patient messages need to route each one to the
right process (scheduling, health questions, technical issues, billing, or
human escalation) before it can be actioned. Doing this manually for every
message is slow and inconsistent.

## Approach
- Python script calling the Groq API (openai/gpt-oss-20b model)
- Structured system prompt defining 5 fixed categories and a strict JSON
  output format (category, confidence, draft_reply)
- temperature=0 for consistent, repeatable classification
- Evaluation harness: 17 hand-labeled test messages (English + German),
  including deliberately ambiguous and safety-critical cases
- Results logged to CSV for inspection, not just a single accuracy number

## Results
- Initial 10-message set: 100% accuracy
- Added 6 harder cases (ambiguous category boundaries, a medical safety
  edge case): 100% accuracy, safety rule held
- Added a multi-signal stress case: accuracy dropped to 94% (16/17) —
  "I want to change my coach, this one doesn't understand me" was
  misclassified as an appointment request because of the word "change"
- Root cause: the model matched surface keywords instead of intent
- Fix: added an explicit prompt rule distinguishing coach-relationship
  complaints from scheduling requests
- Re-ran: 100% accuracy (17/17)

## What I'd add with more time
- Connect the agent to a live data source (CSV or Google Sheet) instead of
  a hardcoded test set, so it runs on a real message queue
- Add a proper agent framework with tool calling, so the agent can take
  actions (e.g. actually book a reschedule) rather than only classify
- Expand the test set further and track accuracy over time as a real
  evaluation suite, not a one-off script
- Add human-in-the-loop review for low-confidence predictions before any
  reply is sent


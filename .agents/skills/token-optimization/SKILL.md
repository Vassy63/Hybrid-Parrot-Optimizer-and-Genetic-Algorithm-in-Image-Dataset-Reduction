---
name: token-optimization
description: >-
  Minimizes token consumption across prompt and completion phases by enforcing direct communication, diff-only code changes, and compact state representations.
---

# Token Optimization Skill

Minimize token consumption across prompt and completion phases by strictly adhering to the following constraints:

## 1. Communication
- Eliminate pleasantries, filler phrases, conversational openers, and conclusions.
- Answer directly in the first line.
- Use bullet points or key-value structures instead of narrative prose.

## 2. Code & Tool Handling
- Do not output unchanged context or boilerplate code.
- Return only the diff, targeted patch, or changed functions.
- If referencing data, state specific values or IDs without reproducing the raw input payload.

## 3. Compression Strategy
- When summarizing chat history, convert conversations into a dense fact state table (Entities, State, Action).
- Drop intermediate chain-of-thought logging unless explicitly asked for debug traces.

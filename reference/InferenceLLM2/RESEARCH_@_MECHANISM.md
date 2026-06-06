# Phase 0 Research: `@` Symbolic Reference Mechanism

## Date: 2026-06-01
## Status: COMPLETE — NO-GO for project-level `@`

---

## 1. `@` Mechanism Found

**YES** — `@` references are processed by the Claude Code harness, but **ONLY for global CLAUDE.md**.

### Evidence (Global - CONFIRMED)
- Global `~/.claude/CLAUDE.md` contains: `@RTK.md` (1 line)
- Context reminder shows:
  ```
  Contents of /Users/apple/.claude/CLAUDE.md (user's private global instructions for all projects):
  @RTK.md

  Contents of /Users/apple/.claude/RTK.md (user's private global instructions for all projects):
  # RTK - Rust Token Killer
  ...
  ```
- The reference `@RTK.md` is preserved AND the content appears as a separate section

### Evidence (Project - FAILED)
- Project `/Users/apple/CLAUDE.md` contains: `@test_symref.md` at line 240
- Test file exists at `/Users/apple/test_symref.md` with content `SYMBOLIC_REFERENCE_TEST_CONTENT_12345`
- Context reminder shows the raw `@test_symref.md` text but **NO separate section** for the resolved file
- Contrast: Global `@RTK.md` shows BOTH reference AND resolved content; project `@test_symref.md` shows ONLY the reference

### Conclusion
**Project-level `@` references are NOT processed by the harness.** Only global `~/.claude/CLAUDE.md` gets `@` resolution.

---

## 2. Why Project `@` Fails

Hypotheses:
1. **Harness limitation**: The harness only scans `~/.claude/CLAUDE.md` for `@` references, not project-level `CLAUDE.md` files
2. **Security boundary**: Project CLAUDE.md could reference arbitrary files, so `@` is disabled for safety
3. **Implementation oversight**: Project CLAUDE.md parsing was added after global and `@` support wasn't extended

Most likely: **Implementation oversight or intentional security boundary**. The harness treats global and project CLAUDE.md differently.

---

## 3. Path Resolution Rules (Confirmed for Global Only)

For global CLAUDE.md (`~/.claude/CLAUDE.md`):
- `@RTK.md` → `~/.claude/RTK.md` ✓ CONFIRMED
- Resolution is **relative to the CLAUDE.md file's directory**

For project CLAUDE.md (`/Users/apple/CLAUDE.md`):
- `@test_symref.md` → NOT RESOLVED ✗ CONFIRMED FAIL

---

## 4. GO/NO-GO Verdict

### Verdict: NO-GO

**Project-level `@` symbolic references do not work.** The harness only resolves `@` in global `~/.claude/CLAUDE.md`.

**Evidence**:
- Global `@RTK.md`: Reference + resolved content both appear in context ✓
- Project `@test_symref.md`: Only reference appears, no resolved content ✗
- Test file confirmed existing at correct path (`/Users/apple/test_symref.md`)

---

## 5. Alternative Approaches

Since project `@` doesn't work, here are alternatives to reduce CLAUDE.md bloat:

### Option A: Move Everything to Global CLAUDE.md (RECOMMENDED)
- Move all 237 lines from project CLAUDE.md to global `~/.claude/CLAUDE.md`
- Use `@` references in global CLAUDE.md for sections (proven working)
- Project CLAUDE.md becomes minimal or empty

**Pros**:
- Uses proven `@` mechanism
- Sections can be shared across projects
- No custom tooling needed

**Cons**:
- Rules apply globally (may not want this for non-Python projects)
- All projects get the same persona

**Mitigation**: Add conditional logic in global CLAUDE.md:
```markdown
@if: python-project
@.claude/python_rules.md
@endif
```
(But this conditional syntax doesn't exist — would need to be simulated)

### Option B: Preprocessor Script
- Build a simple script that expands `@` references before session start
- Run manually: `python3 expand_claude_md.py`
- Script reads project CLAUDE.md, resolves `@` references, generates expanded file

**Pros**:
- Works around harness limitation
- Can add features like conditional includes

**Cons**:
- Custom tooling (reinvention)
- Must run before each session
- Risk of divergence between source and expanded file

### Option C: Compress Inline (Minimal Change)
- Keep everything inline but aggressively compress
- Remove examples, reduce verbosity
- Target: 237 lines → ~80 lines

**Pros**:
- No new files or tools
- Works immediately

**Cons**:
- Still loads everything every turn
- Less modular than `@` approach
- Less readable when compressed

### Option D: Hybrid — Global for Shared, Inline for Project-Specific
- Move shared rules (behavioral guidelines, execution policy) to global CLAUDE.md
- Keep project-specific rules inline

**Pros**:
- Reduces per-project bloat
- Shared rules only loaded once

**Cons**:
- Global CLAUDE.md becomes large
- Still no `@` for project-specific sections

---

## 6. Recommendation

**Option A (Move to Global)** is the cleanest solution:

1. Decompose project CLAUDE.md into section files in `~/.claude/`
2. Reference them from global CLAUDE.md using `@` (proven working)
3. Project CLAUDE.md becomes empty or contains only project-specific overrides

**For multi-project concerns**: If the user works on non-Python projects, the global persona might be too specific. In that case, Option D (hybrid) is better — keep a minimal generic persona globally and add Python-specific rules via project CLAUDE.md (still inline, but much shorter).

**Given the user's focus**: The user primarily works on Python projects (InferenceLLM2, discoveryLoop, etc.). A global Python engineer persona is appropriate.

---

## 7. Context Size Impact

### Current (all inline, project CLAUDE.md):
- Project CLAUDE.md: 237 lines (~4KB)
- Global CLAUDE.md: 1 line (~50B)
- RTK.md: ~30 lines (~500B)
- **Per turn**: ~4.5KB of persona context

### With Global `@` (Option A):
- Project CLAUDE.md: 0-5 lines (~100B)
- Global CLAUDE.md: ~10 lines of `@` references (~200B)
- Loaded sections: ~2KB per relevant section
- **Per turn**: ~2.3KB (assuming 2 sections loaded) — **~50% reduction**

---

## 8. Next Steps

1. **User decision**: Which alternative to pursue?
2. **If Option A**: Create section files, update global CLAUDE.md, clear project CLAUDE.md
3. **If Option D**: Move shared sections to global, keep compressed project-specific rules
4. **Clean up test files**: Remove `/Users/apple/test_symref.md` and `~/.claude/test_symref.md`
5. **Remove `@test_symref.md`** from project CLAUDE.md (it doesn't work anyway)

---

## Test Artifacts

- `/Users/apple/test_symref.md` — test file (to be cleaned up)
- `~/.claude/test_symref.md` — original test file (to be cleaned up)
- `@test_symref.md` in project CLAUDE.md — to be removed

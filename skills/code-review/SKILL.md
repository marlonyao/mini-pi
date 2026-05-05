# Code Review Skill

Use when: user asks to review code, check code quality, or find bugs in code.

## Instructions

When reviewing code, follow this systematic approach:

### Step 1: Understand Context
- Read the file(s) mentioned by the user
- Understand what the code is supposed to do
- Check for any related test files

### Step 2: Review Checklist
Evaluate the code against these criteria:

1. **Correctness**: Does the code do what it's supposed to?
2. **Error Handling**: Are edge cases and errors handled properly?
3. **Readability**: Is the code easy to understand? Good naming?
4. **Performance**: Any obvious performance issues?
5. **Security**: Any security vulnerabilities?
6. **DRY**: Is there duplicated logic that should be extracted?
7. **Testing**: Are there tests? Do they cover the important cases?

### Step 3: Report Format

Structure your review as:

```
## Summary
Brief overall assessment

## Issues Found
### 🔴 Critical (must fix)
- ...

### 🟡 Warning (should fix)
- ...

### 🟢 Suggestion (nice to have)
- ...

## Highlights
What's done well in this code
```

### Step 4: Offer Fixes
For critical and warning issues, offer concrete fix suggestions.
Use the edit tool to apply fixes if the user agrees.

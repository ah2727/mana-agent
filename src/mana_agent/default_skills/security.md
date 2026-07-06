# Security Skill

This default security guidance is used by `mana_agent`’s **Security** skill.

## When to use

Use this skill when your agent is handling:

- **Authentication** and session management
- **Authorization** / permissions
- **Secrets** (API keys, tokens, credentials)
- **Sensitive user data**
- **Secure-by-default changes**

## Required behavior

1. **Follow the repository’s security policy**
   - If `SECURITY.md` is missing, create it.
   - See the repository root `CONTRIBUTING.md` for contribution workflow.

2. **Minimize secret exposure**
   - Never log secrets.
   - Never hardcode secrets in code or docs.
   - Prefer environment variables / secret managers.

3. **Principle of least privilege**
   - Use the minimum permissions required.
   - Avoid overly broad roles and admin actions.

4. **Validate and sanitize inputs**
   - Treat all external input as untrusted.
   - Validate types and ranges.
   - Sanitize anything rendered to a user or used in commands.

5. **Prefer safe defaults**
   - Secure transport (TLS) where applicable.
   - Secure cookies / session settings.
   - Avoid insecure cryptographic choices.

## Checklist before submitting changes

- [ ] Did you avoid introducing new sensitive data exposure?
- [ ] Are auth/permission changes documented and reviewed?
- [ ] Is any new functionality resilient to malformed/untrusted input?
- [ ] Did you update documentation or references if security-relevant behavior changed?

## Notes

- Keep user-impacting security changes clear and documented.
- If you discover a potential vulnerability, report it following the repository’s process.

# Skill: security

Security audit skill.

1. Inventory entry points and data flows (search_files, read_file).
2. Check for OWASP Top 10 style issues:
   - Injection (SQL, command, template)
   - Broken auth / session
   - Hardcoded secrets
   - Path traversal / SSRF
   - Insecure deserialization
3. Score risk and list findings with severity.
4. Produce hardened patches via write_file where appropriate.
5. Re-check after patches.

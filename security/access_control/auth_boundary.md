# Auth Boundary

The API is auth-ready even in local development:

- Public endpoints: `/health`, `/ready`, `/metrics`.
- User endpoints: `/v1/align`, `/v1/evaluate`.
- Operator endpoints: `/v1/admin/*`.

Production deployments should validate JWTs or provider sessions and map claims to the roles in `backend/auth/policy.yaml`.

## Secure Defaults

- Deny unknown roles.
- Keep secrets in provider secret stores.
- Do not log raw prompts or responses without redaction.
- Require benchmark and security gates before model promotion.

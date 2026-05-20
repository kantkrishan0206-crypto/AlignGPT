# Vercel Frontend Deployment

Deploy `frontend/web/nextjs_app` as the public AlignGPT web app.

## Required Settings

- Framework: Next.js
- Root directory: repository root
- Build command: `cd frontend/web/nextjs_app && npm install && npm run build`
- Environment variable: `NEXT_PUBLIC_ALIGNGPT_API_BASE_URL`

The included root `vercel.json` prepares rewrites for an external API host once a backend deployment URL exists.

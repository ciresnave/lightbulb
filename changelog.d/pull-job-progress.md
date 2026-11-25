kind: feat: background model pulls with job progress
---
`POST /v1/lightbulb/admin/models/pull` accepts `"wait": false`: it answers 202 with a job id at once and downloads in the background, and `GET /v1/lightbulb/admin/models/pull/{id}` reports `state`, per-file `downloaded`/`total` bytes and the final result or failing status. Without `wait` the route behaves as before. Tested against the localhost stub Hub (progress, failure status, unknown job 404, 202 and validation order); the real Hub is not run. ROADMAP F1 and F2 rows updated.

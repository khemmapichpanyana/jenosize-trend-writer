"""Studio: the content agent, its artifacts, uploaded assets and published pages.

Served by the jobs API on Modal (see pipeline/api/app.py). Kept apart from `app`
(the Vercel article API, which must stay lean) and `pipeline` (data + training).
"""

# Svend SEO Playbook

Breadcrumb trail for SEO infrastructure, monitoring, and keyword strategy.
Last updated: 2026-02-21.

---

## 1. Infrastructure Status

### What's in place
- **Sitemap**: Django sitemaps framework in `web/svend/urls.py` (lines 7–76, 143)
  - `StaticSitemap` — home, blog index, tools hub, 9 tool pages, privacy, terms, whitepapers, comparison pages
  - `BlogSitemap` — all published `BlogPost` objects (dynamic lastmod)
  - `WhitePaperSitemap` — all published `WhitePaper` objects (dynamic lastmod)
  - Live at: `https://svend.ai/sitemap.xml`
  - When you add a new public page, add it to the `StaticSitemap.items()` list
- **robots.txt**: `web/templates/robots.txt` → served at `https://svend.ai/robots.txt`
  - Allows: `/`, `/blog/`, `/privacy/`, `/terms/`
  - Disallows: `/app/`, `/api/`, `/admin/`, `/login/`, `/register/`, `/internal/`
  - Points to sitemap
- **Meta tags / OG / Twitter cards**: Every public template has full meta tags
  - Landing: `templates/landing.html` (lines 1–46)
  - Blog posts: `templates/blog_detail.html` (dynamic from model fields)
  - Whitepapers: `templates/whitepaper_detail.html`
  - Tools: `templates/tool_base.html` + `templates/tools/*.html`
- **Structured data (JSON-LD)**: Organization, SoftwareApplication, FAQPage, BreadcrumbList, WebApplication, Article, TechArticle — varies by page type
- **Canonical URLs**: Set on all public pages
- **Security headers**: HSTS (2yr, preload), CSP, X-Frame-Options, Referrer-Policy — via Caddy

### What's NOT in place
- **Google Analytics / Plausible** — no external analytics yet (only internal BlogView/WhitePaperDownload tracking)
- **Backlink profile** — very thin, not listed in any roundup/review sites
- **Google Ads Keyword Planner** account — needed for actual search volume data

---

## 2. Google Search Console

### Setup (completed 2026-02-21)
1. Go to https://search.google.com/search-console
2. Add property → URL prefix → `https://svend.ai`
3. Verify ownership via DNS TXT record in Cloudflare
4. Submit sitemap: Sitemaps → enter `sitemap.xml` → Submit

### Ongoing monitoring
- **Index Coverage**: Shows which pages are indexed, which are excluded and why
- **Performance**: Real search queries, impressions, clicks, average position (the actual ranking data)
- **URL Inspection**: Paste any URL to check index status, request indexing manually
- **robots.txt Tester**: Validate syntax — this is where the Content-Signal error showed up

### Requesting indexing for priority pages
Use URL Inspection tool, paste URL, click "Request Indexing". Daily limit ~10-12. Priority order:
1. `https://svend.ai/` (homepage)
2. `https://svend.ai/tools/cpk-calculator/`
3. `https://svend.ai/tools/oee-calculator/`
4. `https://svend.ai/tools/control-chart-generator/`
5. `https://svend.ai/tools/gage-rr-calculator/`
6. `https://svend.ai/tools/sample-size-calculator/`
7. `https://svend.ai/tools/pareto-chart-generator/`
8. `https://svend.ai/tools/takt-time-calculator/`
9. `https://svend.ai/tools/sigma-calculator/`
10. `https://svend.ai/tools/kanban-card-generator/`
11. `https://svend.ai/tools/` (hub)
12. `https://svend.ai/blog/`
13. `https://svend.ai/svend-vs-minitab/`
14. Individual blog posts and whitepapers

Expect 1–2 weeks for initial indexing, longer for ranking improvements.

---

## 3. Cloudflare Gotchas

### Managed robots.txt injection (RESOLVED 2026-02-21)
- **Problem**: Cloudflare's "AI Audit" feature was prepending ~25 lines of `Content-Signal` directives and duplicate `User-agent: *` blocks to robots.txt
- **Symptom**: Google Search Console reported "Syntax not understood (line 29)" — the `Content-Signal: search=yes,ai-train=no` directive
- **Structural issue**: Two `User-agent: *` blocks meant Google might ignore our Disallow rules and Sitemap directive entirely (only first matching group is used per spec)
- **Fix**: Disabled "Managed robots.txt" / AI content signals in Cloudflare dashboard (AI → AI Audit section)
- **Verification**: `curl -sk https://svend.ai/robots.txt` should return only our template content (14 lines)
- **Note**: AI bot blocking (ClaudeBot, GPTBot, Bytespider, etc.) was already handled by per-bot `Disallow: /` stanzas, so disabling the managed injection lost nothing

### Cloudflare blocking crawlers
- Before disabling managed robots.txt, `curl` to svend.ai returned 403 (Cloudflare was blocking)
- After disabling, landing page and sitemap both return 200
- If 403s return, check: Cloudflare → Security → WAF rules, and Bot Fight Mode settings
- The "Under Attack" mode will also block legitimate crawlers — only enable temporarily during actual attacks

### Where to find these settings
- Cloudflare Dashboard → select `svend.ai` domain
- **AI Audit / Content Signals**: AI section (or search "AI" in sidebar)
- **Bot Fight Mode**: Security → Bots
- **WAF Rules**: Security → WAF
- **Page Rules / Caching**: Rules → Page Rules

---

## 4. Keyword Strategy

### How to get search volume data
Google Keyword Planner is the free option:
1. Go to https://ads.google.com → Tools → Keyword Planner
2. Choose "Get search volume and forecasts"
3. Paste keywords (one per line)
4. Get monthly volume ranges, competition level, CPC

Paid alternatives with more precise data: Ahrefs ($99/mo), Semrush ($130/mo), Mangools/KWFinder ($30/mo).

### Target keywords (paste into Keyword Planner)
```
cpk calculator
ppk calculator
process capability calculator
oee calculator
sample size calculator
control chart generator
gage r&r calculator
takt time calculator
pareto chart generator
sigma calculator
six sigma calculator
kanban card generator
doe software
design of experiments software
spc software
continuous improvement software
fmea software
root cause analysis software
value stream mapping software
hoshin kanri software
a3 report template
cpk vs ppk
svend vs minitab
minitab alternative
jmp alternative
```

### Ranking snapshot (2026-02-21, pre-indexing)

| Keyword | svend.ai Position | Notes |
|---|---|---|
| svend.ai (brand) | ~10 | Shows with wrong description ("reasoning system") |
| cpk calculator | Not in top 10 | Dominated by Cuemath, BYJU's, six-sigma.us |
| oee calculator | Not in top 10 | oee.com, Mingo, Factory AI |
| sample size calculator | Not in top 10 | calculator.net, ClinCalc, SurveyMonkey |
| control chart generator | Not in top 10 | ASQ, Visual Paradigm |
| gage r&r calculator | Not in top 10 | Few strong competitors — opportunity |
| doe software | Not in top 10 | Stat-Ease, Minitab, Sartorius |
| spc software | Not in top 10 | Minitab, Siemens, Statgraphics |
| continuous improvement software | Not in top 10 | Impruver, ClickUp, SweetProcess |

### Opportunity assessment

**Best bets (lower competition, specific intent):**
- `cpk calculator` / `ppk calculator` / `process capability calculator`
- `oee calculator`
- `control chart generator`
- `gage r&r calculator`
- `pareto chart generator`
- `takt time calculator`
- `kanban card generator`

**Hard targets (dominated by enterprise vendors or listicle sites):**
- `doe software`, `spc software`, `continuous improvement software`
- `fmea software`, `root cause analysis software`

**Comparison pages (high buyer intent, lower volume):**
- `minitab alternative`, `jmp alternative`, `svend vs minitab`

### Strategy
1. **Free tools** are the primary organic acquisition channel — each targets a specific long-tail keyword
2. **Blog posts** support tool pages with internal links (e.g., "Cpk vs Ppk" blog → links to Cpk calculator)
3. **Comparison pages** capture high-intent "alternative to X" traffic
4. **Whitepapers** (gated) capture emails for nurture sequences
5. Get listed in "best X software" roundup posts for backlinks (reach out to authors)

---

## 5. On-Page SEO Checklist (for new pages)

When adding a new public page:

- [ ] Add to `StaticSitemap.items()` in `urls.py` (or ensure model-based sitemap covers it)
- [ ] Set `<title>` — include primary keyword, keep under 60 chars
- [ ] Set `<meta name="description">` — compelling, include keyword, 150-160 chars
- [ ] Set `<meta name="keywords">` — 5-10 relevant terms
- [ ] Set `<link rel="canonical">` with full `https://svend.ai/...` URL
- [ ] Add Open Graph tags: `og:title`, `og:description`, `og:image`, `og:url`, `og:type`
- [ ] Add Twitter card tags: `twitter:card`, `twitter:title`, `twitter:description`, `twitter:image`
- [ ] Add JSON-LD structured data (FAQPage for tools, Article for blog, BreadcrumbList for navigation)
- [ ] Add `robots` meta tag: `index, follow`
- [ ] Internal links: link to/from related pages (tools ↔ blog posts ↔ comparison pages)
- [ ] Verify with: `curl -sk https://svend.ai/new-page/ | grep -i '<title\|og:\|canonical'`

---

## 6. Monitoring Cadence

| Task | Frequency | Where |
|---|---|---|
| Check index coverage | Weekly (first month), then monthly | Search Console → Index Coverage |
| Review search queries & positions | Weekly | Search Console → Performance |
| Check for crawl errors | Weekly (first month) | Search Console → Pages |
| Validate robots.txt | After any Cloudflare changes | Search Console → robots.txt tester |
| Review blog/whitepaper view analytics | Weekly | `/api/internal/blog/analytics/?days=7` |
| Check keyword rankings for targets | Monthly | Search Console Performance or paid tool |
| Audit structured data | After template changes | https://search.google.com/test/rich-results |
| Re-check Cloudflare isn't injecting into robots.txt | After Cloudflare updates | `curl -sk https://svend.ai/robots.txt` |

---

## 7. Internal Analytics Endpoints

Built-in tracking (no external dependency):
- Blog views: `GET /api/internal/blog/analytics/?days=30` (staff auth required)
- Whitepaper downloads: `GET /api/internal/whitepapers/analytics/?days=30`
- Tracks: daily view counts, unique visitors (IP hash), top posts, referrer domains, bot filtering
- Models: `BlogView`, `WhitePaperDownload` in `api/models.py`

---

## 8. File Reference

| What | Path |
|---|---|
| Sitemap classes | `web/svend/urls.py` (lines 7–76) |
| Sitemap route | `web/svend/urls.py` (line 143) |
| robots.txt template | `web/templates/robots.txt` |
| Landing page (meta/SEO) | `web/templates/landing.html` |
| Blog detail (meta/SEO) | `web/templates/blog_detail.html` |
| Whitepaper detail (meta/SEO) | `web/templates/whitepaper_detail.html` |
| Tool base template (meta/SEO) | `web/templates/tool_base.html` |
| Individual tools | `web/templates/tools/*.html` |
| Comparison page | `web/templates/svend_vs_minitab.html` |
| Blog/whitepaper models | `web/api/models.py` |
| Internal analytics views | `web/api/internal_views.py` |
| Caddy config (headers) | `web/Caddyfile` |
| Django settings | `web/svend/settings.py` |
| OG image | `web/static/og-image.png` |
| Competitive analysis | `reference_docs/COMPETITIVE_POSITIONING.md` |

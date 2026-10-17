# External data

Maintained by Claude Code teammate 1 (Omar Al-Aghbari). Every file here is an outside source joined to `data/stories_transactions.csv`. Branch locations are **not confirmed by the company**, so no file below establishes which branch was exposed to anything.

| File | What it is | Source | Location requested → grid point returned | Period | Retrieved | Added by | Used in |
|---|---|---|---|---|---|---|---|
| `beirut_weather_2026H1.json` | Daily max/min temperature (°C), total precipitation (mm), max 10 m wind (km/h), timezone Asia/Beirut | [Open-Meteo historical weather archive](https://open-meteo.com/en/docs/historical-weather-api) | 33.89, 35.50 → 33.85, 35.53 (60 m) | 2026-01-01 to 2026-06-30 | 2026-10-17 (commit date) | Marwa | `src/pitch_visuals.py`, notebook 05 (check), the Codex model's weather proxy |
| `batroun_weather_2026H1.json` | Same variables as above | Open-Meteo historical weather archive | 34.2553, 35.6581 → 34.27, 35.68 (16 m) | 2026-01-01 to 2026-06-30 | 2026-10-17 | Omar | Notebook 05 (main analysis, North-region assumption) |
| `broummana_weather_2026H1.json` | Same variables as above | Open-Meteo historical weather archive | 33.8828, 35.6206 → 33.85, 35.64 (629 m) | 2026-01-01 to 2026-06-30 | 2026-10-17 | Omar | Notebook 05 (Jabal check) |
| `lebanon_cpi_2026H1.json` | Lebanon consumer price index, monthly and year-on-year changes, selected components, regional changes; exchange-rate and input-cost context | Central Administration of Statistics, via Credit Libanais, BLOMINVEST and Trading Economics (links inside the file) | National, with some regional splits | 2026-01 to 2026-06 | 2026-10-17 | Omar | Notebook 08 |
| `external_events.json` | Teammate 1 event list (war milestones, ceasefire and extensions, displacement, AUB commencement, holidays, inflation shock, and the README's unverified flagged days) in the AGENTS.md schema | Cited per event (UN News, Al Jazeera, NPR, Wikipedia, AUB, LBCI, Credit Libanais, Office Holidays and others) | National / per event | 2026-01-01 to 2026-06-30 | 2026-10-17 | Omar | To be merged with the Codex registry; `branch_exposure` is `unknown` everywhere |
| `outlier_event_registry.json` | Event dates, known-at times, sources, mechanisms and counterchecks | Codex teammate's research (see `reports/source_reconciliation.md`) | Per event | 2026-01-01 to 2026-06-30 | – | Charbel | Outlier dossiers and the Day explorer |
| `context_calendar.csv`, `context_v2/`, `macro_context_audit.json` | Cached calendar, weather/forecast/air-quality/internet context features and a macro-data audit | See [EXTERNAL_CONTEXT.md](../../EXTERNAL_CONTEXT.md) | Per source | 2026-01-01 to 2026-06-30 | – | Charbel | External-context models (`src/context_*.py`) |

## Download link pattern (weather)

```
https://archive-api.open-meteo.com/v1/archive?latitude=<lat>&longitude=<lon>&start_date=2026-01-01&end_date=2026-06-30&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max&timezone=Asia%2FBeirut
```

## Rules for using these files

- **Weather files are observed (archive) weather, not forecasts.** They were only known after each day, so they may be used for retrospective analysis but **not as forecast inputs**. An operational forecast would need weather forecasts as issued the day before.
- **CPI is monthly and national.** Four of the six monthly index levels are derived from published monthly changes; the file marks `published` vs `derived` for each month. The flour, bread and fuel figures are news and NGO reports, not official statistics.
- **Exposure stays unknown** until the company confirms branch coordinates and opening hours. The North (Batroun) and Jabal (Broummana) choices are team assumptions, not facts.
- When adding a file, add a row above with its source, location, period, retrieval date and where it's used.

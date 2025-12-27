# Heat Chronomap

Automated, public, hourly climate-risk chronomap built from Open-Meteo ERA5-based data.

## How it works

- `inputs/latest.json` defines latitude, longitude, year, planting/harvest, and base temperature.
- `scripts/chronomap.py`:
  - fetches hourly temperature from Open-Meteo (no auth, global)
  - converts to local time and computes sunrise/sunset
  - computes GDD-based phenology stage windows
  - classifies hourly thermal risk by stage and photoperiod
  - generates a chronomap and saves `docs/output/chronomap.png`
- `.github/workflows/build.yml` runs the pipeline on pushes that affect:
  - `inputs/latest.json`
  - `scripts/**`
  - the workflow itself
- GitHub Actions commits the updated PNG back to the repo.
- GitHub Pages serves `docs/index.html`, which displays the latest chronomap.

## Local run

```bash
pip install -r requirements.txt
python scripts/chronomap.py

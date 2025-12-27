# Chronomap Generator (ERA5-powered)

This repository hosts a fully automated, ERA5-driven chronomap generator.

## How it works

- The web UI is hosted on **GitHub Pages** (`/docs/index.html`)
- Users enter:
  - Latitude
  - Longitude
  - Year
  - Planting date
  - Harvest date
- A **GitHub Action** runs the Python chronomap engine
- ERA5 data is downloaded via `cdsapi`
- The chronomap is generated and saved to `/docs/output/chronomap.png`
- GitHub Pages automatically serves the updated image

## Run locally

```bash
python scripts/chronomap.py \
  --lat 50.067 \
  --lon -112.097 \
  --year 2024 \
  --planting 2024-05-20 \
  --harvest 2024-09-30 \
  --out chronomap.png

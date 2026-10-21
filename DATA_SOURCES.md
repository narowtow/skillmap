# Data sources and attribution

The code is MIT-licensed (see `LICENSE`). The data below keep their own licenses.

- **Weather:** [Weather data by Open-Meteo.com](https://open-meteo.com/), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Underlying dataset: ERA5 reanalysis, Copernicus Climate Change Service / ECMWF (Hersbach et al., 2020).
  Changes: site selection, daily feature processing, cleaning and model-derived outputs.
- **Climate classes:** Beck, H. E., et al. High-resolution (1 km) Köppen-Geiger maps for 1901–2099 based on
  constrained CMIP6 projections. *Scientific Data* 10, 724 (2023). 1991–2020 map, CC BY 4.0.
- **Places:** [GeoNames](https://www.geonames.org/) (`cities15000`, `cities500`), CC BY 4.0.
- **Research stations:** coordinates compiled by hand from public station information (`config/research_stations.csv`).
- **Land outline (static figures):** [Natural Earth](https://www.naturalearthdata.com/) 1:110m land, public domain (`data/external/ne_110m_land.geojson`).

"""Local Köppen lookup and deterministic, geographically spread site selection."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from skillmap import config

SITE_COLUMNS = [
    "site_id", "name", "country_code", "lat", "lon", "elevation_m", "population",
    "site_type", "koppen_code", "koppen", "koppen_group", "koppen_desc", "koppen_fallback", "geonameid",
]
EARTH_RADIUS_KM = 6371.0088


def haversine(lat: np.ndarray | float, lon: np.ndarray | float, lat2: float, lon2: float) -> np.ndarray:
    """Great-circle distances in km, with broadcasting and dateline wrapping."""
    a, b = np.radians(lat), np.radians(lat2)
    delta = np.sin((a - b) / 2) ** 2 + np.cos(a) * np.cos(b) * np.sin(np.radians(lon - lon2) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(delta, 0, 1)))


def read_legend(path: Path = config.KOPPEN_LEGEND) -> pd.DataFrame:
    """Parse Beck's numeric legend, excluding RGB and citation text."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*(\d+):\s+(\w+)\s+(.+?)\s*\[", line)
        if match:
            code, koppen, desc = match.groups()
            rows.append((int(code), koppen, koppen[0], desc.strip()))
    result = pd.DataFrame(rows, columns=["koppen_code", "koppen", "koppen_group", "koppen_desc"])
    if sorted(result.koppen_code) != list(range(1, 31)):
        raise ValueError("Köppen legend must contain each code 1–30 exactly once")
    return result


def _ascii(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")


def read_cities(path: Path) -> pd.DataFrame:
    """Read only populated places, preserving literal country codes such as NA."""
    names = [
        "geonameid", "name", "asciiname", "alternatenames", "lat", "lon", "feature_class",
        "feature_code", "country_code", "cc2", "admin1", "admin2", "admin3", "admin4",
        "population", "elevation", "elevation_m", "timezone", "modified",
    ]
    cities = pd.read_csv(path, sep="\t", names=names, keep_default_na=False, usecols=[
        "geonameid", "name", "asciiname", "lat", "lon", "feature_class", "country_code",
        "population", "elevation_m",
    ])
    cities = cities.loc[cities.feature_class == "P"].copy()
    cities["name"] = cities.asciiname.where(cities.asciiname != "", cities.name).map(_ascii)
    cities["geonameid"] = cities.geonameid.astype("Int64")
    cities["site_id"] = "gn-" + cities.geonameid.astype(str)
    cities["population"] = pd.to_numeric(cities.population).astype("int64")
    cities["elevation_m"] = pd.to_numeric(cities.elevation_m, errors="coerce")
    cities["site_type"] = "city"
    return cities.drop(columns=["asciiname", "feature_class"])


def read_stations(path: Path = config.STATIONS_CSV) -> pd.DataFrame:
    """Load curated research stations as population-zero candidates."""
    stations = pd.read_csv(path, keep_default_na=False)
    stations["site_id"] = "st-" + stations.station_id
    stations["name"] = stations.name.map(_ascii)
    stations["geonameid"] = pd.array([None] * len(stations), dtype="Int64")
    stations["population"] = 0
    stations["site_type"] = "station"
    return stations.drop(columns="station_id")


def _coastal_code(raster, lat: float, lon: float, radius_km: float) -> int:
    """Find the nearest classified cell center in a geodesic 5 km disk."""
    from rasterio.windows import Window, from_bounds

    angle = radius_km / EARTH_RADIUS_KM
    lat_delta = np.degrees(angle)
    lon_delta = (180.0 if abs(lat) + lat_delta >= 90 else
                 np.degrees(np.arcsin(min(1, np.sin(angle) / np.cos(np.radians(lat))))))
    best_distance, best_code = float("inf"), 0
    # Shifts include cells across the date line without reading the whole globe.
    for shifted_lon in (lon, lon - 360, lon + 360):
        left, right = max(-180, shifted_lon - lon_delta), min(180, shifted_lon + lon_delta)
        if right <= left:
            continue
        window = from_bounds(left, max(-90, lat - lat_delta), right,
                             min(90, lat + lat_delta), raster.transform)
        col0, row0 = max(0, int(np.floor(window.col_off))), max(0, int(np.floor(window.row_off)))
        col1 = min(raster.width, int(np.ceil(window.col_off + window.width)))
        row1 = min(raster.height, int(np.ceil(window.row_off + window.height)))
        values = raster.read(1, window=Window(col0, row0, col1 - col0, row1 - row0))
        rr, cc = np.where((values >= 1) & (values <= 30))
        if not len(rr):
            continue
        xs, ys = raster.xy(rr + row0, cc + col0)
        distances = haversine(np.asarray(ys), np.asarray(xs), lat, lon)
        nearest = int(np.argmin(distances))
        distance = float(distances[nearest])
        if distance <= radius_km and distance < best_distance:
            best_distance, best_code = distance, int(values[rr[nearest], cc[nearest]])
    return best_code


def sample_koppen(candidates: pd.DataFrame, tif: Path = config.KOPPEN_TIF) -> pd.DataFrame:
    """Sample containing raster cells; use nearest classified center only for NoData."""
    import rasterio
    from rasterio.transform import rowcol
    from rasterio.windows import Window

    result = candidates.copy().reset_index(drop=True)
    codes = np.zeros(len(result), dtype=np.uint8)
    with rasterio.open(tif) as raster:
        rows, cols = rowcol(raster.transform, result.lon.to_numpy(), result.lat.to_numpy())
        rows, cols = np.asarray(rows), np.asarray(cols)
        valid = (rows >= 0) & (rows < raster.height) & (cols >= 0) & (cols < raster.width)
        # Read each occupied raster block once, rather than once per candidate.
        bh, bw = raster.block_shapes[0]
        blocks = pd.DataFrame({"row": rows // bh, "col": cols // bw})
        for (br, bc), indices in blocks.loc[valid].groupby(["row", "col"]).groups.items():
            indices = np.asarray(indices)
            r0, c0 = int(br * bh), int(bc * bw)
            values = raster.read(1, window=Window(c0, r0, min(bw, raster.width - c0),
                                                min(bh, raster.height - r0)))
            codes[indices] = values[rows[indices] - r0, cols[indices] - c0]
        fallback = np.zeros(len(result), dtype=bool)
        for index in np.flatnonzero(valid & (codes == 0)):
            codes[index] = _coastal_code(raster, result.at[index, "lat"], result.at[index, "lon"], 5.0)
            fallback[index] = codes[index] != 0
    result["koppen_code"] = codes
    result["koppen_fallback"] = fallback
    return result.loc[result.koppen_code.between(1, 30)].copy()


def load_candidates(
    geonames_dir: Path = config.GEONAMES_DIR,
    tif: Path = config.KOPPEN_TIF,
    legend_path: Path = config.KOPPEN_LEGEND,
    stations_path: Path = config.STATIONS_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Supplement cities15000 only for its underrepresented raster classes."""
    legend = read_legend(legend_path)
    cities = sample_koppen(read_cities(geonames_dir / "cities15000.txt"), tif)
    counts = cities.koppen_code.value_counts()
    rare = [code for code in legend.koppen_code if counts.get(code, 0) < config.MIN_PER_CLASS]
    frames = [cities, sample_koppen(read_stations(stations_path), tif)]
    if rare:
        small = read_cities(geonames_dir / "cities500.txt")
        small = sample_koppen(small.loc[~small.site_id.isin(cities.site_id)], tif)
        frames.append(small.loc[small.koppen_code.isin(rare)])
    candidates = pd.concat(frames, ignore_index=True).drop_duplicates("site_id")
    return candidates.merge(legend, on="koppen_code", validate="many_to_one")[SITE_COLUMNS], legend


def select_sites(
    candidates: pd.DataFrame,
    n_sites: int = config.N_SITES,
    quotas: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Cover rare classes first, then fill quotas by greedy farthest-point sampling.

    Grid uniqueness is never relaxed. Distance thresholds are relaxed only after
    exhausting a pool at the previous threshold. Country caps may relax in the
    class coverage pass; fill and redistribution preserve them.
    """
    quotas = config.GROUP_QUOTAS if quotas is None else quotas
    if candidates.site_id.duplicated().any():
        raise ValueError("Candidate site_id must be unique")
    pool = candidates.sort_values("site_id").reset_index(drop=True).copy()
    # Seed provides stable tie-breaking independent of input row order.
    pool["_tie"] = np.random.default_rng(config.SEED).random(len(pool))
    lat, lon = pool.lat.to_numpy(), pool.lon.to_numpy()
    lon_cells = np.rint(lon / config.ERA5_GRID_DEG).astype(int) % round(360 / config.ERA5_GRID_DEG)
    cells = list(zip(np.rint(lat / config.ERA5_GRID_DEG).astype(int), lon_cells, strict=True))
    selected: list[int] = []
    occupied: set[tuple[int, int]] = set()
    countries: dict[str, int] = {}
    distances = np.full(len(pool), np.inf)
    available = np.ones(len(pool), dtype=bool)
    relaxations = {"distance": 0, "country": 0, "redistributed": 0}
    thresholds = sorted({config.MIN_SEPARATION_KM, *[d for d in (100.0, 25.0, 0.0)
                                                 if d < config.MIN_SEPARATION_KM]}, reverse=True)

    def choose(mask: np.ndarray, coverage: bool = False) -> bool:
        if len(selected) >= n_sites:
            return False
        for relax_country in ([False, True] if coverage else [False]):
            for threshold in thresholds:
                eligible = available & mask & (distances >= threshold)
                eligible &= np.array([cell not in occupied for cell in cells])
                if not relax_country:
                    eligible &= pool.country_code.map(lambda c: countries.get(c, 0) < config.MAX_PER_COUNTRY)
                options = pool.loc[eligible].copy()
                if options.empty:
                    continue
                if coverage:
                    options = options.sort_values(["population", "_tie"], ascending=[False, True])
                else:
                    options["_recognizable"] = (options.site_type == "city") & (options.population >= 100_000)
                    options["_distance"] = distances[options.index]
                    options = options.sort_values(["_recognizable", "_distance", "population", "_tie"],
                                                  ascending=[False, False, False, True])
                index = int(options.index[0])
                country = pool.at[index, "country_code"]
                relaxations["distance"] += int(distances[index] < config.MIN_SEPARATION_KM)
                relaxations["country"] += int(countries.get(country, 0) >= config.MAX_PER_COUNTRY)
                selected.append(index)
                available[index] = False
                occupied.add(cells[index])
                countries[country] = countries.get(country, 0) + 1
                np.minimum(distances, haversine(lat, lon, lat[index], lon[index]), out=distances)
                return True
        return False

    counts = pool.koppen_code.value_counts()
    for code in sorted(counts.index, key=lambda c: (counts[c], c)):
        for _ in range(config.MIN_PER_CLASS):
            if not choose((pool.koppen_code == code).to_numpy(), coverage=True):
                break
    for group, quota in quotas.items():
        mask = (pool.koppen_group == group).to_numpy()
        while sum(mask[i] for i in selected) < quota and choose(mask):
            pass
    while len(selected) < n_sites and choose(np.ones(len(pool), dtype=bool)):
        relaxations["redistributed"] += 1
    result = pool.loc[selected, SITE_COLUMNS].reset_index(drop=True)
    result["geonameid"] = result.geonameid.astype("Int64")
    result.attrs["relaxations"] = relaxations
    return result


def coverage_table(candidates: pd.DataFrame, selected: pd.DataFrame, legend: pd.DataFrame) -> pd.DataFrame:
    """Audit all 30 classes, including classes absent from the candidates."""
    result = legend[["koppen_code", "koppen", "koppen_group"]].copy()
    counts = candidates.koppen_code.value_counts()
    result["n_candidates"] = result.koppen_code.map(counts).fillna(0).astype(int)
    result["n_selected"] = result.koppen_code.map(selected.koppen_code.value_counts()).fillna(0).astype(int)
    countries = selected.groupby("koppen_code").country_code.agg(lambda x: ";".join(sorted(set(x))))
    result["countries"] = result.koppen_code.map(countries).fillna("")
    return result

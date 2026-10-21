from skillmap import config
from skillmap.synthetic import make_daily, make_sites


def test_synthetic_follows_contract():
    sites = make_sites()
    daily = make_daily(sites, start=config.START, end=config.START + __import__("pandas").Timedelta(days=99))
    assert list(daily.columns) == ["site_id", "date", *config.WEATHER_COLUMNS]
    assert daily.groupby("site_id").size().eq(100).all()
    assert set(daily["site_id"]) == set(sites["site_id"])
    assert sites["site_id"].is_unique

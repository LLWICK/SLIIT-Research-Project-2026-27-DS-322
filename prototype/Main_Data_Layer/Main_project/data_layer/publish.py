"""Publish the local curated tables to the Neon production branch."""

import io
import secrets
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import duckdb
import psycopg
from psycopg import sql

from data_layer.market_week import VIEW_NAME, VIEW_SQL
from data_layer.paths import DUCKDB_PATH, PROJECT_ROOT

ENV_PATH = PROJECT_ROOT / ".env"
READER_ROLE = "cobweb_reader"

TABLES = {
    "dim_market": """
        market_code text primary key,
        market_name text not null,
        district text not null,
        supply_rule text not null
    """,
    "dim_crop": """
        crop_code text primary key,
        crop_name text not null
    """,
    "harti_price_weekly": """
        source text not null,
        price_type text not null,
        market_code text not null,
        crop_code text not null,
        year integer not null,
        week integer not null,
        price double precision,
        is_missing boolean not null,
        sheet_name text not null,
        source_market_name text not null,
        source_crop_name text not null,
        frequency text not null
    """,
    "dim_weather_point": """
        point_code text primary key,
        point_name text not null,
        district text not null,
        requested_lat double precision not null,
        requested_lon double precision not null,
        grid_lat double precision not null,
        grid_lon double precision not null,
        grid_elevation_m double precision not null,
        source text not null,
        precip_divisor integer not null,
        time_standard text,
        start_date date not null,
        end_date date not null
    """,
    "weather_daily": """
        obs_date date not null,
        point_code text not null,
        t2m_c double precision,
        t2m_max_c double precision,
        t2m_min_c double precision,
        precip_mm double precision,
        rh2m_pct double precision,
        is_missing boolean not null,
        primary key (point_code, obs_date)
    """,
}

WEATHER_TABLES = ("dim_weather_point", "weather_daily")
SEASON_TABLES = ("season_calendar",)
CBSL_TABLES = ("cbsl_price_daily",)
EXTENT_TABLES = ("cultivation_extent",)

TABLES["cbsl_price_daily"] = """
        source text not null,
        report_date date not null,
        price_type text not null,
        market_code text not null,
        crop_code text not null,
        price double precision,
        baseline_price double precision,
        is_missing boolean not null,
        unit text not null,
        source_market_name text not null,
        source_crop_name text not null,
        primary key (report_date, price_type, market_code, crop_code)
    """

TABLES["season_calendar"] = """
        calendar_date date primary key,
        calendar_year integer not null,
        month integer not null,
        season_code text not null,
        season_label text not null
    """

TABLES["cultivation_extent"] = """
        source text not null,
        report_date date not null,
        season_label text not null,
        as_of_date date not null,
        crop_code text not null,
        region_name text not null,
        region_level text not null,
        region_group text not null,
        target_ha double precision,
        achievement_ha double precision,
        target_missing boolean not null,
        achievement_missing boolean not null,
        page_number integer not null,
        source_file text not null,
        primary key (report_date, crop_code, region_name)
    """


def load_env(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def save_env(path: Path, values: dict[str, str]) -> None:
    lines = [f"{key}={value}" for key, value in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reader_url(owner_url: str, password: str) -> str:
    parts = urlparse(owner_url)
    host = parts.hostname
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunparse(parts._replace(netloc=f"{READER_ROLE}:{password}@{host}"))


def local_frames(names: tuple[str, ...]) -> dict:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        return {
            name: con.execute(f"SELECT * FROM curated.{name}").fetchdf()
            for name in names
        }
    finally:
        con.close()


def copy_frame(cur, table: str, frame) -> None:
    columns = list(frame.columns)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, header=False, na_rep="")
    buffer.seek(0)
    column_list = ", ".join(columns)
    with cur.copy(
        f"COPY curated.{table} ({column_list}) FROM STDIN WITH (FORMAT CSV, NULL '')"
    ) as copy:
        copy.write(buffer.getvalue())


def grant_reader(cur) -> None:
    cur.execute(f"GRANT USAGE ON SCHEMA curated TO {READER_ROLE}")
    cur.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA curated TO {READER_ROLE}")


def ensure_reader(cur, password: str) -> None:
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (READER_ROLE,))
    statement = "ALTER ROLE {} WITH LOGIN PASSWORD {}" if cur.fetchone() else "CREATE ROLE {} WITH LOGIN PASSWORD {}"
    cur.execute(
        sql.SQL(statement).format(sql.Identifier(READER_ROLE), sql.Literal(password))
    )
    cur.execute(f"GRANT CONNECT ON DATABASE neondb TO {READER_ROLE}")
    grant_reader(cur)
    cur.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA curated GRANT SELECT ON TABLES TO {READER_ROLE}"
    )


def refresh_market_week(cur) -> None:
    cur.execute(f"DROP VIEW IF EXISTS curated.{VIEW_NAME}")
    cur.execute(VIEW_SQL)


def replace_tables(cur, names: tuple[str, ...], frames: dict) -> dict[str, int]:
    counts = {}
    cur.execute(f"DROP VIEW IF EXISTS curated.{VIEW_NAME}")
    for name in reversed(names):
        cur.execute(f"DROP TABLE IF EXISTS curated.{name}")
    for name in names:
        cur.execute(f"CREATE TABLE curated.{name} ({TABLES[name]})")
        copy_frame(cur, name, frames[name])
        cur.execute(f"SELECT count(*) FROM curated.{name}")
        counts[name] = cur.fetchone()[0]
    return counts


def owner_url_from_env() -> str:
    if not ENV_PATH.exists():
        raise FileNotFoundError(f"Missing {ENV_PATH}")
    owner_url = load_env(ENV_PATH).get("NEON_DATABASE_URL")
    if not owner_url:
        raise RuntimeError("NEON_DATABASE_URL is missing from .env")
    return owner_url


def publish() -> dict[str, int]:
    env = load_env(ENV_PATH)
    owner_url = owner_url_from_env()
    names = ("dim_market", "dim_crop", "harti_price_weekly")
    frames = local_frames(names)
    password = secrets.token_urlsafe(24)
    with psycopg.connect(owner_url, connect_timeout=30) as con:
        with con.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS curated")
            counts = replace_tables(cur, names, frames)
            refresh_market_week(cur)
            ensure_reader(cur, password)
        con.commit()
    env["NEON_READER_URL"] = reader_url(owner_url, password)
    save_env(ENV_PATH, env)
    return counts


def publish_cbsl() -> dict[str, int]:
    owner_url = owner_url_from_env()
    frames = local_frames(CBSL_TABLES)
    with psycopg.connect(owner_url, connect_timeout=30) as con:
        with con.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS curated")
            counts = replace_tables(cur, CBSL_TABLES, frames)
            refresh_market_week(cur)
            grant_reader(cur)
        con.commit()
    return counts


def publish_season() -> dict[str, int]:
    owner_url = owner_url_from_env()
    frames = local_frames(SEASON_TABLES)
    with psycopg.connect(owner_url, connect_timeout=30) as con:
        with con.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS curated")
            counts = replace_tables(cur, SEASON_TABLES, frames)
            refresh_market_week(cur)
            grant_reader(cur)
        con.commit()
    return counts


def publish_extent() -> dict[str, int]:
    owner_url = owner_url_from_env()
    frames = local_frames(EXTENT_TABLES)
    with psycopg.connect(owner_url, connect_timeout=30) as con:
        with con.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS curated")
            counts = replace_tables(cur, EXTENT_TABLES, frames)
            refresh_market_week(cur)
            grant_reader(cur)
        con.commit()
    return counts


def publish_weather() -> dict[str, int]:
    owner_url = owner_url_from_env()
    frames = local_frames(WEATHER_TABLES)
    with psycopg.connect(owner_url, connect_timeout=30) as con:
        with con.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS curated")
            counts = replace_tables(cur, WEATHER_TABLES, frames)
            refresh_market_week(cur)
            grant_reader(cur)
        con.commit()
    return counts


def publish_market_week() -> dict[str, int]:
    owner_url = owner_url_from_env()
    with psycopg.connect(owner_url, connect_timeout=30) as con:
        with con.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS curated")
            refresh_market_week(cur)
            grant_reader(cur)
            cur.execute(f"SELECT row_kind, count(*) FROM curated.{VIEW_NAME} GROUP BY row_kind ORDER BY row_kind")
            counts = {kind: count for kind, count in cur.fetchall()}
        con.commit()
    return counts


def main() -> None:
    choice = sys.argv[1] if len(sys.argv) > 1 else "prices"
    if choice == "weather":
        counts = publish_weather()
    elif choice == "season":
        counts = publish_season()
    elif choice == "cbsl":
        counts = publish_cbsl()
    elif choice == "extent":
        counts = publish_extent()
    elif choice == "view":
        counts = publish_market_week()
    else:
        counts = publish()
    print("published to Neon schema curated")
    for name, count in counts.items():
        print(f"  {name}: {count:,}")
    if choice not in ("weather", "season", "cbsl", "extent", "view"):
        print("reader connection string saved as NEON_READER_URL in .env")


if __name__ == "__main__":
    main()

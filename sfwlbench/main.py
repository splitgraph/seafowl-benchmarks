import functools
import json
import os
import shutil
import string
import random
import timeit
from abc import abstractmethod, ABC
from datetime import datetime
from numbers import Number

import duckdb
from typing import Any, NamedTuple, List, Dict, Optional

import pandas as pd
import psycopg2
import requests

from sfwlbench.queries import Query, QUERIES

SEAFOWL = "http://localhost:8080"

# Output benchmark results to the demo Seafowl instance
TARGET_SEAFOWL = "https://demo.seafowl.io"
TARGET_SEAFOWL_PW = os.environ.get("SEAFOWL_PASSWORD")

# PG 14; we assume it has parquet_fdw installed
POSTGRES = "postgresql://postgres@:5434/postgres"

PATTERN = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{}-{:02d}.parquet"
)

DATA_DIR = "taxi-data"


COLUMNS = {
    "VendorID": "INTEGER",
    "tpep_pickup_datetime": "TIMESTAMP",
    "tpep_dropoff_datetime": "TIMESTAMP",
    "passenger_count": "DOUBLE",
    "trip_distance": "DOUBLE",
    "RatecodeID": "DOUBLE",
    "store_and_fwd_flag": "VARCHAR",
    "PULocationID": "INTEGER",
    "DOLocationID": "INTEGER",
    "payment_type": "INTEGER",
    "fare_amount": "DOUBLE",
    "extra": "DOUBLE",
    "mta_tax": "DOUBLE",
    "tip_amount": "DOUBLE",
    "tolls_amount": "DOUBLE",
    "improvement_surcharge": "DOUBLE",
    "total_amount": "DOUBLE",
    "congestion_surcharge": "DOUBLE",
    "airport_fee": "DOUBLE",
}

COLUMN_STR = ",".join([f'"{c}"::{t} AS "{c}"' for c, t in COLUMNS.items()])

COLUMNS_PG = {
    "VendorID": "INTEGER",
    "tpep_pickup_datetime": "TIMESTAMP",
    "tpep_dropoff_datetime": "TIMESTAMP",
    "passenger_count": "DOUBLE PRECISION",
    "trip_distance": "DOUBLE PRECISION",
    "RatecodeID": "DOUBLE PRECISION",
    "store_and_fwd_flag": "VARCHAR",
    "PULocationID": "INTEGER",
    "DOLocationID": "INTEGER",
    "payment_type": "INTEGER",
    "fare_amount": "DOUBLE PRECISION",
    "extra": "DOUBLE PRECISION",
    "mta_tax": "DOUBLE PRECISION",
    "tip_amount": "DOUBLE PRECISION",
    "tolls_amount": "DOUBLE PRECISION",
    "improvement_surcharge": "DOUBLE PRECISION",
    "total_amount": "DOUBLE PRECISION",
    "congestion_surcharge": "DOUBLE PRECISION",
    # Ignore this column in PG: it changes its type halfway through the
    # dataset and parquet_fdw can't handle it
    # "airport_fee": "DOUBLE PRECISION",
}


class Database(ABC):
    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def query(self, sql: str) -> Any:
        pass

    @abstractmethod
    def database_id(self) -> str:
        """
        Short database ID, used to compare the performance of a database across time.

        Example: `seafowl`
        """
        pass

    @abstractmethod
    def database_version_id(self) -> str:
        """Full database ID that includes the version.

        The benchmark results for the same query on the same hardware for the same version ID shouldn't change

        Example: `duckdb-0.4.0`, `seafowl-0.1.0-dev.1`, `seafowl-0.1.0-dev.1-simd`
        """
        pass

    @abstractmethod
    def database_version_id_human(self) -> str:
        """Human-readable database string.

        Example: `DuckDB 0.4.0`, `Seafowl 0.1.0-dev.1 with SIMD`
        """
        pass


def random_table_name():
    return "".join(random.choices(string.ascii_lowercase, k=16))


def query_seafowl(endpoint: str, sql: str, access_token: Optional[str] = None) -> Any:
    headers = {"Content-Type": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    response = requests.post(f"{endpoint}/q", json={"query": sql}, headers=headers)

    if not response.ok:
        print(response.text)
    response.raise_for_status()
    if response.text:
        return [json.loads(t) for t in response.text.strip().split("\n")]
    return None


class Seafowl(Database):
    def database_id(self) -> str:
        return "seafowl"

    def database_version_id(self) -> str:
        # TODO expose it somehow?
        return "seafowl-0.1.0-dev.1"

    def database_version_id_human(self) -> str:
        return "Seafowl 0.1.0-dev.1"

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def setup(self) -> None:
        files = sorted(os.listdir(DATA_DIR))
        basepath = os.path.abspath(DATA_DIR)

        # Check if table already exists
        if self.query(
            "SELECT TRUE AS exists FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = 'tripdata'"
        ) == [{"exists": True}]:
            print("Seafowl table already exists, skipping")
            return

        # Create the table to store the data
        # query_seafowl(f"DROP TABLE IF EXISTS tripdata;")

        staging_table_names = []
        for filename in files:
            table_name = random_table_name()
            staging_table_names.append(table_name)

            self.query(
                f"CREATE EXTERNAL TABLE {table_name} STORED AS PARQUET LOCATION '{os.path.join(basepath, filename)}';"
            )

        # Merge all files into a table
        self.query(
            f"CREATE TABLE tripdata AS "
            + " UNION ALL ".join(
                f"SELECT {COLUMN_STR} FROM staging.{t}" for t in staging_table_names
            )
        )

    def query(self, sql: str) -> Any:
        return query_seafowl(self.endpoint, sql)


class DuckDB(Database):
    def database_id(self) -> str:
        return "duckdb"

    def database_version_id(self) -> str:
        return f"duckdb-{duckdb.__version__}"

    def database_version_id_human(self) -> str:
        return f"DuckDB {duckdb.__version__}"

    def __init__(self) -> None:
        self.conn = duckdb.connect(":memory:")

    def setup(self) -> None:
        # Create a UNION view of all Parquet files
        basepath = os.path.abspath(DATA_DIR)

        self.conn.execute(
            f"CREATE VIEW tripdata AS SELECT * FROM '{os.path.join(basepath, '*.parquet')}'"
        )
        self.conn.commit()

    def query(self, sql: str) -> Any:
        return self.conn.execute(sql).fetchall()


class PostgreSQLParquet(Database):
    def __init__(self, dsn: str):
        self.conn = psycopg2.connect(dsn)

    def setup(self) -> None:
        files = sorted(os.listdir(DATA_DIR))
        basepath = os.path.abspath(DATA_DIR)

        if self.query(
            "SELECT TRUE AS exists FROM information_schema.tables "
            "WHERE table_schema = 'parquet' AND table_name = 'tripdata'"
        ) == [(True,)]:
            print("PostgreSQL Parquet table already exists, skipping")
            return

        self.query("CREATE SCHEMA IF NOT EXISTS parquet")

        self.query(
            "create server if not exists  parquet_srv foreign data wrapper parquet_fdw;"
            "create user mapping if not exists for postgres server parquet_srv options (user 'postgres');"
        )

        self.query(
            "CREATE FOREIGN TABLE parquet.tripdata ("
            + ",".join(f'"{c}" {t}' for c, t in COLUMNS_PG.items())
            + f") SERVER parquet_srv OPTIONS (filename '{' '.join(os.path.join(basepath, f) for f in files)}', "
            f"use_threads 'true')"
        )
        self.conn.commit()

    def query(self, sql: str) -> Any:
        with self.conn.cursor() as cur:
            cur.execute("SET search_path TO 'parquet';" + sql)
            if cur.description is None:
                return None
            return cur.fetchall()

    def database_id(self) -> str:
        return "postgresql-parquet-fdw"

    def database_version_id(self) -> str:
        return "postgresql-14.4-parquet-fdw"

    def database_version_id_human(self) -> str:
        return "PostgreSQL 14.4 (parquet_fdw)"


class PostgreSQL(PostgreSQLParquet):
    def __init__(self, dsn: str):
        super(PostgreSQL, self).__init__(dsn)

    def setup(self) -> None:
        super(PostgreSQL, self).setup()
        if self.query(
            "SELECT TRUE AS exists FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = 'tripdata'"
        ) == [(True,)]:
            print("PostgreSQL table already exists, skipping")
            return

        self.query("CREATE TABLE tripdata AS SELECT * FROM parquet.tripdata")
        self.query("CREATE INDEX ON tripdata(tpep_pickup_datetime)")
        self.query("CREATE INDEX ON tripdata(tpep_dropoff_datetime)")
        self.query('CREATE INDEX ON tripdata("PULocationID")')
        self.query('CREATE INDEX ON tripdata("DOLocationID")')
        self.conn.commit()

    def query(self, sql: str) -> Any:
        with self.conn.cursor() as cur:
            cur.execute(sql)
            if cur.description is None:
                return None
            return cur.fetchall()

    def database_id(self) -> str:
        return "postgresql"

    def database_version_id(self) -> str:
        return "postgresql-14.4"

    def database_version_id_human(self) -> str:
        return "PostgreSQL 14.4"


class Timing(NamedTuple):
    times: List[float]
    per_attempt: int


def bench_query(query: Query, db: Database) -> Timing:
    t = timeit.Timer(stmt=lambda: db.query(query.sql))
    per_attempt = t.autorange()[0]

    result = t.repeat(5, per_attempt)
    return Timing(times=[r / per_attempt for r in result], per_attempt=per_attempt)


def setup_taxi() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    for year in [2020, 2021]:
        for month in range(1, 13):
            url = PATTERN.format(year, month)
            filename = url.rsplit("/")[-1]
            target_path = os.path.join(DATA_DIR, filename)

            if os.path.exists(target_path):
                print(f"{filename} already exists, skipping")
                continue

            print(f"{url} -> {target_path}")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                r.raw.read = functools.partial(r.raw.read, decode_content=True)
                with open(target_path, "wb") as f:
                    shutil.copyfileobj(r.raw, f, length=16 * 1024 * 1024)


def make_queries_df(bench_id: str) -> pd.DataFrame:
    return pd.DataFrame([{"bench_id": bench_id, **q._asdict()} for q in QUERIES])


def make_timings_df(
    bench_id: str, timings: Dict[Database, Dict[Query, Timing]]
) -> pd.DataFrame:
    rows = []
    for database, db_timings in timings.items():
        for query, timing in db_timings.items():
            for t in timing.times:
                rows.append(
                    {
                        "bench_id": bench_id,
                        "database_id": database.database_id(),
                        "query_id": query.id,
                        "time": t,
                        "loops": timing.per_attempt,
                    }
                )

    return pd.DataFrame(rows)


def make_database_df(bench_id: str, databases: List[Database]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "bench_id": bench_id,
                "database_id": db.database_id(),
                "version_id": db.database_version_id(),
                "database_version_human": db.database_version_id_human(),
            }
            for db in databases
        ]
    )


def emit_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.20f}"

    if isinstance(value, Number) and not isinstance(value, bool):
        return str(value)

    if isinstance(value, datetime):
        return f"'{value.isoformat()}'"

    quoted = str(value).replace("'", "''")
    return f"'{quoted}'"


def make_seafowl_query(df: pd.DataFrame, table: str, schema: str) -> str:
    query = f"INSERT INTO {schema}.{table} (" + ",".join(df.columns) + ") VALUES \n"
    query += ",\n".join(
        "(" + ",".join(emit_value(v) for v in row) + ")"
        for row in df.itertuples(index=False)
    )

    return query


def output_results(
    seafowl: str,
    bench_df: pd.DataFrame,
    database_df: pd.DataFrame,
    query_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    schema: str = "benchmarks",
    access_token: Optional[str] = None,
):
    schema_exists = query_seafowl(
        seafowl,
        f"SELECT 1 AS exists FROM information_schema.tables WHERE table_schema = '{schema}'",
    )

    if not schema_exists:
        for query in [
            f"CREATE SCHEMA {schema}",
            f"""CREATE TABLE {schema}.database (
        bench_id VARCHAR,
        database_id VARCHAR,
        version_id VARCHAR,
        database_version_human VARCHAR
    );""",
            f"""CREATE TABLE {schema}.benchmark (
        id VARCHAR,
        run_at TIMESTAMP
    );""",
            f""" CREATE TABLE {schema}.query (
        bench_id VARCHAR,
        id VARCHAR,
        sql VARCHAR,
        notes VARCHAR
    );""",
            f"""
    CREATE TABLE {schema}.timing (
        bench_id VARCHAR,
        database_id VARCHAR,
        query_id VARCHAR,
        time DOUBLE,
        loops INTEGER
    );""",
        ]:
            query_seafowl(
                seafowl,
                query,
                access_token,
            )

    # Append data to the tables
    headers = {}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    for table_name, df in [
        ("database", database_df),
        ("benchmark", bench_df),
        ("query", query_df),
        ("timing", timing_df),
    ]:
        with open(f"{table_name}.csv", "wb") as f:
            df.to_csv(f, index=False)

        # csv = df.to_csv(header=True, index=False)
        # buf = StringIO(csv)

        # TODO: can't dump to CSV/Parquet and upload (schema mismatch)
        # buf = BytesIO()
        # df.to_parquet(buf)
        # buf.seek(0)
        #
        # with open(f"{table_name}.parquet", "wb") as f:
        #     f.write(buf.read())
        # buf.seek(0)
        #
        # # csv = df.to_csv(header=True, index=False)
        # # buf = StringIO(csv)
        #
        # print(f"Uploading {table_name} -> {seafowl}")
        # multipart_form_data = {
        #     "data": (f"{table_name}.parquet", buf),
        # }
        #
        # response = requests.post(
        #     f"{seafowl}/upload/{schema}/{table_name}",
        #     files=multipart_form_data,
        #     headers=headers,
        # )
        # if not response.ok:
        #     print(response.text)
        # response.raise_for_status()

        query = make_seafowl_query(df, table_name, schema)
        print(query)
        query_seafowl(seafowl, query, access_token)


if __name__ == "__main__":
    setup_taxi()

    databases = [
        PostgreSQL(POSTGRES),
        PostgreSQLParquet(POSTGRES),
        DuckDB(),
        Seafowl(SEAFOWL),
    ]

    print("Setting up databases...")
    for db in databases:
        db.setup()

    timings: Dict[Database, Dict[Query, Timing]] = {}

    bench_time = datetime.now()
    bench_id = f"bench-{bench_time.strftime('%Y%m%d-%H%M%S')}"

    for db in databases:
        timings[db] = {}

        print(f"Benchmarking {db.database_version_id_human()}...")
        for query in QUERIES:
            print(f"Running {query.id}...")
            timings[db][query] = bench_query(query, db)

    bench_df = pd.DataFrame([{"id": bench_id, "run_at": bench_time}])
    database_df = make_database_df(bench_id, databases)
    query_df = make_queries_df(bench_id)
    timing_df = make_timings_df(bench_id, timings)

    print("Done. Outputting results to Seafowl")
    print(bench_df)
    print(database_df)
    print(query_df)
    print(timing_df)

    output_results(
        TARGET_SEAFOWL,
        bench_df,
        database_df,
        query_df,
        timing_df,
        "benchmarks",
        TARGET_SEAFOWL_PW,
    )

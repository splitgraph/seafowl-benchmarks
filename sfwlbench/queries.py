from typing import NamedTuple


class Query(NamedTuple):
    id: str
    sql: str
    notes: str


QUERIES = [
    Query(
        "count_all",
        "SELECT COUNT(*) FROM tripdata",
        "Basic count of all values (satisfied via the metadata",
    ),
    Query("select_one", "SELECT 1", "Test baseline latency"),
    Query(
        "month_agg",
        """SELECT
    date_trunc('month', tpep_pickup_datetime) AS month, 
    COUNT(*) AS total_trips,
    SUM(total_amount) AS total_amount 
FROM tripdata
GROUP BY 1 ORDER BY 1 ASC""",
        "Aggregate statistics by month",
    ),
    Query(
        "aggregate_and_order",
        """SELECT
    "PULocationID",
    "DOLocationID",
    COUNT(*)
FROM tripdata 
GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 10""",
        "10 most popular pickup, dropoff location pairs",
    ),
    # NB: weird issue in Seafowl when using date_trunc('month', ...) or normal comparisons
    # (< '2020-04-01')
    Query(
        "filter_and_aggregate_and_order",
        """SELECT
    \"DOLocationID\",
    SUM(passenger_count) AS total_passengers
FROM tripdata
WHERE EXTRACT(month FROM tpep_pickup_datetime) = 3 
    AND EXTRACT(year FROM tpep_pickup_datetime) = 2020 
GROUP BY 1 ORDER BY 2 DESC LIMIT 10""",
        "Top 10 destinations in March 2020 by total passenger count",
    ),
    Query(
        "window_aggregate_join",
        """WITH total_passengers_per_location_per_month AS (SELECT
    date_trunc('month', tpep_pickup_datetime) AS month,
    "PULocationID" AS location_id,
    SUM(passenger_count) AS total_passengers
    FROM tripdata
    WHERE
        -- Remove weird values from other years
        EXTRACT(YEAR FROM tpep_pickup_datetime) IN (2020, 2021)
    GROUP BY 1, 2
    HAVING 
        -- Ignore locations with <1000 passengers / month
        COALESCE(SUM(passenger_count), 0) > 1000
),
previous_passengers AS (SELECT
    month,
    location_id,
    total_passengers,
    LAG(total_passengers) OVER (PARTITION BY location_id ORDER BY month) AS previous_passengers
    FROM total_passengers_per_location_per_month
    ORDER BY month, location_id
),
location_passenger_change AS (SELECT 
    month,
    location_id,
    total_passengers,
    previous_passengers, 
    total_passengers / previous_passengers - 1 AS fraction_change
    FROM previous_passengers
    WHERE COALESCE(previous_passengers, 0) != 0
),
biggest_mom_growth AS (SELECT
    month, location_id, total_passengers, previous_passengers, fraction_change
    FROM (
        SELECT *, ROW_NUMBER() OVER(PARTITION BY month ORDER BY fraction_change DESC) AS rn
        FROM location_passenger_change
        ORDER BY month, location_id
    ) AS r
    WHERE rn = 1
),
biggest_mom_decline AS (SELECT
    month, location_id, total_passengers, previous_passengers, fraction_change
    FROM (
        SELECT *, ROW_NUMBER() OVER(PARTITION BY month ORDER BY fraction_change ASC) AS rn
        FROM location_passenger_change
        ORDER BY month, location_id
    ) AS r
    WHERE rn = 1
),
month_stats AS (SELECT month,
    SUM(total_passengers) AS total_passengers,
    SUM(previous_passengers) AS previous_passengers,
    SUM(total_passengers) / SUM(previous_passengers) - 1 AS fraction_change
    FROM previous_passengers
    GROUP BY month
)
SELECT
    month_stats.month,
    month_stats.total_passengers,
    month_stats.fraction_change,
    biggest_mom_growth.location_id AS biggest_growth_location_id,
    biggest_mom_growth.total_passengers AS biggest_growth_total_passengers,
    biggest_mom_growth.fraction_change AS biggest_growth_fraction_change,
    biggest_mom_decline.location_id AS biggest_decline_location_id,
    biggest_mom_decline.total_passengers AS biggest_decline_total_passengers,
    biggest_mom_decline.fraction_change AS biggest_decline_fraction_change
FROM month_stats
INNER JOIN biggest_mom_growth ON month_stats.month = biggest_mom_growth.month
INNER JOIN biggest_mom_decline ON month_stats.month = biggest_mom_decline.month
ORDER BY month_stats.month ASC;""",
        "Dropoff location with largest month-over-month total passenger count growth/decline",
    ),
]
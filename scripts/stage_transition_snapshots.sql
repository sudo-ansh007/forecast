-- Stage-transition snapshots: one row per A->B stage move, capturing how the
-- opportunity looked at that moment plus the meetings held up to that point.
--
-- Timestamp note: fact_opportunity.created_date is the DEAL creation date and is
-- constant across every object_version. modified_date is the per-version
-- timestamp, so it is what "as of this transition" has to key on.

COPY (

WITH opp AS (
    SELECT
        id,
        display_id,
        object_version,
        modified_date,
        created_date,
        state,
        probability,
        forecast_category,
        target_close_date,
        amount,
        contact_ids,
        custom_fields,
        json_extract_string(stage_json, '$.stage_id') AS stage_id,
        json_extract_string(stage_json, '$.name')     AS stage_name
    FROM 'devrev_fact_opportunity_*.parquet'
    WHERE is_deleted = false
      AND stage_json IS NOT NULL
      AND stage_json <> ''
),

history AS (
    SELECT
        *,
        LAG(stage_id)      OVER w AS prev_stage_id,
        LAG(stage_name)    OVER w AS prev_stage_name,
        LAG(modified_date) OVER w AS prev_version_date
    FROM opp
    WINDOW w AS (PARTITION BY id ORDER BY object_version)
),

-- gaps-and-islands: one run id per uninterrupted spell in a stage
marked AS (
    SELECT
        *,
        SUM(CASE WHEN stage_id IS DISTINCT FROM prev_stage_id THEN 1 ELSE 0 END)
            OVER (PARTITION BY id ORDER BY object_version ROWS UNBOUNDED PRECEDING) AS stage_run
    FROM history
),

runs AS (
    SELECT
        *,
        MIN(modified_date) OVER (PARTITION BY id, stage_run) AS stage_entry_date,
        MIN(modified_date) OVER (PARTITION BY id)            AS first_seen_date
    FROM marked
),

transitions AS (
    SELECT
        *,
        LAG(stage_entry_date) OVER (PARTITION BY id ORDER BY object_version) AS prev_stage_entry_date
    FROM runs
    QUALIFY prev_stage_id IS NOT NULL
        AND stage_id <> prev_stage_id
),

/* dim_link is current state and holds 31,735 meeting->opportunity edges;
   fact_link is the versioned log on the same rolling window as
   fact_opportunity, so on its own it sees only 10,263 of them. Union both:
   dim_link supplies the edges created before the window opened, fact_link the
   1,593 that dim_link no longer reflects.

   Only meeting -> opportunity appears in either table today, but keep both
   directions so this survives a schema that starts emitting the reverse. */
opportunity_meetings AS (
    SELECT DISTINCT target_id AS opportunity_id, source_id AS meeting_id
    FROM 'devrev_dim_link_0.parquet'
    WHERE source_object_type = 'meeting' AND target_object_type = 'opportunity'
    UNION
    SELECT DISTINCT source_id AS opportunity_id, target_id AS meeting_id
    FROM 'devrev_dim_link_0.parquet'
    WHERE source_object_type = 'opportunity' AND target_object_type = 'meeting'
    UNION
    SELECT DISTINCT target_id AS opportunity_id, source_id AS meeting_id
    FROM 'devrev_fact_link_*.parquet'
    WHERE is_deleted = false
      AND source_object_type = 'meeting'
      AND target_object_type = 'opportunity'
    UNION
    SELECT DISTINCT source_id AS opportunity_id, target_id AS meeting_id
    FROM 'devrev_fact_link_*.parquet'
    WHERE is_deleted = false
      AND source_object_type = 'opportunity'
      AND target_object_type = 'meeting'
),

-- 10% of linked meetings never got an ended_date; fall back to the scheduled
-- slot rather than dropping them.
meetings AS (
    SELECT
        id AS meeting_id,
        COALESCE(ended_date, scheduled_date) AS meeting_ts
    FROM 'devrev_dim_meeting_0.parquet'
    WHERE is_deleted = false
      AND COALESCE(ended_date, scheduled_date) IS NOT NULL
),

meetings_as_of AS (
    SELECT
        t.id,
        t.object_version,
        COUNT(DISTINCT m.meeting_id) AS meetings_conducted,
        LIST(DISTINCT m.meeting_id) FILTER (WHERE m.meeting_id IS NOT NULL) AS meeting_ids,
        MAX(m.meeting_ts) AS latest_meeting_conducted_date
    FROM transitions t
    LEFT JOIN opportunity_meetings om
        ON t.id = om.opportunity_id
    LEFT JOIN meetings m
        ON om.meeting_id = m.meeting_id
       AND m.meeting_ts <= t.modified_date
    GROUP BY t.id, t.object_version
),

final AS (
    SELECT
        t.id                                        AS opportunity_id,
        t.display_id,
        t.object_version,
        t.modified_date                             AS transition_date,
        t.prev_stage_name                           AS prev_stage,
        t.stage_name                                AS stage,
        t.created_date                              AS opportunity_created_date,

        -- state of the deal as written by this version
        t.state,
        t.probability,
        t.forecast_category,
        t.target_close_date,
        t.amount,

        json_extract_string(t.custom_fields, '$.tnt__opportunity_type') AS opportunity_type,
        json_extract_string(t.custom_fields, '$.tnt__segment')          AS segment,
        CASE WHEN json_extract(t.custom_fields, '$.tnt__champion') IS NOT NULL
             THEN 1 ELSE 0 END                                          AS has_champion,
        COALESCE(len(t.contact_ids), 0)                                 AS contacts_length,
        TRY_CAST(json_extract_string(t.custom_fields, '$.tnt__contract_months') AS DOUBLE)
                                                                        AS contract_months,

        TRY_CAST(json_extract_string(t.custom_fields, '$.tnt__days_in_current_stage') AS DOUBLE)
                                                                        AS days_in_current_stage,
        TRY_CAST(json_extract_string(t.custom_fields, '$.tnt__days_in_sales_cycle') AS DOUBLE)
                                                                        AS days_in_sales_cycle,

        -- measured from the version log instead of the nightly-refreshed custom
        -- field; censored when the stage was entered before the log window opens
        DATE_DIFF('day', t.prev_stage_entry_date, t.modified_date)      AS days_in_prev_stage_observed,
        CASE WHEN t.prev_stage_entry_date <= t.first_seen_date THEN 1 ELSE 0 END
                                                                        AS prev_stage_entry_censored,
        DATE_DIFF('day', t.created_date, t.modified_date)               AS days_in_sales_cycle_computed,

        COALESCE(ma.meetings_conducted, 0)          AS meetings_conducted,
        ma.meeting_ids,
        ma.latest_meeting_conducted_date,
        DATE_DIFF('day', ma.latest_meeting_conducted_date, t.modified_date)
                                                    AS days_since_latest_meeting,

        -- <60s means the write came from a bulk/integration replay, not a rep
        DATE_DIFF('second', t.prev_version_date, t.modified_date) AS secs_since_prev_version

    FROM transitions t
    LEFT JOIN meetings_as_of ma
        ON t.id = ma.id
       AND t.object_version = ma.object_version
)

SELECT
    *,
    CASE WHEN days_in_sales_cycle > 0
         THEN days_in_current_stage / days_in_sales_cycle
    END AS stall_ratio
FROM final
ORDER BY opportunity_id, object_version

) TO 'stage_transition_snapshots.csv' (HEADER, DELIMITER ',');

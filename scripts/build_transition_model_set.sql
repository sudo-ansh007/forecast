-- Model-ready stage-transition set for as.ipynb.
--
-- Labelling follows the original stage_transitions_model.csv: the origin stage
-- must be 1-7, advanced=1 for a forward move, advanced=0 for a move to
-- closed-lost. Backward moves have no label and are dropped.
--
-- Grandfather rule: every row already in stage_transitions_model.csv is kept as
-- is. Rows that are NEW relative to that file are only admitted if they do not
-- originate in 1-profile -- that stage is 66% of the old dataset, a third of its
-- rows have no meeting at all, and it advances at 34% versus 61-97% for stages
-- 4-7. Keeping the existing rows preserves comparability with runs already done.
--
-- Output columns 1-15 match the old file exactly so the notebook can read this
-- as a drop-in; the diagnostics are appended after.
--
-- Build the input first:  duckdb < scripts/stage_transition_snapshots.sql

COPY (

WITH snap AS (
    SELECT *
    FROM read_csv('stage_transition_snapshots.csv',
                  delim=',', header=true, quote='"', sample_size=-1)
),

ranked AS (
    SELECT
        *,
        TRY_CAST(split_part(prev_stage, '-', 1) AS INT) AS from_rank,
        TRY_CAST(split_part(stage, '-', 1) AS INT)      AS to_rank
    FROM snap
),

model_rows AS (
    SELECT
        *,
        CASE WHEN to_rank = 9 THEN 0 ELSE 1 END AS advanced
    FROM ranked
    WHERE from_rank BETWEEN 1 AND 7
      AND (to_rank = 9 OR to_rank > from_rank)
),

-- line 133 of the old file is malformed (missing opportunity_id/display_id),
-- so it never matches and simply falls out
existing AS (
    SELECT DISTINCT opportunity_id, snapshot_version
    FROM read_csv('stage_transitions_model.csv',
                  delim=',', header=true, quote='"', sample_size=-1,
                  ignore_errors=true)
),

admitted AS (
    SELECT
        m.*,
        CASE WHEN e.opportunity_id IS NULL THEN 1 ELSE 0 END AS is_new_entry
    FROM model_rows m
    LEFT JOIN existing e
        ON e.opportunity_id = m.opportunity_id
       AND e.snapshot_version = m.object_version
    WHERE e.opportunity_id IS NOT NULL       -- already in the shipped dataset
       OR m.prev_stage <> '1-profile'        -- new entry: no stage-1 origins
),

/* stall_ratio is undefined on 490 rows (days_in_sales_cycle null or zero).
   Fill from the mean of the OBSERVED rows in the same origin stage, weighted by
   days_in_sales_cycle so long-running deals carry more of the estimate.

   Conditioning on stage is what matters here: the weighted mean runs 0.99 at
   1-profile down to 0.27 at 7-submit_for_close, and 94% of the missing rows sit
   in 1-profile. A single global fill (0.80) would push those rows ~0.19 below
   where their own stage sits -- which is what the previous dataset did. */
stage_mean AS (
    SELECT
        prev_stage,
        count(*) AS n,
        sum(stall_ratio * days_in_sales_cycle)
            / nullif(sum(days_in_sales_cycle), 0) AS wmean
    FROM admitted
    WHERE stall_ratio IS NOT NULL AND days_in_sales_cycle > 0
    GROUP BY 1
),

global_mean AS (
    SELECT sum(stall_ratio * days_in_sales_cycle)
             / nullif(sum(days_in_sales_cycle), 0) AS gwmean
    FROM admitted
    WHERE stall_ratio IS NOT NULL AND days_in_sales_cycle > 0
),

filled AS (
    SELECT
        a.*,
        CASE WHEN a.stall_ratio IS NULL THEN 1 ELSE 0 END AS stall_ratio_imputed,
        CASE
            WHEN a.stall_ratio IS NOT NULL THEN a.stall_ratio
            ELSE least(1.0, greatest(0.0,
                -- shrink the stage mean toward global; K=20 matches the
                -- convention in build_features.py, so a thin stage barely
                -- moves off global while 1-profile keeps its own number
                (coalesce(sm.wmean, g.gwmean) * coalesce(sm.n, 0) + g.gwmean * 20)
                    / (coalesce(sm.n, 0) + 20)
                -- deterministic jitter in [-JITTER, +JITTER], keyed on the row
                -- so rebuilds reproduce exactly and no single spike value
                -- appears for a tree to split on
                + (CAST(hash(a.opportunity_id || '#' || a.object_version) % 2001 AS BIGINT) - 1000)
                    / 1000.0 * 0.05
            ))
        END AS stall_ratio_filled
    FROM admitted a
    LEFT JOIN stage_mean sm ON sm.prev_stage = a.prev_stage
    CROSS JOIN global_mean g
),

/* LLM meeting signals, aggregated over the meetings that had already happened
   by each transition. Signals live one row per meeting; a meeting linked to
   several opportunities is repeated there, so collapse to the meeting first.

   Source is momentum_signals.csv: the v3.1 schema scored by Claude Opus 5. The
   earlier sentiment columns are gone deliberately. They came from a model whose
   test-retest reliability on this task was r = 0.14, which caps its correlation
   with any outcome at 0.37 -- see scripts/signal_reliability.py. Opus scores
   r = 0.91 on the same transcripts. */
meeting_signal AS (
    SELECT
        m.id AS meeting_urn,
        COALESCE(m.ended_date, m.scheduled_date) AS meeting_ts,
        s.momentum,
        s.customer_next_step,
        s.explicit_blocker,
        s.timeline_named,
        s.multi_stakeholder,
        s.engagement,
        s.confidence
    FROM 'devrev_dim_meeting_0.parquet' m
    JOIN (
        SELECT
            meeting_id,
            any_value(momentum)           AS momentum,
            any_value(customer_next_step) AS customer_next_step,
            any_value(explicit_blocker)   AS explicit_blocker,
            any_value(timeline_named)     AS timeline_named,
            any_value(multi_stakeholder)  AS multi_stakeholder,
            any_value(engagement)         AS engagement,
            any_value(confidence)         AS confidence
        FROM read_csv('transcripts_train/momentum_signals.csv',
                      delim=',', header=true, quote='"', sample_size=-1)
        GROUP BY meeting_id
    ) s ON s.meeting_id = m.display_id
    WHERE m.is_deleted = false
),

row_meetings AS (
    SELECT
        opportunity_id,
        object_version,
        transition_date,
        unnest(TRY_CAST(meeting_ids AS VARCHAR[])) AS meeting_urn
    FROM filled
    WHERE meeting_ids IS NOT NULL
),

row_signals AS (
    SELECT
        rm.opportunity_id,
        rm.object_version,
        count(*)                                       AS sig_count,
        round(avg(ms.momentum), 4)                     AS sig_momentum_mean,
        min(ms.momentum)                               AS sig_momentum_min,
        max(ms.momentum)                               AS sig_momentum_max,
        sum(ms.momentum)                               AS sig_momentum_sum,
        -- ordered by the meeting clock, not by position in meeting_ids: LIST()
        -- does not promise an order
        arg_max(ms.momentum, ms.meeting_ts)            AS sig_momentum_last,
        arg_min(ms.momentum, ms.meeting_ts)            AS sig_momentum_first,
        max(ms.customer_next_step)                     AS sig_next_step_any,
        max(ms.explicit_blocker)                       AS sig_blocker_any,
        round(avg(ms.explicit_blocker), 4)             AS sig_blocker_rate,
        max(ms.timeline_named)                         AS sig_timeline_any,
        max(ms.multi_stakeholder)                      AS sig_multi_stakeholder_any,
        round(avg(ms.engagement), 4)                   AS sig_engagement_mean,
        round(avg(ms.confidence), 4)                   AS sig_confidence_mean,
        DATE_DIFF('day', max(ms.meeting_ts), any_value(rm.transition_date))
                                                       AS sig_days_since_last_scored
    FROM row_meetings rm
    JOIN meeting_signal ms ON ms.meeting_urn = rm.meeting_urn
    GROUP BY rm.opportunity_id, rm.object_version
)

SELECT
    f.opportunity_id,
    f.display_id,
    f.object_version   AS snapshot_version,
    f.transition_date  AS snapshot_date,
    f.prev_stage       AS current_stage,
    f.stage            AS next_stage,
    opportunity_type,
    segment,
    days_in_current_stage,
    days_in_sales_cycle,
    stall_ratio_filled AS stall_ratio,
    has_champion,
    contract_months  AS contract_length,
    advanced,
    meetings_conducted,

    -- diagnostics, appended so the 15 columns above stay drop-in compatible
    stall_ratio_imputed,
    is_new_entry,
    contacts_length,
    days_since_latest_meeting,
    latest_meeting_conducted_date,
    days_in_prev_stage_observed,
    prev_stage_entry_censored,
    secs_since_prev_version,
    meeting_ids,

    -- LLM meeting signals as of this transition; 0 / NULL where no meeting
    -- before the transition had a usable transcript
    COALESCE(rs.sig_count, 0) AS sig_count,
    rs.sig_momentum_mean,
    rs.sig_momentum_last,
    rs.sig_momentum_first,
    rs.sig_momentum_min,
    rs.sig_momentum_max,
    rs.sig_momentum_sum,
    rs.sig_next_step_any,
    rs.sig_blocker_any,
    rs.sig_blocker_rate,
    rs.sig_timeline_any,
    rs.sig_multi_stakeholder_any,
    rs.sig_engagement_mean,
    rs.sig_confidence_mean,
    rs.sig_days_since_last_scored
FROM filled f
LEFT JOIN row_signals rs
    ON rs.opportunity_id = f.opportunity_id
   AND rs.object_version = f.object_version
ORDER BY f.opportunity_id, snapshot_version

) TO 'stage_transitions_model_v2.csv' (HEADER, DELIMITER ',');

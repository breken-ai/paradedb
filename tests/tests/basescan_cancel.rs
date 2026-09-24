// Copyright (c) 2023-2026 ParadeDB, Inc.
//
// This file is part of ParadeDB - Postgres for Search and Analytics
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

//! A cancel or `pg_terminate_backend` of a parallel columnar Base Scan's leader sends SIGTERM to
//! its workers. A worker that handles it inside the tokio `block_on` driving its scan tears down
//! the live runtime, and an assert-enabled server aborts and resets the cluster. This test
//! signals a running scan from another connection and checks that a witness connection open
//! throughout survives.

use anyhow::Result;
use rstest::*;
use sqlx::{AssertSqlSafe, Executor, PgConnection};
use std::time::{Duration, Instant};
use tests::fixtures::*;
use tokio::time::sleep;

// Many small segments and a selective query: each worker spends its time claiming segments that
// have no matches, all inside one `block_on`, where a die lands.
const SEGMENTS: usize = 100;

const SETUP_SQL: &str = r#"
CREATE EXTENSION IF NOT EXISTS pg_search CASCADE;
CREATE TABLE bs_cancel (id bigserial primary key, body text, grp int, extra int);
CREATE INDEX bs_cancel_idx ON bs_cancel USING bm25 (id, body, grp);
"#;

const INSERT_SEGMENT: &str = r#"
INSERT INTO bs_cancel (body, grp, extra)
SELECT md5(random()::text) || ' ' || md5(random()::text), g % 10, g
FROM generate_series(1, 2000) AS g
"#;

const PARALLEL_GUCS: &str = r#"
SET max_parallel_workers_per_gather TO 2;
SET max_parallel_workers TO 8;
SET parallel_setup_cost TO 0;
SET parallel_tuple_cost TO 0;
SET paradedb.min_rows_per_worker TO 0;
"#;

// Each query matches nothing, so every segment is walked: md5 bodies are hex, and `extra` is
// never negative. `extra` is not indexed, so the second is evaluated per doc as a heap filter.
const QUERIES: [(&str, &str, &str); 2] = [
    (
        "regex",
        "SELECT grp FROM bs_cancel WHERE id @@@ paradedb.regex('body', '.*q.*z.*') AND grp = 1",
        "regex",
    ),
    (
        "heapfilter",
        "SELECT grp FROM bs_cancel WHERE id @@@ paradedb.all() AND extra = -1",
        "heap_filter",
    ),
];

// A scan runs for tens of milliseconds, so a longer wait would mostly land between queries.
const SIGNAL_DELAY: Duration = Duration::from_millis(20);
const ATTEMPTS: usize = 3;
const TARGET_LOOP_TIMEOUT: Duration = Duration::from_secs(30);

// A crash-triggered cluster reset reports `57P02` (crash_shutdown) instead, which fails the test.
const QUERY_CANCELED: &str = "57014";
const ADMIN_SHUTDOWN: &str = "57P01";

async fn assert_query_plans_as_parallel_columnar(
    conn: &mut PgConnection,
    query: &str,
    tantivy_query_contains: &str,
) -> Result<()> {
    conn.execute(PARALLEL_GUCS).await?;
    let rows: Vec<(String,)> = sqlx::query_as(AssertSqlSafe(format!(
        "EXPLAIN (COSTS OFF, VERBOSE) {query}"
    )))
    .fetch_all(&mut *conn)
    .await?;
    let explain = rows
        .into_iter()
        .map(|(line,)| line)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        explain.contains("Parallel Custom Scan (ParadeDB Base Scan)")
            && explain.contains("ColumnarExecState")
            && explain.contains(tantivy_query_contains),
        "Base Scan cancel test query must plan as a parallel columnar Base Scan:\n{explain}"
    );
    Ok(())
}

/// Each time `target_app`'s backend shows up running the scan, wait briefly and run
/// `signal_fn(pid)` (`pg_cancel_backend` or `pg_terminate_backend`) on it, until the target's
/// loop ends. A cancel that lands between two queries finds the backend idle and is ignored, so
/// a single signal isn't enough when each query is this short.
async fn signal_until_stopped(
    signaller: &mut PgConnection,
    target_app: &str,
    signal_fn: &str,
    target: &tokio::task::JoinHandle<Result<()>>,
) -> Result<()> {
    let deadline = Instant::now() + TARGET_LOOP_TIMEOUT;
    while !target.is_finished() {
        if Instant::now() >= deadline {
            anyhow::bail!("{signal_fn} never stopped the target within {TARGET_LOOP_TIMEOUT:?}");
        }
        // Parallel workers inherit the leader's `application_name`, so pin to the client backend
        // to signal the leader; its abort is what sends SIGTERM to the workers.
        let pid: Option<i32> = sqlx::query_scalar(
            "SELECT pid FROM pg_stat_activity \
             WHERE application_name = $1 AND backend_type = 'client backend' \
             AND state = 'active' AND query LIKE '%FROM bs_cancel%'",
        )
        .bind(target_app)
        .fetch_optional(&mut *signaller)
        .await?;

        if let Some(pid) = pid {
            sleep(SIGNAL_DELAY).await;
            // The backend may have exited during the wait; signalling it only while it's still
            // listed avoids a "not a PostgreSQL backend process" warning.
            sqlx::query(AssertSqlSafe(format!(
                "SELECT {signal_fn}(pid) FROM pg_stat_activity WHERE pid = $1"
            )))
            .bind(pid)
            .execute(&mut *signaller)
            .await?;
        }
        sleep(Duration::from_millis(5)).await;
    }
    Ok(())
}

/// Run the scan in a loop on `target` until the signal cuts it off, and check it ended with
/// `expected` (the SQLSTATE the signal should produce). `signal_until_stopped` bounds the loop.
async fn run_until_signalled(
    mut target: PgConnection,
    app_name: String,
    query: &'static str,
    expected: &'static str,
) -> Result<()> {
    target
        .execute(AssertSqlSafe(format!(
            "SET application_name = '{app_name}';"
        )))
        .await?;
    target.execute(PARALLEL_GUCS).await?;

    let err = loop {
        if let Err(e) = target.execute(query).await {
            break e;
        }
    };

    let ok = match &err {
        sqlx::Error::Database(db) => db.code().as_deref() == Some(expected),
        // A terminate may surface as the connection closing under us instead of the FATAL.
        sqlx::Error::Io(_) => expected == ADMIN_SHUTDOWN,
        _ => false,
    };
    anyhow::ensure!(ok, "expected SQLSTATE {expected}, target ended with: {err}");
    Ok(())
}

async fn assert_cluster_alive(witness: &mut PgConnection) -> Result<()> {
    // A worker crash makes the postmaster reset the cluster, which would have killed this
    // long-lived connection. If it still answers, no reset happened.
    let one: i32 = sqlx::query_scalar("SELECT 1")
        .fetch_one(&mut *witness)
        .await?;
    assert_eq!(one, 1, "witness connection lost: a backend crashed");
    Ok(())
}

#[rstest]
#[tokio::test]
async fn basescan_signal_does_not_crash_workers(database: Db) -> Result<()> {
    let mut setup = database.connection().await;
    setup.execute(SETUP_SQL).await?;
    setup
        .execute("SET paradedb.global_mutable_segment_rows = 0")
        .await?;
    for _ in 0..SEGMENTS {
        setup.execute(INSERT_SEGMENT).await?;
    }
    setup
        .execute("RESET paradedb.global_mutable_segment_rows")
        .await?;
    setup.execute("ANALYZE bs_cancel").await?;

    let mut witness = database.connection().await;
    assert_cluster_alive(&mut witness).await?;
    for (_, query, tantivy_query_contains) in QUERIES {
        assert_query_plans_as_parallel_columnar(&mut setup, query, tantivy_query_contains).await?;
    }

    for (name, query, _) in QUERIES {
        for (signal, signal_fn, expected) in [
            ("cancel", "pg_cancel_backend", QUERY_CANCELED),
            ("terminate", "pg_terminate_backend", ADMIN_SHUTDOWN),
        ] {
            for attempt in 0..ATTEMPTS {
                let app_name = format!("bs_{name}_{signal}_{attempt}");
                let target = database.connection().await;
                let loop_handle = tokio::spawn(run_until_signalled(
                    target,
                    app_name.clone(),
                    query,
                    expected,
                ));

                let mut signaller = database.connection().await;
                signal_until_stopped(&mut signaller, &app_name, signal_fn, &loop_handle).await?;

                let outcome = loop_handle.await?;
                // Check the witness first so a crash reports as a crash, not as the target's error.
                assert_cluster_alive(&mut witness).await?;
                outcome?;
            }
        }
    }

    Ok(())
}

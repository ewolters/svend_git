# DEPRECATED — pending sunset (2026-05-28)

This package (`syn/sched/`, codename **Tempora**) is the original in-repo
cognitive scheduler. It is **deprecated and pending sunset**.

## Superseded by

`~/tempora-service/` — the standalone Tempora service, run via
`manage.py run_scheduler`. That is the **active production drainer** for
`email_queue` (SignalFire email) and `task_queue`, running as the
`tempora.service` systemd unit.

## Why this is safe to sunset

- Nothing in the active trees (`~/svend`, `~/tempora-service`,
  `~/kjerne-services`) imports `syn.sched`. The only importers are other apps
  inside the sunset `~/kjerne` tree.
- The `tempora_server` management command has **no live systemd unit**. The one
  stale reference, `/etc/systemd/system/svend-tempora.service`, is **disabled
  and inert** — its `tempora_server` command no longer exists (only
  `run_scheduler` does), so it cannot start a second drainer.

## Do not

- Do not extend or fix this scheduler. Make scheduling/worker changes in
  `~/tempora-service/` instead.
- Do not install `ops/tempora.service` or run `ops/start_tempora.sh`.

## Scope marked deprecated

- `syn/sched/` (scheduler, executor, worker_pool, temporal/, backpressure/,
  dashboard/, models, core, etc.)
- `syn/sched/management/commands/tempora_server.py`
- `ops/start_tempora.sh`
- `ops/tempora.service`

Retained only as reference until the `~/kjerne` tree is removed.

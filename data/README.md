# Curated local demonstration results

`tasks.jsonl` is the starter benchmark. The `examples/` folders under
`processed/`, `traces/`, and `evaluations/` contain small representative artifacts:

- Synthetic KO/PEP acceptance: typed report and full local trace.
- Synthetic NVDA/AMD rejection: high correlation alone does not imply cointegration.
- Aggregate summary of the 20-task synthetic benchmark.

All example prices are artificial seeded fixtures. These outputs are not claims
about actual stocks, future returns, or benchmark accuracy. Reports preserve the
original configuration, source, trace ID and timestamps. They do not contain keys.

Uncurated research runs, training/failure exports, charts, real Yahoo downloads
and metadata caches remain local and are ignored by Git. Raw directories are
created automatically by the data provider. To publish another small example,
review its content and copy it deliberately into one of the `examples/` folders.

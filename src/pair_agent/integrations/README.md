# Future adapters (intentionally not implemented)

* **LangfuseTraceSink**: implement `TraceSink`; map trace ID to a trace, tool/model steps
  to observations, deterministic `EvaluationResult` fields to scores. Preserve local
  JSONL as an optional secondary sink. TODO: SDK version, authentication, retry and
  retention policy once access exists. No dummy network integration is installed.
* **DistilDatasetExporter**: wrap `evaluation.dataset.select_traces` and map its generic
  conversations/metadata to the documented Distil Labs schema once available.
  Success, failure and score-range exports work locally today. Review and label
  exports before training; failures are diagnostic examples, not successful targets.
* **Autoresearch**: TODO external runner consuming versioned datasets and deterministic
  validation scores. Keep final test results outside optimization objectives.
* **Runpod/OpenAICompatibleModelClient**: the implemented `ModelClient` adapter calls
  `/chat/completions`. Supply a compatible server's `/v1` URL through `LLM_BASE_URL`.
  Tool calling must be supported by the serving stack/model. Hosted models requiring
  the Responses API need a separate `ModelClient` adapter; model names are never guessed.

`TraceSink` and `ModelClient` are the boundaries. Do not put provider SDKs in quant tools.

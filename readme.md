MVP Stillshot Tech Stack

| Purpose                           | Library / Tool                        | Notes                                                  |
|-----------------------------------|----------------------------------------|--------------------------------------------------------|
| Image captioning                  | `transformers`, `PIL`, `torch`         | Uses HuggingFace BLIP                                  |
| Metadata fetching / I/O           | `requests`, `json`, `os`, `re`, `time` | JSON streaming, URL normalization, retries             |
| Text embedding                    | `sentence-transformers`                | Uses `all-MiniLM-L6-v2`                                |
| Agentic workflow orchestration    | `langgraph`                            | Composable state-machine graph for structured flows    |
| Local LLM-based field extraction  | `requests` to `localhost:11434`        | Assumes Ollama running LLaMA3 model                    |


TODO:
Build full flow for real-time updates of new registered IP
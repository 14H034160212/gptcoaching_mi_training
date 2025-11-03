# Rasa Integration (Minimal)

## Setup
```bash
pip install rasa==3.6.20 rasa-sdk==3.6.2
export FASTAPI_URL=http://localhost:8000/chat   # FastAPI demo endpoint
rasa train
rasa run actions &
rasa shell
```
When the story triggers `action_llm_coach`, Rasa will call the FastAPI `/chat` endpoint (served by `scripts/app_demo.py`) to get an MI-style response.

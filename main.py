from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from qa_agent.agent import QATestCaseArchitect


app = FastAPI(title="Test Case Architect Agent v1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
_agent: QATestCaseArchitect | None = None


def get_agent() -> QATestCaseArchitect:
    global _agent
    if _agent is None:
        _agent = QATestCaseArchitect(knowledge_dir="knowledge")
    return _agent


class GenerateRequest(BaseModel):
    story: str = Field(min_length=1)
    acceptance_criteria: str = ""
    business_rules: str = ""
    historical_bugs: str = ""
    top_k: int = Field(default=5, ge=1, le=20)


class GenerateResponse(BaseModel):
    raw: str
    retrieved_documents: list[dict]


@app.post("/reindex")
def reindex() -> dict:
    try:
        return get_agent().reindex()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    query = "\n\n".join(
        [
            "Historia de usuario:\n" + req.story.strip(),
            "Criterios de aceptación:\n" + (req.acceptance_criteria or "").strip(),
            "Reglas de negocio:\n" + (req.business_rules or "").strip(),
            "Bugs históricos:\n" + (req.historical_bugs or "").strip(),
        ]
    ).strip()

    try:
        out = get_agent().generate(query=query, top_k=req.top_k)
        return GenerateResponse(**out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

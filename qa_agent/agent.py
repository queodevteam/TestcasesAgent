import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators.openai import OpenAIGenerator
from haystack.utils.auth import Secret


QA_PROMPT_TEMPLATE = """Eres un QA Lead experto en diseño avanzado de casos de prueba.

Debes aplicar explícitamente estas técnicas:
- Partición de equivalencia
- Valores límite
- Tabla de decisión
- Transición de estados
- Pruebas negativas
- Análisis basado en riesgo
- Regresión basada en historial de bugs

Contexto del sistema:
{% for doc in documents %}
- {{ doc.content }}
{% endfor %}

Historia de usuario:
{{ query }}

Genera:

1️⃣ Identificación de condiciones y variables
2️⃣ Tabla de decisiones si aplica
3️⃣ Casos de prueba organizados en:
   - Positivos
   - Negativos
   - Valores límite
   - Transición de estados
   - Riesgo alto
   - Regresión
4️⃣ Casos de prueba en formato estructurado:
   ID:
   Título:
   Precondiciones:
   Pasos:
   Resultado esperado:
   Prioridad:
   Técnica aplicada:

Responde SOLO en JSON válido con esta forma:
{
  "conditions": [
    {
      "name": "",
      "description": "",
      "equivalence_partitions": [""],
      "boundary_values": [""],
      "notes": ""
    }
  ],
  "decision_table": {
    "applicable": true,
    "headers": [""],
    "rows": [[""]]
  },
  "test_cases": [
    {
      "id": "TC-001",
      "title": "",
      "category": "positive|negative|boundary|state_transition|high_risk|regression|security|performance",
      "preconditions": [""],
      "steps": [""],
      "expected_result": "",
      "priority": "P0|P1|P2|P3",
      "technique_applied": [""],
      "risk_rationale": "",
      "regression_links": [""],
      "integration_impact": [""],
      "data": {
        "inputs": {},
        "expected_outputs": {}
      }
    }
  ]
}
"""


def load_knowledge_documents(knowledge_dir: str) -> list[Document]:
    base = Path(knowledge_dir)
    docs: list[Document] = []
    if not base.exists() or not base.is_dir():
        return docs

    for p in sorted(base.glob("*.txt")):
        try:
            content = p.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            content = p.read_text(encoding="latin-1").strip()

        if not content:
            continue

        docs.append(Document(content=content, meta={"source": str(p.name)}))

    return docs


class QATestCaseArchitect:
    def __init__(self, knowledge_dir: str = "knowledge"):
        load_dotenv()
        self.knowledge_dir = knowledge_dir
        self.document_store = InMemoryDocumentStore()
        self.retriever = InMemoryBM25Retriever(document_store=self.document_store)
        self.prompt_builder = PromptBuilder(
            template=QA_PROMPT_TEMPLATE,
            required_variables=["documents", "query"],
        )

        llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        if llm_provider == "openai":
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY no está configurada. Crea un .env basado en .env.example")

            self.generator = OpenAIGenerator(model=model, api_key=Secret.from_env_var("OPENAI_API_KEY"))
        elif llm_provider in {"ollama", "ollama_cloud", "ollama-cloud"}:
            model = os.getenv("OLLAMA_MODEL", "llama3.2")
            api_key = os.getenv("OLLAMA_API_KEY")
            if not api_key:
                raise RuntimeError("OLLAMA_API_KEY no está configurada. Crea una API key en https://ollama.com/settings/keys")

            api_base_url = os.getenv("OLLAMA_API_BASE_URL", "https://ollama.com/v1")
            self.generator = OpenAIGenerator(
                model=model,
                api_key=Secret.from_env_var("OLLAMA_API_KEY"),
                api_base_url=api_base_url,
                timeout=120,
            )
        elif llm_provider in {"deepseek"}:
            model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError("DEEPSEEK_API_KEY no está configurada. Crea una API key en https://platform.deepseek.com/")

            api_base_url = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1")
            self.generator = OpenAIGenerator(
                model=model,
                api_key=Secret.from_env_var("DEEPSEEK_API_KEY"),
                api_base_url=api_base_url,
                timeout=120,
            )
        else:
            raise RuntimeError(f"LLM_PROVIDER inválido: {llm_provider}. Usa 'openai', 'ollama_cloud' o 'deepseek'.")

        self.pipeline = Pipeline()
        self.pipeline.add_component("retriever", self.retriever)
        self.pipeline.add_component("prompt_builder", self.prompt_builder)
        self.pipeline.add_component("generator", self.generator)

        self.pipeline.connect("retriever.documents", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder", "generator")

        self.reindex()

    def reindex(self) -> dict[str, Any]:
        docs = load_knowledge_documents(self.knowledge_dir)
        existing = self.document_store.filter_documents()
        existing_ids = [d.id for d in existing]
        if existing_ids:
            self.document_store.delete_documents(existing_ids)
        if docs:
            self.document_store.write_documents(docs)
        return {"indexed": len(docs)}

    def generate(self, query: str, top_k: int = 5) -> dict[str, Any]:
        result = self.pipeline.run(
            {
                "retriever": {"query": query, "top_k": top_k},
                "prompt_builder": {"query": query},
            }
        )

        replies = result.get("generator", {}).get("replies", [])
        text = replies[0] if replies else ""
        return {"raw": text, "retrieved_documents": [d.meta for d in result.get("retriever", {}).get("documents", [])]}

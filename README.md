# Test Case Architect Agent

Agente de IA para generación automatizada de casos de prueba QA. A partir de una historia de usuario, criterios de aceptación y reglas de negocio, genera casos de prueba estructurados en JSON aplicando técnicas avanzadas de QA.

## Características

- Generación de casos de prueba con técnicas: partición de equivalencia, valores límite, tabla de decisión, transición de estados, pruebas negativas, análisis de riesgo y regresión
- Pipeline RAG con Haystack AI: recupera contexto relevante desde la carpeta `knowledge/` usando BM25
- Soporte para tres proveedores de LLM: **OpenAI**, **Ollama Cloud** y **DeepSeek**
- API REST con FastAPI
- Interfaz web incluida (Tailwind CSS + JS vanilla) con filtros, tabla de decisiones y exportación a JSON/CSV

## Requisitos

- Python 3.10 o superior
- Cuenta y API key en al menos uno de los proveedores LLM soportados

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd TestcasesAgent

# 2. Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
```

Edita el archivo `.env` con tus credenciales (ver sección [Configuración](#configuración)).

## Configuración

El archivo `.env` controla qué proveedor LLM se usa mediante la variable `LLM_PROVIDER`.

### OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

### Ollama Cloud

```env
LLM_PROVIDER=ollama_cloud
OLLAMA_API_KEY=TU_KEY_AQUI
OLLAMA_MODEL=llama3.2
OLLAMA_API_BASE_URL=https://ollama.com/v1
```

Obtén tu API key en [https://ollama.com/settings/keys](https://ollama.com/settings/keys).

### DeepSeek

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_API_BASE_URL=https://api.deepseek.com/v1
```

Obtén tu API key en [https://platform.deepseek.com/](https://platform.deepseek.com/).

## Uso

### Iniciar el servidor

```bash
uvicorn main:app --reload
```

El servidor queda disponible en `http://127.0.0.1:8000`.

### Interfaz web

Abre en el navegador:

```
http://127.0.0.1:8000/static/qa.html
```

### API REST

#### `POST /generate`

Genera casos de prueba a partir de una historia de usuario.

**Body (JSON):**

```json
{
  "story": "Como usuario quiero iniciar sesión con email y contraseña",
  "acceptance_criteria": "El sistema debe validar el formato del email",
  "business_rules": "Máximo 3 intentos fallidos antes de bloquear la cuenta",
  "historical_bugs": "Bug #42: sesión no expiraba correctamente",
  "top_k": 5
}
```

| Campo                | Tipo   | Requerido | Descripción                                      |
|----------------------|--------|-----------|--------------------------------------------------|
| `story`              | string | Sí        | Historia de usuario                              |
| `acceptance_criteria`| string | No        | Criterios de aceptación                          |
| `business_rules`     | string | No        | Reglas de negocio adicionales                    |
| `historical_bugs`    | string | No        | Bugs históricos relevantes                       |
| `top_k`              | int    | No        | Documentos a recuperar del knowledge (1–20, default: 5) |

**Respuesta:**

```json
{
  "raw": "{ ...JSON con casos de prueba generados... }",
  "retrieved_documents": [{ "source": "reglas_negocio.txt" }]
}
```

#### `POST /reindex`

Recarga los documentos de la carpeta `knowledge/` en el almacén de documentos.

```bash
curl -X POST http://127.0.0.1:8000/reindex
```

**Respuesta:**

```json
{ "indexed": 3 }
```

## Base de conocimiento

La carpeta `knowledge/` contiene archivos `.txt` que el agente usa como contexto al generar casos de prueba:

| Archivo                  | Descripción                              |
|--------------------------|------------------------------------------|
| `reglas_negocio.txt`     | Reglas de negocio del sistema            |
| `criterios_acceso.txt`   | Criterios de acceso y permisos           |
| `bugs_históricos.txt`    | Historial de bugs para pruebas de regresión |

Puedes agregar o modificar archivos `.txt` en esta carpeta y luego llamar a `POST /reindex` para actualizar el índice sin reiniciar el servidor.

## Estructura del proyecto

```
TestcasesAgent/
├── main.py                  # Servidor FastAPI
├── requirements.txt         # Dependencias Python
├── .env.example             # Plantilla de configuración
├── qa_agent/
│   └── agent.py             # Agente QA con pipeline Haystack
├── static/
│   └── qa.html              # Interfaz web
└── knowledge/
    ├── reglas_negocio.txt
    ├── criterios_acceso.txt
    └── bugs_históricos.txt
```

## Dependencias principales

| Paquete          | Versión  | Uso                              |
|------------------|----------|----------------------------------|
| `haystack-ai`    | 2.18.0   | Pipeline RAG y gestión de LLMs   |
| `fastapi`        | 0.115.8  | Framework API REST               |
| `uvicorn`        | 0.34.0   | Servidor ASGI                    |
| `python-dotenv`  | 1.0.1    | Carga de variables de entorno    |
| `pydantic`       | 2.10.6   | Validación de datos              |

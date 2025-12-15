# RAG Builder API

This module provides REST API endpoints for managing RAG (Retrieval-Augmented Generation) elements, states, and transitions.

## Architecture

The RAG API endpoints are part of `qontinui-api` but interact with the **main backend's database** (`qontinui_db`), not the qontinui-api's own database. This is because RAG data is stored in the `projects.rag_config` JSON column in the main backend.

### Database Connection

- **qontinui-api database**: `postgresql://qontinui:qontinui_dev_password@localhost:5432/qontinui` (for snapshots, captures, etc.)
- **Main backend database**: `postgresql://qontinui_user:qontinui_dev_password@localhost:5432/qontinui_db` (for projects, RAG config, users, etc.)

The RAG router creates a separate SQLAlchemy engine and session factory for the main backend database.

## Endpoints

### Elements

- `GET /api/rag/projects/{project_id}/elements` - List all elements
- `POST /api/rag/projects/{project_id}/elements` - Create element
- `GET /api/rag/elements/{element_id}?project_id={project_id}` - Get element
- `PUT /api/rag/elements/{element_id}?project_id={project_id}` - Update element
- `DELETE /api/rag/elements/{element_id}?project_id={project_id}` - Delete element
- `POST /api/rag/elements/{element_id}/generate-description?project_id={project_id}` - Generate AI description

### States

- `GET /api/rag/projects/{project_id}/states` - List all states
- `POST /api/rag/projects/{project_id}/states` - Create state
- `GET /api/rag/states/{state_id}?project_id={project_id}` - Get state
- `PUT /api/rag/states/{state_id}?project_id={project_id}` - Update state
- `DELETE /api/rag/states/{state_id}?project_id={project_id}` - Delete state

### Transitions

- `GET /api/rag/projects/{project_id}/transitions` - List all transitions
- `POST /api/rag/projects/{project_id}/transitions` - Create transition
- `GET /api/rag/transitions/{transition_id}?project_id={project_id}` - Get transition
- `PUT /api/rag/transitions/{transition_id}?project_id={project_id}` - Update transition
- `DELETE /api/rag/transitions/{transition_id}?project_id={project_id}` - Delete transition

### Search

- `POST /api/rag/projects/{project_id}/search` - Search elements

### Export/Import

- `GET /api/rag/projects/{project_id}/export` - Export project
- `POST /api/rag/projects/{project_id}/import` - Import project

## Data Storage

RAG data is stored in the `projects.rag_config` JSON column with this structure:

```json
{
  "elements": {
    "element-id-1": { /* RAGElement */ },
    "element-id-2": { /* RAGElement */ }
  },
  "states": {
    "state-id-1": { /* RAGState */ },
    "state-id-2": { /* RAGState */ }
  },
  "transitions": {
    "transition-id-1": { /* RAGTransition */ },
    "transition-id-2": { /* RAGTransition */ }
  }
}
```

## Usage Example

```typescript
// Frontend service usage
import { RAGBuilderService } from '@/services/rag-builder-service';

const ragService = new RAGBuilderService(httpClient);

// Create an element
const element = await ragService.createElement(projectId, {
  source_app: 'MyApp',
  extraction_method: 'manual',
  element_type: 'button',
  ocr_text: 'Submit',
  // ... other fields
});

// Get all elements
const elements = await ragService.getElements(projectId);

// Search elements
const results = await ragService.search(projectId, {
  query: 'submit button',
  element_types: ['button'],
  limit: 10,
});
```

## Development Notes

- The API uses synchronous SQLAlchemy (not async) to match the pattern from other qontinui-api routes
- UUIDs are generated using Python's `uuid.uuid4()`
- Timestamps are in ISO 8601 format
- The `generate-description` endpoint currently uses rule-based description generation but can be enhanced with LLM integration

## Testing

```bash
# Start qontinui-api
cd /mnt/c/Users/Joshua/Documents/qontinui_parent_directory/qontinui-api
poetry run uvicorn main:app --reload --port 8001 --host 0.0.0.0

# Test endpoints
curl http://localhost:8001/api/rag/projects/{project-id}/elements
```

# Intelligent Cloud Resource Allocation and Cost Optimization

## 1. Project Overview

High-Performance Computing (HPC) workloads require significant computational resources, including processing nodes, CPU cores, memory, and execution time. Inefficient resource allocation can lead to unnecessary resource consumption, longer execution times, and increased operational costs.

**ML-CLOUD** is an intelligent HPC resource allocation and cost optimization system that combines Machine Learning (ML), Explainable AI (XAI), Retrieval-Augmented Generation (RAG), and Large Language Models (LLMs) to improve resource planning and support informed allocation decisions.

The system analyzes historical HPC job data, predicts resource usage and runtime, evaluates alternative allocations, estimates computational costs, and provides explainable recommendations through an interactive web application.

The project is divided into **four independently developable modules**, enabling parallel implementation and integration through predefined interfaces.

## 2. Project Objectives

- Analyze and validate large-scale historical HPC job datasets.
- Predict HPC resource usage and job execution time using machine learning.
- Explain model predictions using Explainable AI techniques.
- Optimize resource allocation under configurable HPC constraints.
- Estimate resource consumption, node-hours, and computational costs.
- Retrieve relevant HPC documentation and allocation policies using RAG.
- Provide natural-language explanations and answers through an LLM-powered assistant.
- Develop a web dashboard for predictions, optimization comparisons, explanations, and job history.
- Support transparent, data-driven resource allocation decisions.

## 3. Core Features

### 3.1 HPC Dataset Engineering
- Parquet dataset inspection and processing.
- Missing-value and duplicate analysis.
- Resource-usage validation.
- Feature engineering and data preparation.
- Chronological training, validation, and test splits.
- Prevention of target leakage.

### 3.2 Machine Learning Resource Prediction
- Predict node usage and job runtime.
- Train and compare multiple regression algorithms.
- Evaluate models using MAE, RMSE, and R².
- Save trained models for inference.
- Expose prediction functions through a consistent interface.

### 3.3 Explainable AI
- Explain individual resource and runtime predictions.
- Identify influential input features using SHAP.
- Generate global feature-importance reports.
- Display feature contributions alongside predictions.
- Communicate model limitations and uncertainty where supported.

### 3.4 Intelligent Resource Optimization
- Evaluate alternative resource allocations.
- Apply configurable resource and scheduling constraints.
- Compare predicted runtimes across feasible candidates.
- Calculate node-hours and estimated computational costs.
- Compare proposed allocations against a baseline.
- Report estimated savings when supported by valid comparisons.

### 3.5 RAG-Based HPC Knowledge Assistant
- Index HPC documentation and scheduling guidelines.
- Retrieve relevant information from a vector database.
- Ground answers in retrieved documents.
- Provide source references for supporting evidence.
- Answer questions about resource allocation, optimization results, and HPC policies.

### 3.6 LLM Integration
- Explain optimization recommendations in natural language.
- Summarize resource and cost comparisons.
- Answer user questions using retrieved documentation and verified system results.
- Generate structured explanations for dashboard presentation.
- Clearly communicate missing information and unsupported conclusions.

### 3.7 Interactive Web Application
- Job parameter input forms.
- Resource and runtime prediction results.
- Resource allocation recommendations.
- Cost comparison charts.
- Explainable AI visualizations.
- RAG-powered conversational interface.
- Prediction and optimization history.

---

## 4. System Architecture

The project consists of four modules.

```text
                 HISTORICAL HPC JOB DATA
                           |
                           v
            +-----------------------------+
            | MODULE 1                    |
            | Dataset Engineering         |
            | Cleaning, Validation,       |
            | Feature Engineering         |
            +-----------------------------+
                           |
                           v
            +-----------------------------+
            | MODULE 2                    |
            | Resource Prediction + XAI   |
            | ML Models, Evaluation, SHAP |
            +-----------------------------+
                           |
                           v
            +-----------------------------+
            | MODULE 3                    |
            | Intelligent Optimization    |
            | Cost Estimation             |
            | RAG Knowledge Retrieval     |
            | LLM Assistant               |
            +-----------------------------+
                           |
                           v
            +-----------------------------+
            | MODULE 4                    |
            | FastAPI Backend             |
            | React Web Application       |
            | Integration and Testing     |
            +-----------------------------+
                           |
                           v
              INTELLIGENT HPC DASHBOARD
```

**Development principle:** The modules can be developed in parallel using fixed interfaces, sample datasets, mock predictions, and predefined API schemas. Integration occurs when the actual implementations are ready.

The diagram represents the conceptual data flow. During implementation, the application may invoke prediction, optimization, explanation, and retrieval services independently according to the user's request.

---

## 5. Four Development Modules

### Module 1: Dataset Engineering and Validation

**Objective:** Prepare reliable, ML-ready HPC job data.

**Responsibilities:**
1. Inspect source Parquet files.
2. Analyze missing values, duplicate records, and invalid resource values.
3. Validate resource-request and execution fields.
4. Engineer relevant features.
5. Prepare chronological training, validation, and test datasets.
6. Document feature definitions, data types, and target variables.
7. Generate data-quality reports.

**Suggested technologies:**
- Python
- Pandas
- PyArrow
- NumPy
- Parquet

**Expected deliverables:**
- Validated datasets.
- Data-quality reports.
- Feature dictionary.
- Data preparation scripts.
- Sample input and output records.
- Module documentation.

**Independent development:** This module can work with the existing EAGLE HPC dataset and produce documented sample datasets for the other modules.

### Module 2: Resource Prediction and Explainable AI

**Objective:** Predict HPC resource usage and explain the predictions.

**Responsibilities:**
1. Train regression models for node usage and runtime.
2. Compare baseline and advanced algorithms.
3. Evaluate model performance.
4. Save trained models.
5. Implement reusable prediction functions.
6. Apply SHAP to explain individual predictions.
7. Generate global feature-importance reports.
8. Document model limitations and evaluation results.

**Suggested technologies:**
- Python
- scikit-learn
- XGBoost
- SHAP
- Joblib

**Expected deliverables:**
- Trained ML models.
- Model evaluation reports.
- Prediction functions.
- SHAP explanation outputs.
- Feature-importance visualizations.
- Module documentation.

**Independent development:** This module can initially use a sample dataset and synthetic job records. It does not need to wait for the final dataset-preparation pipeline.

**Important:** Features must be available at prediction time. Post-execution fields such as `nodes_used`, `processors_used`, and `wallclock_used_seconds` must not be used as input features for predictions made before a job starts.

### Module 3: Intelligent Resource Optimization, RAG, and LLM

**Objective:** Recommend feasible resource allocations and explain them using verified results and relevant HPC knowledge.

This module combines three cooperating components.

#### A. Resource Optimization and Cost Estimation

Responsibilities:
- Generate candidate resource allocations.
- Apply configurable allocation constraints.
- Estimate runtime for candidate allocations.
- Compare feasible candidates.
- Calculate node-hours and estimated costs.
- Compare recommendations with a baseline.
- Report estimated resource and cost differences.

Basic calculations:

```text
Node-hours = Allocated nodes × Runtime in seconds / 3600

Estimated cost = Node-hours × Cost per node-hour
```

These calculations require an appropriate allocation definition and pricing configuration. If actual billing rates are unavailable, the system must label monetary estimates as hypothetical.

#### B. RAG Knowledge Base

Responsibilities:
- Collect relevant HPC documentation and scheduling policies.
- Split documents into searchable chunks.
- Generate embeddings.
- Store embeddings in a vector database.
- Retrieve relevant passages for user questions.
- Return source metadata with retrieved content.

**Suggested technologies:**
- Sentence Transformers
- ChromaDB or FAISS
- LangChain or LlamaIndex

#### C. LLM-Powered Assistant

Responsibilities:
- Explain resource allocation recommendations.
- Summarize cost and runtime comparisons.
- Answer HPC-related questions.
- Use retrieved documentation to support policy-related answers.
- Explain verified prediction and optimization outputs.
- Identify missing information instead of inventing answers.

**Suggested technologies:**
- Ollama with a suitable local LLM, or a hosted LLM API.
- Python
- Pydantic for structured outputs.

**Expected deliverables:**
- Resource optimization engine.
- Allocation constraint validator.
- Cost estimator.
- RAG ingestion and retrieval pipeline.
- Vector database.
- LLM assistant.
- Grounded explanations and source references.
- Module documentation.

**Independent development:** The optimization component can use a mock predictor, the RAG component can use sample documentation, and the LLM component can use mock optimization results.

**Important:** Deterministic code must validate allocations and calculate costs. The LLM explains results but must not independently override constraints or invent predictions and financial savings.

### Module 4: Backend, Web Application, and Integration

**Objective:** Deliver a unified interface for the intelligent HPC optimization system.

**Responsibilities:**
1. Build REST APIs using FastAPI.
2. Develop the dashboard using React.
3. Create job input forms.
4. Display resource and runtime predictions.
5. Present allocation recommendations and cost comparisons.
6. Display SHAP explanations.
7. Integrate the RAG-powered assistant.
8. Show retrieved source references.
9. Maintain prediction and optimization history.
10. Test integrated workflows and error handling.

**Suggested technologies:**
- FastAPI
- Pydantic
- React
- Vite
- Charting libraries
- SQLite or another suitable database

**Expected deliverables:**
- Backend API.
- Frontend dashboard.
- Prediction and optimization visualizations.
- Conversational assistant interface.
- History functionality.
- API documentation.
- Integration tests.
- Application setup instructions.

**Independent development:** The frontend and backend can use mock JSON responses until the other modules expose their actual implementations.

---

## 6. Technology Stack

| Component | Recommended technology |
|---|---|
| Programming language | Python |
| Dataset processing | Pandas, NumPy, PyArrow |
| Dataset format | Parquet |
| Machine learning | scikit-learn, XGBoost |
| Explainable AI | SHAP |
| Embeddings | Sentence Transformers |
| Vector database | ChromaDB or FAISS |
| RAG orchestration | LangChain or LlamaIndex |
| LLM | Ollama or hosted LLM API |
| Backend | FastAPI |
| API schemas | Pydantic |
| Frontend | React, Vite |
| Database | SQLite |
| Testing | pytest |
| Model persistence | Joblib |
| Version control | Git and GitHub |

Choose one vector database and one RAG orchestration approach initially to keep development manageable.

---

## 7. Independent Development and Integration Strategy

All four modules should be developed simultaneously.

| Module | Initial development strategy | Integration requirement |
|---|---|---|
| Module 1 | Work with the current HPC dataset | Publish documented schemas and features |
| Module 2 | Train models on available ML-ready data | Expose prediction and explanation functions |
| Module 3 | Use synthetic job requests, mock predictions, and sample documents | Replace mock predictions with the actual prediction interface |
| Module 4 | Use mock API responses | Connect the actual backend services |

### Shared Interface Contracts

The modules must agree on the following contracts before implementation.

**Module 1: Dataset contract**
- Feature names and data types.
- Target definitions.
- Missing-value handling.
- Train, validation, and test split conventions.
- Sample input records.

**Module 2: Prediction contract**

```python
predict_nodes(job_input)
predict_runtime(job_input)
explain_prediction(job_input)
```

Each function must have documented inputs, outputs, units, and error behavior.

**Module 3: Optimization and AI contract**

```python
optimize_resources(job_input)
retrieve_context(question)
generate_explanation(context, results)
```

The optimization function should return structured candidate allocations, feasibility information, predicted runtimes, and cost estimates. The retrieval function should return relevant passages with source metadata. The LLM function should consume validated results and retrieved context.

**Module 4: Application contract**
- API request and response schemas.
- Endpoint names.
- Error response format.
- Mock responses for frontend development.
- Authentication and authorization requirements, if applicable.

These contracts allow independent implementation without requiring each team member to wait for another member's code.

---

## 8. Proposed API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check application health |
| GET | `/api/v1/models/metrics` | Retrieve model metrics |
| POST | `/api/v1/predict/nodes` | Predict node usage |
| POST | `/api/v1/predict/runtime` | Predict runtime |
| POST | `/api/v1/explain` | Explain model predictions |
| POST | `/api/v1/optimize` | Evaluate candidate allocations |
| POST | `/api/v1/knowledge/search` | Retrieve HPC documentation |
| POST | `/api/v1/assistant/chat` | Answer questions using verified results and RAG |
| GET | `/api/v1/history` | Retrieve prediction and optimization history |

The endpoint list is a proposed contract. Actual endpoint implementation can be distributed across the modules and integrated through FastAPI routers.

---

## 9. Suggested Repository Structure

```text
ML-CLOUD/
│
├── module1_data/
│   ├── scripts/
│   ├── tests/
│   └── README.md
│
├── module2_prediction/
│   ├── src/
│   ├── models/
│   ├── metrics/
│   ├── tests/
│   └── README.md
│
├── module3_intelligence/
│   ├── optimization/
│   ├── cost_estimation/
│   ├── rag/
│   │   ├── ingestion/
│   │   ├── retrieval/
│   │   └── documents/
│   ├── llm_assistant/
│   ├── tests/
│   └── README.md
│
├── module4_application/
│   ├── backend/
│   │   ├── api/
│   │   ├── services/
│   │   └── tests/
│   ├── frontend/
│   └── README.md
│
├── shared/
│   ├── schemas/
│   ├── sample_requests/
│   └── mock_responses/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── validation/
│
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

The structure is a recommended target architecture. Existing project files can be retained until the new structure is implemented and tested.

---

## 10. Development Roadmap

### Week 1: Foundations and Contracts
- Inspect the available dataset.
- Define shared data schemas.
- Create sample requests and mock responses.
- Establish module directories and dependencies.
- Begin model, optimization, RAG, and application prototypes.

### Week 2: Independent Implementation
- Complete the initial data-preparation pipeline.
- Train and evaluate prediction models.
- Implement SHAP explanations.
- Develop allocation and cost calculations.
- Build the document-ingestion and retrieval pipeline.
- Implement the initial LLM assistant.
- Develop dashboard pages against mock APIs.

### Week 3: Integration
- Connect the validated dataset pipeline to the prediction workflow.
- Integrate prediction models with optimization.
- Connect SHAP outputs to the dashboard.
- Integrate RAG retrieval and LLM responses.
- Replace mock API responses with real service outputs.
- Test error handling and schema compatibility.

### Week 4: Evaluation and Delivery
- Run end-to-end tests.
- Evaluate model accuracy and prediction limitations.
- Validate resource constraints and cost calculations.
- Evaluate RAG retrieval quality and source attribution.
- Test LLM responses for unsupported claims.
- Refine dashboard usability.
- Prepare technical documentation and project presentation.

The schedule assumes that the team can work in parallel and that the dataset and development environment are accessible.

---

## 11. Evaluation Strategy

### Machine Learning
- Mean Absolute Error (MAE).
- Root Mean Squared Error (RMSE).
- Coefficient of Determination (R²).
- Chronological holdout evaluation.
- Comparison with simple baseline models.

### Explainable AI
- Consistency between SHAP outputs and model behavior.
- Global feature-importance analysis.
- Correct mapping between input features and explanations.

### Resource Optimization
- Percentage of recommendations satisfying constraints.
- Predicted runtime relative to the specified deadline.
- Node-hours before and after optimization.
- Estimated cost difference against the baseline.
- Evaluation of infeasible requests and missing pricing information.

### RAG and LLM
- Relevance of retrieved documents.
- Accuracy of source attribution.
- Faithfulness of answers to retrieved evidence.
- Correct handling of unsupported questions.
- Consistency with verified model and optimization results.

### Application
- API correctness.
- Input validation.
- Error handling.
- Integration tests.
- Usability of the dashboard.

---

## 12. Data Integrity and Responsible Use

- Avoid using information unavailable at prediction time.
- Keep training and evaluation datasets chronologically separated.
- Document model assumptions and limitations.
- Distinguish historical observations from model predictions.
- Label hypothetical pricing and estimated savings clearly.
- Do not claim actual financial savings without verified pricing and a valid baseline comparison.
- Use source-backed RAG answers for policy-related questions.
- Ensure the LLM cannot bypass resource-allocation constraints.
- Validate all generated recommendations before displaying them as feasible.
- Protect sensitive information and avoid exposing raw job data unnecessarily.

---

## 13. Expected Outcome

The completed project will provide an integrated platform that can:

1. Process large-scale HPC job datasets.
2. Predict resource usage and runtime.
3. Explain predictions using Explainable AI.
4. Compare feasible resource allocation alternatives.
5. Estimate computational costs.
6. Retrieve relevant HPC knowledge through RAG.
7. Explain recommendations using an LLM.
8. Present results through an interactive dashboard.

The result is an explainable, knowledge-grounded HPC resource optimization platform designed to support more informed resource allocation decisions.

## 14. Project Status

**Current development stage:** Modular architecture and parallel development planning.

| Module | Planned scope | Status |
|---|---|---|
| Module 1 | Dataset Engineering and Validation | In progress / existing data available |
| Module 2 | Resource Prediction and Explainable AI | Planned / existing data available |
| Module 3 | Optimization, RAG, and LLM | Planned |
| Module 4 | Backend, Web Dashboard, and Integration | Planned |

Update the status table as implementation progresses.

---

## 15. Conclusion

ML-CLOUD combines machine learning, explainability, constrained optimization, retrieval-augmented generation, and conversational AI to address HPC resource allocation and cost estimation.

Its four-module architecture supports parallel development, independent testing, and integration through well-defined interfaces. By separating deterministic prediction and optimization from LLM-generated explanations, the system aims to provide recommendations that are both useful and technically verifiable.

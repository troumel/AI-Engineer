# Project 1: ML Model Deployment API

**Difficulty:** ⭐ Easy
**Duration:** 3-5 days
**Tech Stack:** FastAPI, scikit-learn, pandas, pydantic, Docker

## Overview

A FastAPI service that serves a pre-trained scikit-learn model for house price predictions using the California housing dataset. This project teaches the fundamentals of deploying ML models as REST APIs.

## What You'll Learn

- FastAPI basics (routes, request/response models, validation)
- Pydantic models (similar to C# DTOs)
- Loading and serving ML models
- Model serialization (joblib)
- Error handling and logging
- Docker containerization

---

## Project Structure

```
Project_1/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app entry point (like Program.cs)
│   ├── config.py                  # Configuration (like appsettings.json)
│   ├── dependencies.py            # Dependency injection
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py             # Pydantic models (DTOs)
│   ├── services/
│   │   ├── __init__.py
│   │   └── prediction_service.py  # Business logic
│   └── routers/
│       ├── __init__.py
│       ├── predictions.py         # API routes (like Controllers)
│       └── health.py              # Health check endpoint
├── tests/
│   ├── __init__.py
│   └── test_predictions.py        # Unit tests
├── scripts/
│   └── train_model.py             # ML model training script
├── data/                          # Local data storage
├── models/                        # Saved ML models (.pkl, .joblib)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
├── .gitignore
└── README.md
```

---

## Setup Instructions

### 1. Create Virtual Environment

A virtual environment isolates your project dependencies (similar to NuGet packages being scoped to a project).

**Windows:**
```bash
# Navigate to project directory
cd "c:\Users\Theo\source\repos\AI Engineer\Project_1"

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# You should see (venv) in your terminal prompt
```

**macOS/Linux:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**Why virtual environments?**
- Isolates dependencies per project (no global package conflicts)
- Makes projects reproducible
- Similar to having separate NuGet package versions per .NET solution

### 2. Install Dependencies

```bash
# Make sure virtual environment is activated (you should see (venv) in prompt)
pip install -r requirements.txt

# Verify installation
pip list
```

**Key packages installed:**
- `fastapi` - Web framework for building APIs (like ASP.NET Core)
- `uvicorn` - ASGI server to run FastAPI (like Kestrel)
- `pydantic` - Data validation using Python type annotations (like C# DTOs with validation attributes)
- `scikit-learn` - ML library for training/inference
- `pandas` - Data manipulation (like LINQ but for tabular data)
- `numpy` - Numerical computing
- `joblib` - Model serialization
- `pytest` - Testing framework (like xUnit/NUnit)
- `httpx` - HTTP client for testing APIs
- `python-dotenv` - Load environment variables from .env file

### 3. Update pip (Optional)

```bash
python -m pip install --upgrade pip
```

### 4. Deactivate Virtual Environment (When Done)

```bash
deactivate
```

---

## Common Python/pip Commands

### Package Management

```bash
# Install a package
pip install package-name

# Install specific version
pip install package-name==1.2.3

# Uninstall a package
pip uninstall package-name

# List installed packages
pip list

# Show package details
pip show package-name

# Freeze current dependencies (save to file)
pip freeze > requirements.txt

# Install from requirements.txt
pip install -r requirements.txt

# Update a package to latest version
pip install --upgrade package-name
```

### Virtual Environment Commands

```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Deactivate
deactivate

# Check which Python is being used (should show venv path)
which python  # macOS/Linux
where python  # Windows
```

### Python Commands

```bash
# Check Python version
python --version

# Run a Python script
python script_name.py

# Run a module as a script
python -m module_name

# Interactive Python shell (REPL)
python

# Run FastAPI with uvicorn (development server)
uvicorn app.main:app --reload

# Run with specific host/port
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing Commands

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_predictions.py

# Run tests with coverage report
pytest --cov=app tests/
```

---

## Development Workflow

### Running the Application

```bash
# 1. Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# 2. Run the development server
uvicorn app.main:app --reload

# Server will start at http://localhost:8000
# Auto-generated docs available at http://localhost:8000/docs
```

**The `--reload` flag:**
- Auto-restarts server when code changes (like hot reload in .NET)
- Only use in development, NOT in production

### API Documentation

FastAPI automatically generates interactive API documentation:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

You can test your endpoints directly from the Swagger UI!

### Testing the API

```bash
# Option 1: Use Swagger UI in browser
# Go to http://localhost:8000/docs

# Option 2: Use curl
curl http://localhost:8000/health

# Option 3: Use pytest
pytest tests/test_predictions.py -v
```

---

## Environment Variables

Create a `.env` file (copy from `.env.example`):

```bash
# Copy example file
cp .env.example .env  # macOS/Linux
copy .env.example .env  # Windows

# Edit .env with your values
```

**Important:** Never commit `.env` to git (it's already in .gitignore)

---

## Git Workflow

```bash
# Initialize git (if not already done)
git init

# Check status
git status

# Stage files
git add .

# Commit
git commit -m "Initial project setup"

# View commit history
git log --oneline
```

---

## Docker Commands (For Later)

```bash
# Build Docker image
docker build -f docker/Dockerfile -t ml-api:latest .

# Run container
docker run -p 8000:8000 ml-api:latest

# Using docker-compose
docker-compose -f docker/docker-compose.yml up

# Stop containers
docker-compose -f docker/docker-compose.yml down
```

---

## Troubleshooting

### Virtual Environment Not Activating

**Windows:**
If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Wrong Python Version

Make sure you're using Python 3.10+:
```bash
python --version
```

If multiple Python versions installed, you might need:
```bash
python3 --version
python3 -m venv venv
```

### Package Installation Fails

Try upgrading pip first:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Port Already in Use

If port 8000 is busy:
```bash
uvicorn app.main:app --port 8001 --reload
```

---

## Learning Resources

### FastAPI
- Official Docs: https://fastapi.tiangolo.com/
- Tutorial: https://fastapi.tiangolo.com/tutorial/

### Pydantic
- Docs: https://docs.pydantic.dev/

### Scikit-learn
- Getting Started: https://scikit-learn.org/stable/getting_started.html
- User Guide: https://scikit-learn.org/stable/user_guide.html

### Python Testing
- pytest Docs: https://docs.pytest.org/

---

## Next Steps

1. ✅ Setup complete - Virtual environment and dependencies installed
2. 🔄 Train the ML model - Create and save a house price prediction model
3. 🔄 Build the API - Implement FastAPI endpoints
4. 🔄 Add tests - Write unit tests
5. 🔄 Dockerize - Create Docker container
6. 🔄 Deploy - Optional: Deploy to cloud platform

---

## Parallels to .NET (For Reference)

| Python/FastAPI | .NET/ASP.NET Core |
|----------------|-------------------|
| `venv` | Project-scoped NuGet packages |
| `requirements.txt` | `.csproj` dependencies |
| `pip install` | `dotnet add package` |
| `uvicorn` | Kestrel |
| `FastAPI()` | `WebApplication.CreateBuilder()` |
| Pydantic models | DTOs with validation attributes |
| `async def` | `async Task` |
| Type hints | C# static typing |
| `pytest` | xUnit/NUnit |
| `@router.get()` | `[HttpGet]` attribute |
| Dependency injection in FastAPI | ASP.NET Core DI |
| `.env` file | `appsettings.json` + User Secrets |

---

## Project Status

- [x] Project structure created
- [x] Virtual environment setup
- [x] Dependencies installed
- [x] .gitignore configured
- [ ] ML model training script
- [ ] API implementation
- [ ] Unit tests
- [ ] Docker setup
- [ ] Documentation complete

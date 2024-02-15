# Document Q&A

This is an API for parsing a text PDF document and answering the questions asked using an LLM. **This project is still in development.**

## Installation

This project uses poetry and a pyproject.toml file is included for easy installation of requirements

if you are using poetry

```bash
poetry install
```

## Usage

To run a developement server using uvicorn

```python
uvicorn main:app --reload
```

For documentations of the endpoints, hover over to

```
http://localhost:8000/docs
```

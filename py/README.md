# How to execute

## 1. first, install `uv`

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Sync Python library dependencies

```shell
uv sync
```

## 3. run python code with `uv`

```shell
uv run tester.py
```

## Format

```shell
uv run ruff format <filename>
```

or format all files with,

```shell
uv run ruff format .
```

## Code Check

```shell
uv run ruff check <filename>
```

or check all files with,

```shell
uv run ruff check .
```

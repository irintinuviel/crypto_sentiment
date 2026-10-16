"""Wczytywanie i nadpisywanie parametrów z params.yaml.

Użycie z linii poleceń (np. w Jenkinsie):
    python -m src.params train.epochs=3 data.sample_size=500
"""
import copy
import sys

import yaml

PARAMS_PATH = "params.yaml"


def load_params(path: str = PARAMS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_params(params: dict, path: str = PARAMS_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False, allow_unicode=True)


def override(params: dict, assignments: list[str]) -> dict:
    """Zwraca kopię parametrów z nadpisanymi wartościami w formacie `sekcja.klucz=wartość`."""
    params = copy.deepcopy(params)
    for assignment in assignments:
        key, value = assignment.split("=", 1)
        *sections, last = key.split(".")
        target = params
        for section in sections:
            target = target[section]
        if last not in target:
            raise KeyError(f"Nieznany parametr: {key}")
        target[last] = parse_value(value)
    return params


def parse_value(value: str):
    parsed = yaml.safe_load(value)
    if isinstance(parsed, str):
        # YAML 1.1 nie rozpoznaje zapisu "2e-5" (bez kropki) jako liczby
        try:
            return float(parsed)
        except ValueError:
            pass
    return parsed


if __name__ == "__main__":
    updated = override(load_params(), sys.argv[1:])
    save_params(updated)
    print(yaml.safe_dump(updated, sort_keys=False, allow_unicode=True))

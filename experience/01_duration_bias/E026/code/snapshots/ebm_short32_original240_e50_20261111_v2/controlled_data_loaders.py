"""Importable name for the controlled evaluation data loader.

The archived training loader occupies ``data_loaders`` in the rank process.
PyTorch spawn workers must import the evaluation dataset by its module name.
Executing the native source here preserves that distinct, importable name.
"""
from pathlib import Path

_source = Path(__file__).with_name('data_loaders.py')
exec(compile(_source.read_bytes(), str(_source), 'exec'))

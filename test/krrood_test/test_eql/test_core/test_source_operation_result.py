from dataclasses import dataclass

import pytest

from krrood.entity_query_language.core.base_expressions import OperationResult
from krrood.entity_query_language.factories import (
    and_,
    entity,
    inference,
    variable_from,
)


@dataclass(frozen=True)
class Item:
    value: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_result(expr, sources=None):
    """Return the first OperationResult from _evaluate_()."""
    return next(expr._evaluate_(sources))


def _all_results(expr, sources=None):
    return list(expr._evaluate_(sources))


# ---------------------------------------------------------------------------
# source_operation_result is None when no OperationResult is passed as input
# ---------------------------------------------------------------------------


def test_source_none_on_top_level_evaluate():
    """Top-level evaluate() produces results with source_operation_result=None."""
    var = variable_from([1, 2])
    query = entity(var)
    query.build()
    for result in query._evaluate_():
        assert result.source_operation_result is None


def test_source_set_when_empty_operation_result_passed():
    """source_operation_result is set to the input OperationResult even when empty.
    Passing a plain dict is no longer valid; an OperationResult must be used."""
    var = variable_from([1, 2])
    sentinel = OperationResult({})
    for result in _all_results(var, sentinel):
        assert result.source_operation_result is sentinel


def test_source_none_when_no_sources():
    """source_operation_result is None when _evaluate_() is called without sources."""
    var = variable_from([1, 2, 3])
    for result in _all_results(var):
        assert result.source_operation_result is None


# ---------------------------------------------------------------------------
# source_operation_result is set when an OperationResult is passed as input
# ---------------------------------------------------------------------------


def test_source_set_on_variable():
    """Variable results carry source_operation_result when an OperationResult is input."""
    source_var = variable_from([99])
    incoming = _first_result(source_var)

    target_var = variable_from([1, 2, 3])
    for result in _all_results(target_var, incoming):
        assert result.source_operation_result is incoming


def test_source_set_on_query():
    """Query results carry source_operation_result when an OperationResult is input."""
    source_var = variable_from([0])
    incoming = _first_result(source_var)

    val = variable_from([10, 20])
    query = entity(val)
    query.build()
    for result in _all_results(query, incoming):
        assert result.source_operation_result is incoming


def test_source_set_on_query_with_where():
    """Query+where results carry source_operation_result regardless of filter depth."""
    source_var = variable_from([0])
    incoming = _first_result(source_var)

    val = variable_from([1, 6, 11])
    query = entity(val).where(val > 5)
    query.build()
    true_results = [r for r in _all_results(query, incoming) if r.is_true]
    assert len(true_results) == 2
    for result in true_results:
        assert result.source_operation_result is incoming


def test_source_set_on_and_expression():
    """AND expression results carry source_operation_result when OperationResult is input."""
    source_var = variable_from([0])
    incoming = _first_result(source_var)

    val = variable_from([6])
    and_expr = and_(val > 5, val < 10)
    for result in _all_results(and_expr, incoming):
        assert result.source_operation_result is incoming


# ---------------------------------------------------------------------------
# Short-circuit path: expression ID already in incoming bindings
# ---------------------------------------------------------------------------


def test_source_set_on_short_circuit_path():
    """Short-circuit path (ID already in bindings) also sets source_operation_result."""
    var = variable_from([42])
    # Evaluate once to get a result where var._id_ is already in bindings.
    incoming = _first_result(var)
    assert var._id_ in incoming.bindings

    # Calling _evaluate_() again with that result hits the short-circuit branch.
    results = _all_results(var, incoming)
    assert len(results) == 1
    assert results[0].source_operation_result is incoming


# ---------------------------------------------------------------------------
# Chain traversal
# ---------------------------------------------------------------------------


def test_source_chain_is_traversable():
    """source_operation_result forms a traversable chain across evaluation stages."""
    stage1_var = variable_from([1])
    stage1_result = _first_result(stage1_var)
    assert stage1_result.source_operation_result is None

    stage2_var = variable_from([10, 20])
    stage2_results = _all_results(stage2_var, stage1_result)
    assert len(stage2_results) == 2
    for r in stage2_results:
        assert r.source_operation_result is stage1_result
        assert r.source_operation_result.source_operation_result is None


def test_source_chain_two_hops():
    """Three-stage pipeline: each result points back one hop correctly."""
    v1 = variable_from([1])
    r1 = _first_result(v1)

    v2 = variable_from([10])
    r2 = _first_result(v2, r1)
    assert r2.source_operation_result is r1

    v3 = variable_from([100])
    r3 = _first_result(v3, r2)
    assert r3.source_operation_result is r2
    assert r3.source_operation_result.source_operation_result is r1


# ---------------------------------------------------------------------------
# source_operation_result does not affect bindings or truth value
# ---------------------------------------------------------------------------


def test_source_does_not_affect_bindings():
    """Passing an OperationResult as source does not change binding values."""
    source_var = variable_from([99])
    incoming = _first_result(source_var)

    var = variable_from([7])
    result_with_source = _first_result(var, incoming)
    result_without_source = _first_result(var)

    assert result_with_source.bindings[var._id_] == result_without_source.bindings[var._id_]
    assert result_with_source.is_true == result_without_source.is_true


def test_source_does_not_affect_eq():
    """__eq__ ignores source_operation_result: two results identical in content compare equal
    even when their source_operation_result differs."""
    var = variable_from([7])
    r1 = _first_result(var)
    r2 = _first_result(var)

    # Assign different source results to each
    r1.source_operation_result = _first_result(variable_from([1]))
    r2.source_operation_result = _first_result(variable_from([2]))

    # Content (bindings, truth value, operand, previous chain) is identical → equal
    assert r1 == r2


# ---------------------------------------------------------------------------
# source_operation_result with inference (InstantiatedVariable)
# ---------------------------------------------------------------------------


def test_source_set_on_inference_results():
    """InstantiatedVariable results also carry source_operation_result."""
    source_var = variable_from([0])
    incoming = _first_result(source_var)

    val = variable_from([3, 7])
    item_inf = inference(Item)
    query = entity(item_inf(value=val))
    query.build()
    results = [r for r in _all_results(query, incoming) if r.is_true]
    assert len(results) == 2
    for r in results:
        assert r.source_operation_result is incoming

"""Tests for semantic type system and connection validation."""

import pytest
from django.test import TestCase

from flowchart.types import SemanticType, validate_connection


class TestSemanticType(TestCase):
    def test_parse(self):
        st = SemanticType.parse("metric:cpk")
        assert st.category == "metric"
        assert st.subtype == "cpk"

    def test_parse_array(self):
        st = SemanticType.parse("metric:cycle_time[]")
        assert st.is_array is True
        assert st.subtype == "cycle_time"

    def test_parse_wildcard(self):
        st = SemanticType.parse("chart:*")
        assert st.is_wildcard is True

    def test_exact_match(self):
        a = SemanticType.parse("metric:cpk")
        b = SemanticType.parse("metric:cpk")
        assert a.accepts(b)

    def test_wildcard_accepts(self):
        wild = SemanticType.parse("metric:*")
        specific = SemanticType.parse("metric:cpk")
        assert wild.accepts(specific)

    def test_raj_test_rejects(self):
        cpk = SemanticType.parse("metric:cpk")
        pval = SemanticType.parse("metric:p_value")
        assert not cpk.accepts(pval)

    def test_cross_category_rejects(self):
        metric = SemanticType.parse("metric:cpk")
        data = SemanticType.parse("data:column")
        assert not data.accepts(metric)

    def test_spec_not_metric(self):
        wild = SemanticType.parse("metric:*")
        spec = SemanticType.parse("spec:usl")
        assert not wild.accepts(spec)

    def test_array_rejects_scalar(self):
        arr = SemanticType.parse("metric:cycle_time[]")
        scalar = SemanticType.parse("metric:cycle_time")
        assert not arr.accepts(scalar)

    def test_invalid_parse_raises(self):
        with pytest.raises(ValueError):
            SemanticType.parse("cpk")
        with pytest.raises(ValueError):
            SemanticType.parse("")


class TestValidateConnection(TestCase):
    def test_valid_connection(self):
        result = validate_connection("metric:cpk", "metric:*")
        assert result["valid"] is True

    def test_invalid_connection(self):
        result = validate_connection("metric:cpk", "data:column")
        assert result["valid"] is False
        assert "mismatch" in result["error"].lower()

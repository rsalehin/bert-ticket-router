"""Tests for src.errors (custom exception hierarchy)."""

from __future__ import annotations

import pytest

from src.errors import (
    AppError,
    IntentNotInRoutingError,
    ModelNotLoadedError,
    PersistenceError,
    ValidationError,
)


class TestAppErrorBase:
    """AppError is the base class for all domain exceptions."""

    def test_is_exception(self) -> None:
        assert issubclass(AppError, Exception)

    def test_default_code(self) -> None:
        assert AppError.code == "APP_ERROR"

    def test_default_http_status(self) -> None:
        assert AppError.http_status == 500

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(AppError) as info:
            raise AppError("boom")
        assert "boom" in str(info.value)


class TestValidationError:
    def test_inherits_app_error(self) -> None:
        assert issubclass(ValidationError, AppError)

    def test_code(self) -> None:
        assert ValidationError.code == "VALIDATION_ERROR"

    def test_http_status(self) -> None:
        assert ValidationError.http_status == 400

    def test_caught_as_app_error(self) -> None:
        with pytest.raises(AppError):
            raise ValidationError("bad input")


class TestIntentNotInRoutingError:
    def test_inherits_app_error(self) -> None:
        assert issubclass(IntentNotInRoutingError, AppError)

    def test_code(self) -> None:
        assert IntentNotInRoutingError.code == "INTENT_NOT_IN_ROUTING"

    def test_http_status(self) -> None:
        assert IntentNotInRoutingError.http_status == 500


class TestModelNotLoadedError:
    def test_inherits_app_error(self) -> None:
        assert issubclass(ModelNotLoadedError, AppError)

    def test_code(self) -> None:
        assert ModelNotLoadedError.code == "MODEL_NOT_LOADED"

    def test_http_status(self) -> None:
        assert ModelNotLoadedError.http_status == 503


class TestPersistenceError:
    def test_inherits_app_error(self) -> None:
        assert issubclass(PersistenceError, AppError)

    def test_code(self) -> None:
        assert PersistenceError.code == "PERSISTENCE_ERROR"

    def test_http_status(self) -> None:
        assert PersistenceError.http_status == 503


class TestSubclassesAreDistinct:
    """Each subclass should be distinct, not interchangeable."""

    def test_validation_is_not_persistence(self) -> None:
        assert not issubclass(ValidationError, PersistenceError)

    def test_persistence_is_not_validation(self) -> None:
        assert not issubclass(PersistenceError, ValidationError)

    def test_all_subclasses_have_distinct_codes(self) -> None:
        codes = {
            ValidationError.code,
            IntentNotInRoutingError.code,
            ModelNotLoadedError.code,
            PersistenceError.code,
        }
        assert len(codes) == 4

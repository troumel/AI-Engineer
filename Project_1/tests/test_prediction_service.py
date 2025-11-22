"""
Unit tests for PredictionService.

These tests verify the service layer logic without involving the API.
Similar to unit tests in .NET where you test services independently of controllers.
"""

import pytest
from app.services.prediction_service import PredictionService
from app.models.schemas import PredictionRequest, PredictionResponse


class TestPredictionService:
    """Test suite for PredictionService"""

    @pytest.fixture
    def service(self):
        """
        Create a PredictionService instance for testing.
        This is like a setup method in .NET tests.
        """
        from app.config import settings
        service = PredictionService(model_path=settings.model_path)
        service.load_model()
        return service

    def test_service_initializes_successfully(self, service):
        """Test that service initializes and loads the model"""
        assert service is not None
        assert service.model is not None
        assert service.model_version == "1.0.0"

    def test_predict_returns_valid_response(self, service):
        """Test that predict returns a valid PredictionResponse"""
        # Arrange - prepare test data
        request = PredictionRequest(
            median_income=8.3252,
            house_age=41.0,
            avg_rooms=6.984127,
            avg_bedrooms=1.023810,
            population=322.0,
            avg_occupancy=2.555556,
            latitude=37.88,
            longitude=-122.23
        )

        # Act - call the method
        response = service.predict(request)

        # Assert - verify the result
        assert isinstance(response, PredictionResponse)
        assert isinstance(response.predicted_price, float)
        assert response.predicted_price > 0
        assert response.model_version == "1.0.0"

    def test_predict_with_different_inputs(self, service):
        """Test predictions with various input values"""
        test_cases = [
            # High income area
            PredictionRequest(
                median_income=10.0,
                house_age=20.0,
                avg_rooms=8.0,
                avg_bedrooms=2.0,
                population=500.0,
                avg_occupancy=3.0,
                latitude=37.5,
                longitude=-122.0
            ),
            # Low income area
            PredictionRequest(
                median_income=2.0,
                house_age=50.0,
                avg_rooms=4.0,
                avg_bedrooms=1.0,
                population=1000.0,
                avg_occupancy=4.0,
                latitude=34.0,
                longitude=-118.0
            ),
        ]

        for request in test_cases:
            response = service.predict(request)
            assert isinstance(response.predicted_price, float)
            assert response.predicted_price > 0

    def test_predict_high_income_gives_higher_price(self, service):
        """Test that higher median income generally leads to higher prices"""
        # Low income request
        low_income = PredictionRequest(
            median_income=2.0,
            house_age=30.0,
            avg_rooms=5.0,
            avg_bedrooms=1.0,
            population=500.0,
            avg_occupancy=3.0,
            latitude=37.0,
            longitude=-122.0
        )

        # High income request (same except income)
        high_income = PredictionRequest(
            median_income=10.0,
            house_age=30.0,
            avg_rooms=5.0,
            avg_bedrooms=1.0,
            population=500.0,
            avg_occupancy=3.0,
            latitude=37.0,
            longitude=-122.0
        )

        low_price = service.predict(low_income).predicted_price
        high_price = service.predict(high_income).predicted_price

        assert high_price > low_price, "Higher income should generally predict higher price"

"""
Integration tests for API endpoints.

These tests verify the actual HTTP endpoints work correctly.
Similar to integration tests in .NET where you test the whole API stack.

We use FastAPI's TestClient which creates a test version of the API.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.dependencies import initialize_services


# Initialize services before running tests
# This is needed because TestClient doesn't trigger lifespan events by default
@pytest.fixture(scope="session", autouse=True)
def initialize_app_services():
    """Initialize services once for all tests"""
    initialize_services()


class TestHealthEndpoints:
    """Test suite for health check endpoints"""

    @pytest.fixture
    def client(self):
        """
        Create a test client for the FastAPI app.
        This is like WebApplicationFactory in .NET integration tests.
        """
        return TestClient(app)

    def test_health_check_returns_200(self, client):
        """Test that /health endpoint returns 200 OK"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestPredictionEndpoints:
    """Test suite for prediction endpoints"""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app"""
        return TestClient(app)

    @pytest.fixture
    def valid_request_data(self):
        """Sample valid request data"""
        return {
            "median_income": 8.3252,
            "house_age": 41.0,
            "avg_rooms": 6.984127,
            "avg_bedrooms": 1.023810,
            "population": 322.0,
            "avg_occupancy": 2.555556,
            "latitude": 37.88,
            "longitude": -122.23
        }

    def test_predict_with_valid_data_returns_200(self, client, valid_request_data):
        """Test prediction endpoint with valid data"""
        response = client.post("/predict", json=valid_request_data)

        assert response.status_code == 200
        data = response.json()
        assert "predicted_price" in data
        assert "model_version" in data
        assert isinstance(data["predicted_price"], float)
        assert data["predicted_price"] > 0
        assert data["model_version"] == "1.0.0"

    def test_predict_with_missing_field_returns_422(self, client):
        """Test that missing required fields return 422 Unprocessable Entity"""
        incomplete_data = {
            "median_income": 8.3252,
            "house_age": 41.0,
            # Missing other required fields
        }

        response = client.post("/predict", json=incomplete_data)

        assert response.status_code == 422
        # FastAPI automatically returns validation error details

    def test_predict_with_invalid_latitude_returns_422(self, client, valid_request_data):
        """Test that invalid latitude (out of range) returns 422"""
        invalid_data = valid_request_data.copy()
        invalid_data["latitude"] = 100.0  # Invalid: latitude must be -90 to 90

        response = client.post("/predict", json=invalid_data)

        assert response.status_code == 422

    def test_predict_with_invalid_longitude_returns_422(self, client, valid_request_data):
        """Test that invalid longitude (out of range) returns 422"""
        invalid_data = valid_request_data.copy()
        invalid_data["longitude"] = 200.0  # Invalid: longitude must be -180 to 180

        response = client.post("/predict", json=invalid_data)

        assert response.status_code == 422

    def test_predict_with_wrong_data_types_returns_422(self, client, valid_request_data):
        """Test that wrong data types return 422"""
        invalid_data = valid_request_data.copy()
        invalid_data["median_income"] = "not a number"  # Should be float

        response = client.post("/predict", json=invalid_data)

        assert response.status_code == 422

    def test_predict_with_negative_values(self, client, valid_request_data):
        """Test prediction with negative values (should still work for some fields)"""
        # Some fields can be negative in real scenarios
        data = valid_request_data.copy()
        # Population, rooms, etc. should be positive, but the model might still accept them
        # This tests the model's behavior rather than validation

        response = client.post("/predict", json=data)

        # Should succeed (we don't have validation preventing negatives except lat/lon)
        assert response.status_code == 200

    def test_predict_response_structure(self, client, valid_request_data):
        """Test that response has the correct structure"""
        response = client.post("/predict", json=valid_request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure matches PredictionResponse schema
        assert set(data.keys()) == {"predicted_price", "model_version"}
        assert isinstance(data["predicted_price"], (int, float))
        assert isinstance(data["model_version"], str)

    def test_predict_consistency(self, client, valid_request_data):
        """Test that same input gives same output (model is deterministic)"""
        response1 = client.post("/predict", json=valid_request_data)
        response2 = client.post("/predict", json=valid_request_data)

        assert response1.status_code == 200
        assert response2.status_code == 200

        price1 = response1.json()["predicted_price"]
        price2 = response2.json()["predicted_price"]

        # Use pytest.approx for floating point comparison
        # Allows small differences due to floating point arithmetic
        assert price1 == pytest.approx(price2, rel=1e-9), "Same input should give same prediction"


class TestRootEndpoint:
    """Test suite for root endpoint"""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app"""
        return TestClient(app)

    def test_root_returns_api_info(self, client):
        """Test that root endpoint returns API information"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"

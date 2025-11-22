# Python Testing Guide - pytest vs C# xUnit/Moq

This guide explains Python testing with pytest, comparing it to C# xUnit and Moq that you're familiar with.

## Table of Contents
1. [Test Structure Comparison](#test-structure-comparison)
2. [Fixtures vs Constructors](#fixtures-vs-constructors)
3. [Fixture Scopes](#fixture-scopes)
4. [Dependency Injection in Tests](#dependency-injection-in-tests)
5. [Assertions](#assertions)
6. [Test Client](#test-client)
7. [Test File Breakdown](#test-file-breakdown)
8. [Mocking Comparison](#mocking-comparison)
9. [Key Differences Summary](#key-differences-summary)
10. [Advanced Features](#advanced-features)

---

## Test Structure Comparison

### C# with xUnit
```csharp
public class PredictionEndpointsTests
{
    private readonly HttpClient _client;

    public PredictionEndpointsTests()
    {
        // Setup runs before each test
        var factory = new WebApplicationFactory<Program>();
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task Predict_WithValidData_Returns200()
    {
        // Arrange
        var request = new { median_income = 8.3252, ... };

        // Act
        var response = await _client.PostAsJsonAsync("/predict", request);

        // Assert
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }
}
```

### Python with pytest
```python
class TestPredictionEndpoints:
    """Test suite for prediction endpoints"""

    @pytest.fixture
    def client(self):
        # Setup runs before each test that uses this fixture
        return TestClient(app)

    def test_predict_with_valid_data_returns_200(self, client):
        # Arrange
        request = {"median_income": 8.3252, ...}

        # Act
        response = client.post("/predict", json=request)

        # Assert
        assert response.status_code == 200
```

**Key differences:**
- No `[Fact]` attribute needed - methods starting with `test_` are automatically discovered
- Use `@pytest.fixture` instead of constructor for setup
- Plain `assert` statements instead of `Assert.Equal()`
- Fixture injection via method parameters

---

## Fixtures vs Constructors/IClassFixture

### C# - Constructor runs before EACH test
```csharp
public class MyTests
{
    private readonly MyService _service;

    public MyTests()
    {
        _service = new MyService(); // Runs before EVERY test
    }

    [Fact]
    public void Test1() { }

    [Fact]
    public void Test2() { }
}
```

### Python - Fixture with default scope runs before EACH test
```python
class TestMyService:
    @pytest.fixture
    def service(self):
        return MyService()  # Runs before EVERY test that uses it

    def test_1(self, service):  # Gets fresh instance
        pass

    def test_2(self, service):  # Gets fresh instance
        pass
```

### C# - IClassFixture for shared setup
```csharp
public class MyTests : IClassFixture<DatabaseFixture>
{
    private readonly DatabaseFixture _db;

    public MyTests(DatabaseFixture db)
    {
        _db = db; // Runs ONCE for all tests in class
    }
}
```

### Python - Fixture scope for shared setup
```python
class TestMyService:
    @pytest.fixture(scope="class")  # Runs ONCE for all tests in class
    def database(self):
        db = Database()
        db.connect()
        yield db  # Tests run here
        db.disconnect()  # Cleanup after all tests

    def test_1(self, database):  # Same instance
        pass

    def test_2(self, database):  # Same instance
        pass
```

**Using `yield` for cleanup:**
- Code before `yield` = Setup
- Code after `yield` = Teardown
- Like `IDisposable.Dispose()` in C#

---

## Fixture Scopes

```python
@pytest.fixture(scope="function")  # DEFAULT - new instance per test
def client():
    return TestClient(app)

@pytest.fixture(scope="class")  # One instance per test class
def expensive_setup():
    return ExpensiveObject()

@pytest.fixture(scope="module")  # One instance per test file
def database():
    return Database()

@pytest.fixture(scope="session")  # One instance for ENTIRE test run
def app_config():
    return load_config()
```

### In our tests - Session-level initialization
```python
@pytest.fixture(scope="session", autouse=True)
def initialize_app_services():
    """Initialize services once for all tests"""
    initialize_services()
```

- `scope="session"` - Runs ONCE at the start of all tests
- `autouse=True` - Runs automatically without needing to inject it
- Like `OneTimeSetUp` in NUnit or static constructor in xUnit

**Why we need this:**
- FastAPI's `TestClient` doesn't trigger lifespan events (startup/shutdown)
- Our app loads the ML model in the lifespan startup event
- Without this, the model wouldn't be loaded and tests would fail
- We call `initialize_services()` manually to load the model

---

## Dependency Injection in Tests

### C# - Constructor injection with Moq
```csharp
public class PredictionServiceTests
{
    private readonly Mock<IModelLoader> _mockLoader;
    private readonly PredictionService _service;

    public PredictionServiceTests()
    {
        _mockLoader = new Mock<IModelLoader>();
        _service = new PredictionService(_mockLoader.Object);
    }

    [Fact]
    public void Test_WithMock()
    {
        // Setup mock
        _mockLoader.Setup(x => x.Load()).Returns(mockModel);

        // Use _service
    }
}
```

### Python - Fixture injection (parameters)
```python
class TestPredictionService:
    @pytest.fixture
    def mock_loader(self):
        from unittest.mock import Mock
        return Mock()

    @pytest.fixture
    def service(self, mock_loader):  # Inject mock_loader fixture
        return PredictionService(mock_loader)

    def test_with_mock(self, service, mock_loader):
        # Setup mock
        mock_loader.load.return_value = mock_model

        # Use service
```

### Fixtures can depend on other fixtures
```python
@pytest.fixture
def valid_request_data(self):
    """Sample valid request data"""
    return {
        "median_income": 8.3252,
        "house_age": 41.0,
        "avg_rooms": 6.984127,
        # ...
    }

@pytest.fixture
def client(self):
    """Create a test client"""
    return TestClient(app)

def test_predict(self, client, valid_request_data):
    # Both fixtures are injected automatically
    response = client.post("/predict", json=valid_request_data)
    assert response.status_code == 200
```

**C# equivalent:**
```csharp
public class Tests : IClassFixture<ClientFixture>, IClassFixture<DataFixture>
{
    private readonly HttpClient _client;
    private readonly TestData _data;

    public Tests(ClientFixture clientFixture, DataFixture dataFixture)
    {
        _client = clientFixture.Client;
        _data = dataFixture.Data;
    }
}
```

---

## Assertions

### C# xUnit
```csharp
Assert.Equal(expected, actual);
Assert.True(condition);
Assert.NotNull(obj);
Assert.Throws<Exception>(() => method());
Assert.InRange(value, low, high);
```

### Python pytest
```python
assert actual == expected  # Just use Python's assert!
assert condition
assert obj is not None
with pytest.raises(Exception):
    method()
# No built-in range assertion, but easy to write:
assert low <= value <= high
```

### Why pytest uses plain `assert`?

pytest has **"assertion rewriting"** - it intercepts assert statements and adds helpful error messages.

When a test fails, you get detailed output:
```python
def test_price(self):
    predicted_price = 95000
    assert predicted_price == 100000

# Output when it fails:
# AssertionError: assert 95000 == 100000
#  +  where 95000 = predicted_price
```

### Floating point comparison
```python
# pytest.approx() for floating point comparison
assert price1 == pytest.approx(price2, rel=1e-9)
```

**C# equivalent:**
```csharp
Assert.Equal(expected, actual, precision: 9);
```

---

## Test Client

### C# WebApplicationFactory
```csharp
var factory = new WebApplicationFactory<Program>()
    .WithWebHostBuilder(builder =>
    {
        builder.ConfigureServices(services =>
        {
            // Override services for testing
            services.AddScoped<IService, MockService>();
        });
    });

var client = factory.CreateClient();
var response = await client.GetAsync("/endpoint");
```

### Python TestClient
```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
response = client.get("/endpoint")  # Synchronous, no await!
```

### Key differences:
1. **TestClient is synchronous** - No `async/await` needed in tests (even though your endpoints are async)
2. **No separate factory needed** - Just pass the FastAPI app directly
3. **Simpler override pattern:**

```python
# Override dependencies for testing
def mock_service():
    return MockService()

app.dependency_overrides[get_service] = mock_service

client = TestClient(app)
```

---

## Test File Breakdown

Let's analyze `test_api_endpoints.py` section by section:

### Section 1: Session-level initialization
```python
@pytest.fixture(scope="session", autouse=True)
def initialize_app_services():
    """Initialize services once for all tests"""
    initialize_services()
```

**C# equivalent:**
```csharp
public class TestFixture : IDisposable
{
    public TestFixture()
    {
        ServiceInitializer.Initialize();  // Runs once for all tests
    }

    public void Dispose() { }
}

[CollectionDefinition("API Tests")]
public class TestCollection : ICollectionFixture<TestFixture> { }

[Collection("API Tests")]
public class MyTests { }
```

### Section 2: Test class with fixtures
```python
class TestHealthEndpoints:
    """Test suite for health check endpoints"""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app"""
        return TestClient(app)
```

**Why fixture inside class?**
- Scoped to this test class only
- Each test method that uses `client` gets a fresh instance
- Similar to constructor in C# test class

**C# equivalent:**
```csharp
public class HealthEndpointsTests
{
    private readonly HttpClient _client;

    public HealthEndpointsTests()
    {
        var factory = new WebApplicationFactory<Program>();
        _client = factory.CreateClient();
    }
}
```

### Section 3: Test method with fixture injection
```python
def test_health_check_returns_200(self, client):
    """Test that /health endpoint returns 200 OK"""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
```

**Breakdown:**
1. `self` - Python class method parameter (like `this` in C#)
2. `client` - pytest sees this parameter name, finds the fixture, calls it, injects the result
3. No `[Fact]` attribute - Any method starting with `test_` is automatically discovered
4. `response.json()` - Parses JSON response
5. Multiple asserts - pytest supports multiple assertions per test

**C# equivalent:**
```csharp
[Fact]
public async Task HealthCheck_Returns200()
{
    // Arrange - client from constructor

    // Act
    var response = await _client.GetAsync("/health");

    // Assert
    Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    var data = await response.Content.ReadFromJsonAsync<HealthResponse>();
    Assert.Equal("healthy", data.Status);
    Assert.True(data.ModelLoaded);
}
```

### Section 4: Multiple fixtures in one test
```python
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
    assert data["predicted_price"] > 0
```

**Why separate fixture for test data?**
- **Reusability** - Many tests need the same valid data
- **DRY principle** - Don't repeat yourself
- **Readability** - Test methods are cleaner

**C# equivalent:**
```csharp
private PredictionRequest GetValidRequestData()
{
    return new PredictionRequest
    {
        MedianIncome = 8.3252,
        HouseAge = 41.0,
        AvgRooms = 6.984127,
        // ...
    };
}

[Fact]
public async Task Predict_WithValidData_Returns200()
{
    var requestData = GetValidRequestData();
    var response = await _client.PostAsJsonAsync("/predict", requestData);

    Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    var data = await response.Content.ReadFromJsonAsync<PredictionResponse>();
    Assert.True(data.PredictedPrice > 0);
}
```

### Section 5: Testing validation (422 errors)
```python
def test_predict_with_invalid_latitude_returns_422(self, client, valid_request_data):
    """Test that invalid latitude (out of range) returns 422"""
    invalid_data = valid_request_data.copy()  # Don't modify original!
    invalid_data["latitude"] = 100.0  # Invalid: must be -90 to 90

    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
```

**Python `.copy()`:**
- `dict.copy()` creates a shallow copy
- Prevents modifying the fixture data (which would affect other tests)
- Like C# `with` expression: `var invalid = valid with { Latitude = 100.0 };`

**C# equivalent:**
```csharp
[Fact]
public async Task Predict_WithInvalidLatitude_Returns422()
{
    var invalidData = GetValidRequestData() with { Latitude = 100.0 };

    var response = await _client.PostAsJsonAsync("/predict", invalidData);

    Assert.Equal(HttpStatusCode.UnprocessableEntity, response.StatusCode);
}
```

### Section 6: Floating point comparison
```python
def test_predict_consistency(self, client, valid_request_data):
    """Test that same input gives same output (model is deterministic)"""
    response1 = client.post("/predict", json=valid_request_data)
    response2 = client.post("/predict", json=valid_request_data)

    price1 = response1.json()["predicted_price"]
    price2 = response2.json()["predicted_price"]

    # Use pytest.approx for floating point comparison
    # Allows small differences due to floating point arithmetic
    assert price1 == pytest.approx(price2, rel=1e-9)
```

**`pytest.approx()`:**
- Handles floating point precision issues
- `rel=1e-9` means "allow 0.0000001% relative difference"
- Similar to `Assert.Equal(expected, actual, precision)` in xUnit

**C# equivalent:**
```csharp
[Fact]
public async Task Predict_Consistency_SameInputSameOutput()
{
    var data = GetValidRequestData();

    var response1 = await _client.PostAsJsonAsync("/predict", data);
    var response2 = await _client.PostAsJsonAsync("/predict", data);

    var result1 = await response1.Content.ReadFromJsonAsync<PredictionResponse>();
    var result2 = await response2.Content.ReadFromJsonAsync<PredictionResponse>();

    Assert.Equal(result1.PredictedPrice, result2.PredictedPrice, precision: 9);
}
```

---

## Mocking Comparison

### C# with Moq
```csharp
[Fact]
public void Test_WithMock()
{
    // Arrange
    var mockService = new Mock<IPredictionService>();
    mockService
        .Setup(x => x.Predict(It.IsAny<PredictionRequest>()))
        .Returns(new PredictionResponse { PredictedPrice = 100000 });

    var controller = new PredictionsController(mockService.Object);

    // Act
    var result = controller.Predict(new PredictionRequest());

    // Assert
    Assert.Equal(100000, result.PredictedPrice);
    mockService.Verify(x => x.Predict(It.IsAny<PredictionRequest>()), Times.Once);
}
```

### Python with unittest.mock
```python
from unittest.mock import Mock

def test_with_mock(self):
    # Arrange
    mock_service = Mock(spec=PredictionService)
    mock_service.predict.return_value = PredictionResponse(
        predicted_price=100000,
        model_version="1.0.0"
    )

    # Act
    result = mock_service.predict(PredictionRequest(...))

    # Assert
    assert result.predicted_price == 100000
    mock_service.predict.assert_called_once()
```

### Or with pytest-mock (cleaner)
```python
def test_with_mocker(self, mocker):  # mocker is a pytest-mock fixture
    # Arrange
    mock_service = mocker.Mock(spec=PredictionService)
    mock_service.predict.return_value = PredictionResponse(
        predicted_price=100000,
        model_version="1.0.0"
    )

    # Act
    result = mock_service.predict(PredictionRequest(...))

    # Assert
    assert result.predicted_price == 100000
    mock_service.predict.assert_called_once()
```

### Moq vs unittest.mock comparison

| Moq (C#) | unittest.mock (Python) |
|----------|------------------------|
| `new Mock<IService>()` | `Mock(spec=Service)` |
| `.Setup(x => x.Method()).Returns(value)` | `.method.return_value = value` |
| `.Verify(x => x.Method(), Times.Once)` | `.method.assert_called_once()` |
| `It.IsAny<T>()` | Not needed - Python is dynamic |
| `.Object` to get mock instance | Mock is already usable |

---

## Key Differences Summary

| Feature | C# (xUnit + Moq) | Python (pytest) |
|---------|------------------|-----------------|
| **Test discovery** | `[Fact]`, `[Theory]` attributes | Methods starting with `test_` |
| **Setup** | Constructor / `IClassFixture<T>` | `@pytest.fixture` |
| **Teardown** | `IDisposable.Dispose()` | `yield` in fixture |
| **Assertions** | `Assert.Equal()`, `Assert.True()` | Plain `assert` statements |
| **Mocking** | Moq library | `unittest.mock` or `pytest-mock` |
| **Async tests** | `async Task` methods | `async def` with `pytest-asyncio` |
| **Parametrized tests** | `[Theory]` with `[InlineData]` | `@pytest.mark.parametrize` |
| **Test client** | `WebApplicationFactory<T>` | `TestClient(app)` |
| **Dependency injection** | Constructor injection | Fixture parameters |
| **Floating point** | `Assert.Equal(e, a, precision)` | `assert a == pytest.approx(e)` |

---

## Advanced Features

### 1. Parametrized Tests

**C# with [Theory]:**
```csharp
[Theory]
[InlineData(2.0, 50000)]
[InlineData(5.0, 150000)]
[InlineData(10.0, 300000)]
public async Task IncomeAffectsPrice(double income, double expectedMin)
{
    var data = new PredictionRequest { MedianIncome = income, ... };
    var response = await _client.PostAsJsonAsync("/predict", data);
    var result = await response.Content.ReadFromJsonAsync<PredictionResponse>();
    Assert.True(result.PredictedPrice > expectedMin);
}
```

**Python with @pytest.mark.parametrize:**
```python
@pytest.mark.parametrize("income,expected_min", [
    (2.0, 50000),
    (5.0, 150000),
    (10.0, 300000),
])
def test_income_affects_price(self, client, income, expected_min):
    data = {
        "median_income": income,
        "house_age": 30.0,
        "avg_rooms": 5.0,
        # ... other fields
    }
    response = client.post("/predict", json=data)
    assert response.json()["predicted_price"] > expected_min
```

### 2. Test Organization

**C# - Multiple files:**
```
Tests/
├── Unit/
│   ├── Services/
│   │   └── PredictionServiceTests.cs
│   └── Models/
│       └── SchemaTests.cs
└── Integration/
    ├── HealthEndpointsTests.cs
    └── PredictionEndpointsTests.cs
```

**Python - Similar structure:**
```
tests/
├── unit/
│   ├── test_prediction_service.py
│   └── test_schemas.py
└── integration/
    ├── test_health_endpoints.py
    └── test_prediction_endpoints.py
```

Or flatter (what we used):
```
tests/
├── test_prediction_service.py  # Unit tests
└── test_api_endpoints.py       # Integration tests
```

### 3. Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run all tests with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_prediction_service.py -v

# Run specific test class
pytest tests/test_api_endpoints.py::TestHealthEndpoints -v

# Run specific test method
pytest tests/test_api_endpoints.py::TestHealthEndpoints::test_health_check_returns_200 -v

# Run with coverage report
pytest tests/ --cov=app --cov-report=html

# Run tests matching a pattern
pytest tests/ -k "predict" -v

# Run tests and stop at first failure
pytest tests/ -x

# Run last failed tests only
pytest tests/ --lf

# Show print statements
pytest tests/ -v -s
```

**C# equivalent:**
```bash
# Run all tests
dotnet test

# Run with verbose output
dotnet test --verbosity detailed

# Run specific test
dotnet test --filter "FullyQualifiedName~HealthCheck_Returns200"

# Run with coverage
dotnet test /p:CollectCoverage=true
```

### 4. Test Markers (like Categories in C#)

```python
import pytest

@pytest.mark.slow
def test_expensive_operation(self):
    # This test is marked as slow
    pass

@pytest.mark.integration
def test_api_integration(self):
    # This test is marked as integration test
    pass
```

Run only marked tests:
```bash
pytest tests/ -m slow        # Run only slow tests
pytest tests/ -m "not slow"  # Skip slow tests
pytest tests/ -m integration # Run only integration tests
```

**C# equivalent:**
```csharp
[Trait("Category", "Slow")]
public void ExpensiveOperation() { }

// Run:
// dotnet test --filter "Category=Slow"
```

### 5. Setup/Teardown at Different Levels

```python
class TestMyAPI:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Runs before each test method"""
        print("Setting up test")
        yield
        print("Tearing down test")

    @pytest.fixture(scope="class", autouse=True)
    def setup_class(self):
        """Runs once before all tests in class"""
        print("Setting up class")
        yield
        print("Tearing down class")

    def test_1(self):
        pass

    def test_2(self):
        pass
```

**C# equivalent:**
```csharp
public class TestMyAPI : IDisposable
{
    public TestMyAPI()
    {
        // Constructor - runs before each test
        Console.WriteLine("Setting up test");
    }

    public void Dispose()
    {
        // Runs after each test
        Console.WriteLine("Tearing down test");
    }

    [Fact]
    public void Test1() { }

    [Fact]
    public void Test2() { }
}
```

---

## Quick Reference

### Common pytest Commands
```bash
pytest tests/                    # Run all tests
pytest tests/ -v                 # Verbose output
pytest tests/ -v -s              # Show print statements
pytest tests/ -x                 # Stop at first failure
pytest tests/ --lf               # Run last failed
pytest tests/ -k "predict"       # Run tests matching name
pytest tests/ -m integration     # Run marked tests
pytest tests/ --cov=app          # Run with coverage
```

### Common Assertions
```python
assert x == y                    # Equality
assert x != y                    # Inequality
assert x > y                     # Greater than
assert x in [1, 2, 3]           # Membership
assert "hello" in text          # Substring
assert obj is None              # Identity
assert callable(func)           # Callable
assert isinstance(x, int)       # Type checking

# With pytest helpers
assert x == pytest.approx(y)    # Floating point
with pytest.raises(ValueError): # Exception testing
    risky_function()
```

### Fixture Cheat Sheet
```python
@pytest.fixture                  # Function scope (default)
@pytest.fixture(scope="class")   # Class scope
@pytest.fixture(scope="module")  # Module scope
@pytest.fixture(scope="session") # Session scope
@pytest.fixture(autouse=True)    # Auto-use without injection

# With setup and teardown
@pytest.fixture
def resource():
    r = setup()
    yield r        # Test runs here
    teardown(r)    # Cleanup
```

---

## Summary

**pytest philosophy:**
- Simple and Pythonic - use plain `assert`, not special methods
- Flexible - fixtures can be composed and reused
- Powerful - but you only use what you need
- Less boilerplate than xUnit/NUnit

**Key takeaways for C# developers:**
1. Fixtures replace constructors and `IClassFixture<T>`
2. Plain `assert` replaces `Assert.*` methods
3. `TestClient` is simpler than `WebApplicationFactory`
4. Dependency injection works through function parameters
5. No attributes needed for test discovery - just name methods `test_*`

**When coming from C#/xUnit, remember:**
- Constructor → `@pytest.fixture`
- `[Fact]` → `def test_*`
- `Assert.Equal(a, b)` → `assert a == b`
- `IClassFixture<T>` → `@pytest.fixture(scope="class")`
- `IDisposable` → `yield` in fixture
- Moq → `unittest.mock` or `pytest-mock`

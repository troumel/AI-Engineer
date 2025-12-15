class Settings:
    app_name: str = "House Price Prediction API"
    app_version: str = "1.0.0"
    environment: str = "development"
    log_level: str = "INFO"
    model_path: str = "../models/house_price_model.joblib"


settings = Settings()

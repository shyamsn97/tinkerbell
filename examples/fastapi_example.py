from abc import ABC, abstractmethod
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import uvicorn


# Request/Response models
class PredictionRequest(BaseModel):
    input_data: str
    parameters: Dict[str, Any] = {}


class PredictionResponse(BaseModel):
    output: str
    model_name: str
    metadata: Dict[str, Any] = {}


# Abstract Base Class
class BaseInferenceService(ABC):
    """Abstract base class for inference services"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.app = FastAPI(title=f"{model_name} Inference Service")
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "model": self.model_name}
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            try:
                result = await self.predict_async(request)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/info")
        async def info():
            return self.get_model_info()
    
    @abstractmethod
    async def predict_async(self, request: PredictionRequest) -> PredictionResponse:
        """Abstract method that must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the model"""
        pass

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the FastAPI server"""
        uvicorn.run(self.app, host=host, port=port)


# Concrete Implementation 1: Echo Service
class EchoInferenceService(BaseInferenceService):
    """Simple echo service that returns the input"""
    
    def __init__(self):
        super().__init__(model_name="Echo Model")
        self.call_count = 0
    
    async def predict_async(self, request: PredictionRequest) -> PredictionResponse:
        self.call_count += 1
        return PredictionResponse(
            output=f"Echo: {request.input_data}",
            model_name=self.model_name,
            metadata={
                "call_count": self.call_count,
                "parameters_received": request.parameters
            }
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "type": "echo",
            "total_calls": self.call_count,
            "description": "Simple echo service for testing"
        }


# Concrete Implementation 2: Uppercase Service
class UppercaseInferenceService(BaseInferenceService):
    """Service that converts text to uppercase"""
    
    def __init__(self):
        super().__init__(model_name="Uppercase Model")
        self.processed_chars = 0
    
    async def predict_async(self, request: PredictionRequest) -> PredictionResponse:
        result = request.input_data.upper()
        self.processed_chars += len(request.input_data)
        
        return PredictionResponse(
            output=result,
            model_name=self.model_name,
            metadata={
                "input_length": len(request.input_data),
                "total_chars_processed": self.processed_chars
            }
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "type": "transformer",
            "total_characters_processed": self.processed_chars,
            "description": "Converts input text to uppercase"
        }


# Concrete Implementation 3: Mock ML Service
class MockMLInferenceService(BaseInferenceService):
    """Mock ML service with more complex logic"""
    
    def __init__(self, model_version: str = "v1.0"):
        super().__init__(model_name=f"MockML-{model_version}")
        self.model_version = model_version
        self.predictions = []
    
    async def predict_async(self, request: PredictionRequest) -> PredictionResponse:
        # Simulate some ML processing
        confidence = request.parameters.get("confidence_threshold", 0.85)
        
        prediction_result = {
            "prediction": "positive" if len(request.input_data) > 10 else "negative",
            "confidence": confidence,
            "input_length": len(request.input_data)
        }
        
        self.predictions.append(prediction_result)
        
        return PredictionResponse(
            output=prediction_result["prediction"],
            model_name=self.model_name,
            metadata={
                "confidence": confidence,
                "prediction_details": prediction_result,
                "total_predictions": len(self.predictions)
            }
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.model_version,
            "type": "mock_ml_classifier",
            "total_predictions": len(self.predictions),
            "description": "Mock ML classifier for demonstration"
        }


# Usage example
if __name__ == "__main__":
    # Choose which service to run
    import sys
    
    service_type = sys.argv[1] if len(sys.argv) > 1 else "echo"
    
    if service_type == "echo":
        service = EchoInferenceService()
    elif service_type == "uppercase":
        service = UppercaseInferenceService()
    elif service_type == "mockml":
        service = MockMLInferenceService(model_version="v2.0")
    else:
        print(f"Unknown service type: {service_type}")
        print("Available types: echo, uppercase, mockml")
        sys.exit(1)
    
    print(f"Starting {service.model_name}...")
    service.run(port=8000)
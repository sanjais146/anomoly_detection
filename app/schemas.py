from pydantic import BaseModel, Field
from typing import Optional

class TransactionInput(BaseModel):
    TransactionAmt: float = Field(..., example=150.50)
    ProductCD: str = Field(..., example="W")
    card1: int = Field(..., example=10409)
    card2: Optional[float] = Field(None, example=111.0)
    card3: Optional[float] = Field(None, example=150.0)
    card4: Optional[str] = Field(None, example="visa")
    card5: Optional[float] = Field(None, example=226.0)
    card6: Optional[str] = Field(None, example="debit")
    addr1: Optional[float] = Field(None, example=170.0)
    addr2: Optional[float] = Field(None, example=87.0)
    dist1: Optional[float] = Field(None, example=19.0)
    P_emaildomain: Optional[str] = Field(None, example="gmail.com")
    R_emaildomain: Optional[str] = Field(None, example="yahoo.com")
    DeviceInfo: Optional[str] = Field(None, example="Windows")
    
class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: str
    risk_level: str
    threshold: float
    model: str
    signals: list[str]

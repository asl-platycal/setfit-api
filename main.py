from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from setfit import SetFitModel
import uvicorn
import os

# Carregar modelo
print("🚀 Carregando modelo SetFit...")
model = SetFitModel.from_pretrained("Platypical/classificador-intencao-ptbr")
print("✅ Modelo carregado com sucesso!")

app = FastAPI(title="SetFit Classificador de Intenção")

class TextInput(BaseModel):
    inputs: str

@app.post("/classify")
async def classify(data: TextInput):
    """
    Classifica a intenção do texto
    """
    try:
        # Fazer predição
        pred = model.predict([data.inputs])[0]
        proba = model.predict_proba([data.inputs])[0]
        
        # Formatar resposta
        results = []
        for i, label in enumerate(model.labels):
            results.append({
                "label": label,
                "score": float(proba[i])
            })
        
        # Ordenar por score (maior primeiro)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "categoria": results[0]["label"],
            "confianca": results[0]["score"],
            "todas": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"status": "API Online ✅", "modelo": "Platypical/classificador-intencao-ptbr"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

### **Passo 7: Testar a API**

Abra no navegador:
```
https://sua-url.up.railway.app

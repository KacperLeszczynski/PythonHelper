import uvicorn
from fastapi import FastAPI
from routers import chat

app = FastAPI()
app.include_router(chat.router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

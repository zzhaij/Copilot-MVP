"""python -m app 直接启动服务。"""
from app.main import app  # noqa: F401
from app.config import get_settings

if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run("app.main:app", host=s.APP_HOST, port=s.APP_PORT, reload=False)

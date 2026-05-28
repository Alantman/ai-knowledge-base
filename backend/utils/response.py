from fastapi.responses import JSONResponse


def success_response(message: str = "操作成功", data: dict = None) -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content={"code": 200, "message": message, "data": data or {}},
    )


def error_response(message: str = "操作失败", code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={"code": code, "message": message, "data": {}},
    )

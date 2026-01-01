async def set_fps(request):
    """
    上位程序调用：POST /set_fps  {"out_fps": 20, "param_fps": 30}
    允许只传其中一个。
    """
    data = await request.json()
    ctrl: FpsController = request.app["ctrl"]

    if "out_fps" in data:
        ctrl.set_out_fps(data["out_fps"])
    if "param_fps" in data:
        ctrl.set_param_fps(data["param_fps"])

    out_fps, param_fps = ctrl.get()
    return web.json_response({"out_fps": out_fps, "param_fps": param_fps})


并在 main() 注册路由：

app.router.add_post("/set_fps", set_fps)


上位程序调用示例：

curl -X POST http://<ip>:8080/set_fps \
  -H "Content-Type: application/json" \
  -d '{"out_fps": 15}'
# 演示素材说明

- dashboard.png、comparison.png、mobile.png：当前数据表与四曲线界面的浏览器截图。
- browser_check.json：浏览器验收，包括自动推进、固定Canvas节点、基线切换、连续语音结果事件、整日运行和移动端。
- paired_day.mp4：保留的历史户型可视化，使用最终新日复验首日的真实策略轨迹；不是当前前端录屏，不作为额外独立实验。

当前演示以 `python demo/app.py --checkpoint results/layout/mappo42/best_actor.pt --controller lookahead --horizon 32` 为准。浏览器语音验收采用事件注入，不代表麦克风或声学识别已经实测。

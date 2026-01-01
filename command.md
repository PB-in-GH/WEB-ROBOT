查看性能：
lscpu
nvidia-smi
电脑内存
free -h


conda activate MuseTalk

github端口转发
本地powershell：
ssh -N -R 17890:127.0.0.1:7890 pb@101.6.43.215 -p 2666
终端：
export http_proxy=http://127.0.0.1:17890
export https_proxy=http://127.0.0.1:17890

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset no_proxy NO_PROXY

conda指令：
conda create --name 环境名 python=版本号
conda create --name 新名字 --clone 旧名字
conda remove --name 环境名 --all
conda env list

python lite_avatar_old.py --data_dir /home/pb/web-robot/lite-avatar/data/preload --audio_file /home/pb/web-robot/lite-avatar/audio/1.wav --result_dir /home/pb/web-robot/lite-avatar/result

<!-- #高级版本，用音频同步，可以动态调节帧率
python webrtc_server_final.py \
  --data_dir /home/pb/web-robot/lite-avatar/data/preload \
  --audio_file /home/pb/web-robot/lite-avatar/audio/1.wav \
  --out_fps 30 \
  --param_fps 30 \
  --queue_max 1000 \
  --host 0.0.0.0 \
  --port 8080 -->


然后在同一局域网或你能访问该服务器的机器上，用浏览器打开：
http://<服务器IP>:8080/
你应该能看到 <video> 播放头像视频流。


cd /home/pb/web-robot/ai_rtc_avatar
python run_m0_stream_test.py
验证是否可以以不同帧率来控制生成。

MuseTalk
python -m scripts.realtime_inference --inference_config configs/inference/realtime.yaml --skip_save_images --fps 30
#skip意味着不能输出视频。视频的输出result is save to ./results/v15/avatars/avator_1/vid_output/audio_0.mp4
删掉这个文件夹，就需要修改/home/pb/web-robot/MuseTalk/configs/inference/realtime.yaml的preparation=1，源音频也在这里修改即可。

流式输出测试：
PYTHONPATH=/home/pb/web-robot/qoe_transport_lab/src python -m scripts.realtime_inference   --inference_config configs/inference/realtime.yaml   --skip_save_images   --fps 30   --webrtc --webrtc_host 127.0.0.1 --webrtc_port 8080

PYTHONPATH=/home/pb/web-robot/qoe_transport_lab/src python -m qoe_lab.experiments.demo_musetalk_streaming_runner(桌面下载版本)

[GEN] start sec=0 fps=30 submit_ms=0
[GEN] done  sec=0 fps=30 gen_ms=1446 frames=30 ready_ms=1446
含义（你以后论文里可以这样定义）：
sec=k：内容时间第 k 秒（第 k 个 1s 区间）
fps=x：这一秒的生成目标（每秒生成 x 帧）
gen_ms=...：生成这一秒所有帧的 wall-clock 耗时
frames=n：这一秒生成出来的帧数（应当≈fps，除非音频结束/截断）
ready_ms=...：在“发送侧时钟”里，这一秒的帧从什么时候开始可用（你当前实现是 submit_ms + gen_ms）

t=  2.0s  fps=30  recv=3.60Mbps ... e2e=1511.2ms q=12f/0.30MB
含义：
t=2.0s：当前实验时钟（你现在通过 time.sleep(tick) 把它对齐到真实时间了）
fps=30：控制器对当前秒给出的决策 fps（注意：这未必等于生成线程此刻在跑的 fps，下面会说）
recv：接收端吞吐（TraceRtcEmulator 计算出来的接收速率）
loss：丢包率（trace 注入的损失 + 队列行为）
e2e：端到端时延（随队列积压会越来越大）
q=...：网络队列积压（帧数/字节）
这条线告诉你“网络侧的代价”，与控制器/生成侧互动后形成你论文中的 QoE 曲线。



1224重构后
PYTHONPATH=/home/pb/web-robot/qoe_transport_lab/src:/home/pb/web-robot/MuseTalk \
python -m qoe_lab.experiments.run_musetalk_streaming_trace

这是主程序。


PYTHONPATH=/home/pb/web-robot/qoe_transport_lab/src:/home/pb/web-robot/MuseTalk python /home/pb/web-robot/qoe_transport_lab/tmp/verify_musetalk_video.py
验证我的streaming是真的吗？
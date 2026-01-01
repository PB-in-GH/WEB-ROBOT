import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
import sys
from tqdm import tqdm
import copy
import json
from transformers import WhisperModel

from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor

import shutil
import threading
import queue
import time
import subprocess


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break


def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None


@torch.no_grad()
class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        # 根据版本设置不同的基础路径
        if args.version == "v15":
            self.base_path = f"./results/{args.version}/avatars/{avatar_id}"
        else:  # v1
            self.base_path = f"./results/avatars/{avatar_id}"
            
        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "version": args.version
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0
        self.init()

    def init(self):
        if self.preparation:
            if os.path.exists(self.avatar_path):
                response = input(f"{self.avatar_id} exists, Do you want to re-create it ? (y/n)")
                if response.lower() == "y":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    self.input_latent_list_cycle = torch.load(self.latents_out_path)
                    with open(self.coords_path, 'rb') as f:
                        self.coord_list_cycle = pickle.load(f)
                    input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.frame_list_cycle = read_imgs(input_img_list)
                    with open(self.mask_coords_path, 'rb') as f:
                        self.mask_coords_list_cycle = pickle.load(f)
                    input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                    input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.mask_list_cycle = read_imgs(input_mask_list)
            else:
                print("*********************************")
                print(f"  creating avator: {self.avatar_id}")
                print("*********************************")
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
        else:
            if not os.path.exists(self.avatar_path):
                print(f"{self.avatar_id} does not exist, you should set preparation to True")
                sys.exit()

            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)

            if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:
                response = input(f" 【bbox_shift】 is changed, you need to re-create it ! (c/continue)")
                if response.lower() == "c":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    sys.exit()
            else:
                self.input_latent_list_cycle = torch.load(self.latents_out_path)
                with open(self.coords_path, 'rb') as f:
                    self.coord_list_cycle = pickle.load(f)
                input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.frame_list_cycle = read_imgs(input_img_list)
                with open(self.mask_coords_path, 'rb') as f:
                    self.mask_coords_list_cycle = pickle.load(f)
                input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.mask_list_cycle = read_imgs(input_mask_list)

    def prepare_material(self):
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
            print(f"copy files in {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1] == "png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        idx = -1
        # maker if the bbox is not sufficient
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            if args.version == "v15":
                y2 = y2 + args.extra_margin
                y2 = min(y2, frame.shape[0])
                coord_list[idx] = [x1, y1, x2, y2]  # 更新coord_list中的bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)

            x1, y1, x2, y2 = self.coord_list_cycle[i]
            if args.version == "v15":
                mode = args.parsing_mode
            else:
                mode = "raw"
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=mode)

            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)

        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)

        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))

    def process_frames(self, res_frame_queue, video_len, skip_save_images, on_frame=None, fps: int = 25):
        """
        on_frame: 可选回调
            on_frame(frame_bgr: np.ndarray, meta: dict) -> None
        fps: 仅用于给 meta 里生成一个“理想显示时间戳”（毫秒）。
             后续做“每秒可变 fps”时，这里会改成由调度器/RTC 时钟决定。
        """
        while True:
            if self.idx >= video_len - 1:
                break
            try:
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            bbox = self.coord_list_cycle[self.idx % (len(self.coord_list_cycle))]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % (len(self.frame_list_cycle))])
            x1, y1, x2, y2 = bbox

            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception:
                # 这一帧失败直接跳过，不要阻塞整体
                self.idx += 1
                continue

            mask = self.mask_list_cycle[self.idx % (len(self.mask_list_cycle))]
            mask_crop_box = self.mask_coords_list_cycle[self.idx % (len(self.mask_coords_list_cycle))]
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

            # ---- 关键：把帧吐给外部（不落盘）----
            if on_frame is not None:
                ts_ms = int(round(1000.0 * self.idx / max(1, int(fps))))
                meta = {
                    "idx": int(self.idx),
                    "ts_ms": ts_ms,
                    "fps": int(fps),
                    "avatar_id": str(self.avatar_id),
                }
                try:
                    on_frame(combine_frame, meta)
                except Exception as e:
                    print(f"[warn] on_frame callback error: {e}")

            # 仍保留旧逻辑：需要时可以保存 png（调试用）
            if skip_save_images is False:
                cv2.imwrite(f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.png", combine_frame)

            self.idx += 1


    @torch.no_grad()
    def inference(self, audio_path, out_vid_name, fps, skip_save_images, on_frame=None):
        if not skip_save_images:
            os.makedirs(self.avatar_path + '/tmp', exist_ok=True)

        print("start inference")
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        # Extract audio features
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path, weight_dtype=weight_dtype)
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            device,
            weight_dtype,
            whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=args.audio_padding_length_left,
            audio_padding_length_right=args.audio_padding_length_right,
        )
        print(f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")
        ############################################## inference batch by batch ##############################################
        video_num = len(whisper_chunks)
        res_frame_queue = queue.Queue()
        self.idx = 0
        # Create a sub-thread and start it
        process_thread = threading.Thread(
            target=self.process_frames,
            args=(res_frame_queue, video_num, skip_save_images, on_frame, fps),
            daemon=True,
        )
        process_thread.start()

        gen = datagen(whisper_chunks,
                     self.input_latent_list_cycle,
                     self.batch_size)
        start_time = time.time()
        res_frame_list = []

        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))):
            audio_feature_batch = pe(whisper_batch.to(device))
            latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)

            pred_latents = unet.model(latent_batch,
                                    timesteps,
                                    encoder_hidden_states=audio_feature_batch).sample
            pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)
        # Close the queue and sub-thread after all tasks are completed
        process_thread.join()

        if args.skip_save_images is True:
            print('Total process time of {} frames without saving images = {}s'.format(
                video_num,
                time.time() - start_time))
        else:
            print('Total process time of {} frames including saving images = {}s'.format(
                video_num,
                time.time() - start_time))

        if out_vid_name is not None and args.skip_save_images is False:
            # optional
            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {self.avatar_path}/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {self.avatar_path}/temp.mp4"
            print(cmd_img2video)
            os.system(cmd_img2video)

            output_vid = os.path.join(self.video_out_path, out_vid_name + ".mp4")  # on
            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {self.avatar_path}/temp.mp4 {output_vid}"
            print(cmd_combine_audio)
            os.system(cmd_combine_audio)

            os.remove(f"{self.avatar_path}/temp.mp4")
            shutil.rmtree(f"{self.avatar_path}/tmp")
            print(f"result is save to {output_vid}")
        print("\n")


if __name__ == "__main__":
    '''
    This script is used to simulate online chatting and applies necessary pre-processing such as face detection and face parsing in advance. During online chatting, only UNet and the VAE decoder are involved, which makes MuseTalk real-time.
    '''

    parser = argparse.ArgumentParser()
    
    # WebRTC streamer (for real-time viewing)
    parser.add_argument("--webrtc", action="store_true", help="Enable WebRTC streaming viewer")
    parser.add_argument("--webrtc_host", type=str, default="127.0.0.1", help="WebRTC server bind host")
    parser.add_argument("--webrtc_port", type=int, default=8080, help="WebRTC server port")
    parser.add_argument("--webrtc_queue_maxsize", type=int, default=0, help="Max queued frames for WebRTC")

    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Version of MuseTalk: v1 or v15")
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Path to ffmpeg executable")
    parser.add_argument("--gpu_id", type=int, default=2, help="GPU ID to use")
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="Type of VAE model")
    parser.add_argument("--unet_config", type=str, default="./models/musetalk/musetalk.json", help="Path to UNet configuration file")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalk/pytorch_model.bin", help="Path to UNet model weights")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Directory containing Whisper model")
    parser.add_argument("--inference_config", type=str, default="configs/inference/realtime.yaml")
    parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding box shift value")
    parser.add_argument("--result_dir", default='./results', help="Directory for output results")
    parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face cropping")
    parser.add_argument("--fps", type=int, default=40, help="Video frames per second")
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding length for audio")
    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding length for audio")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for inference")
    parser.add_argument("--output_vid_name", type=str, default=None, help="Name of output video file")
    parser.add_argument("--use_saved_coord", action="store_true", help='Use saved coordinates to save time')
    parser.add_argument("--saved_coord", action="store_true", help='Save coordinates for future use')
    parser.add_argument("--parsing_mode", default='jaw', help="Face blending parsing mode")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Width of left cheek region")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Width of right cheek region")
    parser.add_argument("--skip_save_images",
                       action="store_true",
                       help="Whether skip saving images for better generation speed calculation",
                       )

    args = parser.parse_args()

    # Configure ffmpeg path
    if not fast_check_ffmpeg():
        print("Adding ffmpeg to PATH")
        # Choose path separator based on operating system
        path_separator = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

    # Set computing device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model weights
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    timesteps = torch.tensor([0], device=device)

    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)

    # Initialize audio processor and Whisper model
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    # Initialize face parser with configurable parameters based on version
    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
    else:  # v1
        fp = FaceParsing()

    inference_config = OmegaConf.load(args.inference_config)
    print(inference_config)

    # Optional: start WebRTC streamer

    for avatar_id in inference_config:
        data_preparation = inference_config[avatar_id]["preparation"]
        video_path = inference_config[avatar_id]["video_path"]
        if args.version == "v15":
            bbox_shift = 0
        else:
            bbox_shift = inference_config[avatar_id]["bbox_shift"]
        avatar = Avatar(
            avatar_id=avatar_id,
            video_path=video_path,
            bbox_shift=bbox_shift,
            batch_size=args.batch_size,
            preparation=data_preparation)

        audio_clips = inference_config[avatar_id]["audio_clips"]
        for audio_num, audio_path in audio_clips.items():
            from qoe_lab.rtc.webrtc_streamer import WebRtcStreamer, WebRtcStreamerConfig

            streamer = None
            on_frame_cb = None
            if args.webrtc:
                scfg = WebRtcStreamerConfig(
                    host=args.webrtc_host,
                    port=int(args.webrtc_port),
                    queue_maxsize=int(args.webrtc_queue_maxsize),
                    audio_path=str(audio_path),   # 关键：把 wav 传进去
                    audio_loop=False
                )
                streamer = WebRtcStreamer(scfg)
                streamer.start()
                print(f"[webrtc] viewer: http://{args.webrtc_host}:{args.webrtc_port}/")
                print("[webrtc] Please open the viewer URL now. Press Enter to START inference.")
                input()  # <-- 关键：开始推理前等待你打开浏览器并连接


                # ====== 生成->本地缓存 队列（无限长：queue.Queue() 本身不设上限）======
                gen_queue: "queue.Queue[tuple[np.ndarray, int]]" = queue.Queue()

                def on_frame_cb(frame_bgr, meta):
                    # 关键：使用 meta["ts_ms"]（由 process_frames 按 idx/fps 生成的“内容时间”）
                    # 不要用 wall-clock 覆盖它
                    ts_ms = int(meta.get("ts_ms", 0))
                    gen_queue.put((frame_bgr, ts_ms))

                # ====== 上层调度接口（占位：你后面把它替换成真正的策略模块即可）======
                # 约定：scheduler.get_fps(second_index)->int
                class _ConstScheduler:
                    def __init__(self, fps: int):
                        self._fps = int(fps)
                    def get_fps(self, sec: int) -> int:
                        return self._fps

                scheduler = _ConstScheduler(30)

                class _StepScheduler:
                    """
                    简单分段策略：
                    0-4 秒：10 fps
                    5-9 秒：2 fps
                    10-14 秒：15 fps
                    之后：10 fps
                    """
                    def get_fps(self, sec: int) -> int:
                        if 0 <= sec <= 4:
                            return 10
                        if 5 <= sec <= 9:
                            return 2
                        if 10 <= sec <= 14:
                            return 15
                        return 10

                # scheduler = _StepScheduler()


                # ====== 发送线程：按“每秒 fps 决策”从缓存取帧并推给 WebRTC =======
                stop_send = threading.Event()

                def _select_evenly(frames_in_sec, k: int):
                    # frames_in_sec: list[(frame, src_ts_ms)]，长度 N
                    # 选 k 个尽量均匀分布的索引
                    N = len(frames_in_sec)
                    if k <= 0 or N == 0:
                        return []
                    if k >= N:
                        return frames_in_sec
                    # 均匀采样：round(i*(N-1)/(k-1))
                    out = []
                    for i in range(k):
                        idx = int(round(i * (N - 1) / (k - 1))) if k > 1 else int(round((N - 1) / 2))
                        out.append(frames_in_sec[idx])
                    return out


                def send_loop():
                    """
                    把 MuseTalk “按基准 fps 生成的帧序列”重整形为“按上层每秒 fps 决策发送”的序列。
                    ts_ms 的定义：第 s 秒内第 j 帧的呈现时间戳 = s*1000 + j*1000/fps[s]

                    关键改动：
                    1) gen_queue.get(timeout=...)，避免 stop_send 置位后线程卡死
                    2) 退出前 flush 最后一个 bucket，避免尾秒丢失
                    """
                    cur_bucket = []  # list[(frame_bgr, src_ts_ms)]
                    last_src_sec = None

                    def _flush_bucket(sec: int, bucket):
                        """把 bucket 按策略发出去，并打印统计。"""
                        if bucket is None:
                            return
                        target_fps = int(scheduler.get_fps(sec))
                        chosen = _select_evenly(bucket, target_fps)

                        sent_cnt = 0
                        if target_fps > 0:
                            for j, (fbgr, _t) in enumerate(chosen):
                                ts_out = int(sec * 1000 + round(j * 1000.0 / target_fps))
                                streamer.push_frame(fbgr, ts_ms=ts_out)
                                sent_cnt += 1

                        print(
                            f"[VERIFY-1] sec={sec:4d} "
                            f"bucket(gen)={len(bucket):3d} target={target_fps:3d} sent={sent_cnt:3d}"
                        )

                    while True:
                        # 允许外部请求停止
                        if stop_send.is_set():
                            break

                        # 避免永久阻塞，给 stop_send 留出检查机会
                        try:
                            frame_bgr, src_ts_ms = gen_queue.get(timeout=0.2)
                        except queue.Empty:
                            continue

                        src_ts_ms = int(src_ts_ms)
                        src_sec = int(src_ts_ms // 1000)

                        if last_src_sec is None:
                            last_src_sec = src_sec

                        # 跨秒：先 flush 上一秒 bucket
                        if src_sec != last_src_sec:
                            _flush_bucket(last_src_sec, cur_bucket)
                            cur_bucket = []
                            last_src_sec = src_sec

                        # 收集当前秒的帧
                        cur_bucket.append((frame_bgr, src_ts_ms))

                    # ---- 收到 stop 信号后，flush 最后一个 bucket（如果存在）----
                    if last_src_sec is not None and len(cur_bucket) > 0:
                        _flush_bucket(last_src_sec, cur_bucket)

                send_thread = threading.Thread(target=send_loop, daemon=True)
                send_thread.start()


            print("Inferring using:", audio_path)
            avatar.inference(
                audio_path,
                audio_num,
                args.fps,
                args.skip_save_images,
                on_frame=on_frame_cb
            )

            print("[webrtc] inference done. Keep server alive for playback observation.")
            print("[webrtc] Press Enter to STOP and exit.")
            input()  # <-- 推理后继续保持，给浏览器时间播放

            if streamer is not None:
                stop_send.set()
                try:
                    send_thread.join(timeout=2.0)
                except Exception:
                    pass
                streamer.stop()



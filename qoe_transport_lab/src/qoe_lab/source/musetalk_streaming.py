# qoe_transport_lab/src/qoe_lab/source/musetalk_streaming.py
from __future__ import annotations

import os
import glob
import pickle
import math
from dataclasses import dataclass
from typing import Optional, Any, List, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import WhisperModel

from qoe_lab.source.base import FrameSource, GeneratedFrame


def _read_imgs_cv2(paths: list[str]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cv2.imread 失败：{p}")
        out.append(img)
    return out


@dataclass
class MusetalkModelConfig:
    musetalk_root: str = "/home/pb/web-robot/MuseTalk"

    version: str = "v15"
    gpu_id: int = 2
    vae_type: str = "sd-vae"
    unet_config: str = "models/musetalk/musetalk.json"
    unet_model_path: str = "models/musetalk/pytorch_model.bin"
    whisper_dir: str = "models/whisper"

    audio_padding_length_left: int = 2
    audio_padding_length_right: int = 2
    batch_size: int = 20

    # 稳定优先：float32
    use_fp16: bool = False

    # size_bytes 估计
    jpeg_quality: Optional[int] = None
    constant_size_bytes: int = 25_000


class MusetalkStreamingFrameSource(FrameSource):
    """
    逐秒真实生成：
    - 一次性计算 whisper_chunks（基准 base_fps）
    - 每秒根据目标 fps 选择该秒内的若干 chunk，执行 UNet->VAE->blend，输出该秒帧
    """

    def __init__(
        self,
        cfg: MusetalkModelConfig,
        *,
        inference_yaml: str,
        avatar_id: str,
        audio_key: str,
        base_fps: int = 30,
        limit_seconds: Optional[int] = None,
    ):
        self.cfg = cfg
        self.inference_yaml = inference_yaml
        self.avatar_id = avatar_id
        self.audio_key = audio_key
        self.base_fps = int(base_fps)
        self.limit_seconds = limit_seconds

        if self.base_fps <= 0:
            raise ValueError("base_fps 必须为正")

        self._audio_path = self._resolve_audio_path()

        # runtime objs
        self._device: Optional[torch.device] = None
        self._vae = None
        self._unet = None
        self._pe = None
        self._whisper = None
        self._audio_processor = None
        self._timesteps = None
        self._get_image_blending = None

        # avatar cache
        self._latents_cycle = None
        self._coords_cycle = None
        self._frames_cycle = None
        self._mask_cycle = None
        self._mask_coords_cycle = None

        # audio chunks（base_fps 对齐）
        self._whisper_chunks = None
        self._total_base_frames = 0

        self._load_avatar_cache()

    # FrameSource
    def start(self) -> None:
        self._init_models_and_utils()
        self._prepare_audio_chunks()

    def close(self) -> None:
        self._vae = None
        self._unet = None
        self._pe = None
        self._whisper = None
        self._audio_processor = None
        self._get_image_blending = None
        self._whisper_chunks = None

    def generate_second(self, second_idx: int, fps: int) -> list[GeneratedFrame]:
        if fps <= 0:
            return []
        if self._whisper_chunks is None:
            raise RuntimeError("FrameSource 未 start()")

        # 该秒在 base_fps 下对应的 chunk 范围
        start_i = second_idx * self.base_fps
        end_i = start_i + self.base_fps  # 一秒包含 base_fps 个 slot

        if start_i >= self._total_base_frames:
            return []

        end_i = min(end_i, self._total_base_frames)
        slots = list(range(start_i, end_i))
        want = min(int(fps), len(slots))
        chosen = self._uniform_sample(slots, want)

        # 生成 chosen 对应的帧（UNet 次数 = want）
        frames_bgr = self._infer_frames_by_indices(chosen)

        out: list[GeneratedFrame] = []
        # 将 chosen 映射回该秒内的 ts_ms：均匀分布在该秒内
        # 注意：ts_ms 是播放时间戳；这里用 fps 作为播放节奏更合理
        start_ms = second_idx * 1000
        interval_ms = max(1, int(round(1000 / fps)))

        for k, frame_bgr in enumerate(frames_bgr):
            ts_ms = start_ms + k * interval_ms
            size_bytes = self._estimate_size_bytes(frame_bgr)
            out.append(GeneratedFrame(ts_ms=ts_ms, size_bytes=size_bytes, payload=frame_bgr))

        return out

    # paths/config
    def _abspath(self, rel: str) -> str:
        return os.path.join(self.cfg.musetalk_root, rel)

    def _resolve_audio_path(self) -> str:
        conf = OmegaConf.load(self.inference_yaml)
        if self.avatar_id not in conf:
            raise KeyError(f"inference_yaml 中找不到 avatar_id={self.avatar_id}")
        audio_clips = conf[self.avatar_id]["audio_clips"]
        if self.audio_key not in audio_clips:
            raise KeyError(f"inference_yaml 中找不到 audio_key={self.audio_key}，已有键：{list(audio_clips.keys())}")
        audio_rel = str(audio_clips[self.audio_key])
        audio_abs = self._abspath(audio_rel) if not os.path.isabs(audio_rel) else audio_rel
        if not os.path.exists(audio_abs):
            raise FileNotFoundError(f"audio 文件不存在：{audio_abs}")
        return audio_abs

    def _avatar_base_path(self) -> str:
        if self.cfg.version == "v15":
            return os.path.join(self.cfg.musetalk_root, "results", "v15", "avatars", self.avatar_id)
        else:
            return os.path.join(self.cfg.musetalk_root, "results", "avatars", self.avatar_id)

    def _load_avatar_cache(self) -> None:
        base = self._avatar_base_path()
        latents_path = os.path.join(base, "latents.pt")
        coords_path = os.path.join(base, "coords.pkl")
        mask_coords_path = os.path.join(base, "mask_coords.pkl")
        full_imgs_path = os.path.join(base, "full_imgs")
        mask_path = os.path.join(base, "mask")

        need = [latents_path, coords_path, mask_coords_path, full_imgs_path, mask_path]
        for p in need:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"缺少 avatar 预处理缓存：{p}\n"
                    f"请先在 MuseTalk 里把 preparation=1 跑一次，生成 results/.../avatars/{self.avatar_id}/ 下的缓存。"
                )

        self._latents_cycle = torch.load(latents_path, map_location="cpu")
        with open(coords_path, "rb") as f:
            self._coords_cycle = pickle.load(f)
        with open(mask_coords_path, "rb") as f:
            self._mask_coords_cycle = pickle.load(f)

        img_list = sorted(glob.glob(os.path.join(full_imgs_path, "*.*")))
        mask_list = sorted(glob.glob(os.path.join(mask_path, "*.*")))
        if not img_list or not mask_list:
            raise RuntimeError("full_imgs 或 mask 目录为空，avatar 预处理可能未完成。")

        self._frames_cycle = _read_imgs_cv2(img_list)
        self._mask_cycle = _read_imgs_cv2(mask_list)

    # delayed import and model init
    def _init_models_and_utils(self) -> None:
        old_cwd = os.getcwd()
        os.chdir(self.cfg.musetalk_root)
        try:
            from musetalk.utils.utils import datagen, load_all_model  # noqa
            from musetalk.utils.audio_processor import AudioProcessor  # noqa
            from musetalk.utils.blending import get_image_blending  # noqa

            self._datagen = datagen
            self._load_all_model = load_all_model
            self._audio_processor_cls = AudioProcessor
            self._get_image_blending = get_image_blending

            self._device = torch.device(f"cuda:{self.cfg.gpu_id}" if torch.cuda.is_available() else "cpu")

            vae, unet, pe = self._load_all_model(
                unet_model_path=self._abspath(self.cfg.unet_model_path),
                vae_type=self.cfg.vae_type,
                unet_config=self._abspath(self.cfg.unet_config),
                device=self._device,
            )
            self._timesteps = torch.tensor([0], device=self._device)

            # dtype
            if self.cfg.use_fp16:
                dtype = torch.float16
            else:
                dtype = torch.float32

            pe = pe.to(self._device, dtype=dtype)
            vae.vae = vae.vae.to(self._device, dtype=dtype)
            unet.model = unet.model.to(self._device, dtype=dtype)

            self._vae = vae
            self._unet = unet
            self._pe = pe

            self._audio_processor = self._audio_processor_cls(feature_extractor_path=self._abspath(self.cfg.whisper_dir))

            whisper = WhisperModel.from_pretrained(self._abspath(self.cfg.whisper_dir))
            whisper = whisper.to(device=self._device, dtype=dtype).eval()
            whisper.requires_grad_(False)
            self._whisper = whisper
        finally:
            os.chdir(old_cwd)

    @torch.no_grad()
    def _prepare_audio_chunks(self) -> None:
        assert self._audio_processor is not None
        assert self._device is not None and self._whisper is not None and self._pe is not None

        # whisper chunks（base_fps 对齐）
        # 这里沿用 MuseTalk AudioProcessor 的接口
        # 为保证 dtype 对齐，weight_dtype 统一用 unet 参数 dtype
        unet_dtype = next(self._unet.model.parameters()).dtype

        whisper_input_features, librosa_length = self._audio_processor.get_audio_feature(
            self._audio_path,
            weight_dtype=unet_dtype,
        )
        whisper_chunks = self._audio_processor.get_whisper_chunk(
            whisper_input_features,
            self._device,
            unet_dtype,
            self._whisper,
            librosa_length,
            fps=self.base_fps,
            audio_padding_length_left=self.cfg.audio_padding_length_left,
            audio_padding_length_right=self.cfg.audio_padding_length_right,
        )

        total = len(whisper_chunks)
        if self.limit_seconds is not None:
            total = min(total, int(self.limit_seconds * self.base_fps))
            whisper_chunks = whisper_chunks[:total]

        self._whisper_chunks = whisper_chunks
        self._total_base_frames = total

    @torch.no_grad()
    def _infer_frames_by_indices(self, indices: list[int]) -> list[np.ndarray]:
        """
        indices 是 base_fps 对齐的全局索引（0..total_base_frames-1）
        返回相同顺序的 BGR 帧列表。
        """
        assert self._device is not None
        assert self._whisper_chunks is not None
        assert self._vae is not None and self._unet is not None and self._pe is not None
        assert self._timesteps is not None

        # 按 batch_size 分批
        out_frames: list[np.ndarray] = []
        bs = int(self.cfg.batch_size)
        unet_dtype = next(self._unet.model.parameters()).dtype

        for st in range(0, len(indices), bs):
            batch_idx = indices[st:st + bs]

            # whisper_batch shape: [B, ...] 取 chunks 并 stack
            whisper_batch = torch.stack([self._whisper_chunks[i] for i in batch_idx], dim=0).to(self._device, dtype=unet_dtype)
            audio_feature_batch = self._pe(whisper_batch).to(dtype=unet_dtype)

            # latent：用 idx 作为 cycle 索引
            # 每个 latent 可能是 [1, C, H, W]，需要变成 [C, H, W] 再 stack -> [B, C, H, W]
            latents_list = []
            for i in batch_idx:
                lt = self._latents_cycle[i % len(self._latents_cycle)]
                if isinstance(lt, np.ndarray):
                    lt = torch.from_numpy(lt)
                # 统一到 torch.Tensor
                if not torch.is_tensor(lt):
                    lt = torch.tensor(lt)

                # 关键：去掉多余的 batch 维度
                # 常见情况：lt.shape == [1, C, H, W]
                if lt.ndim == 4 and lt.shape[0] == 1:
                    lt = lt[0]
                # 保险：若仍是 4D 且第 0 维不是 1，说明缓存格式不同；直接保持
                # 若是 3D 则正好是 [C,H,W]

                latents_list.append(lt)

            latent_batch = torch.stack(latents_list, dim=0).to(self._device, dtype=unet_dtype)
            if latent_batch.ndim != 4:
                raise RuntimeError(f"latent_batch 维度异常：shape={tuple(latent_batch.shape)}，期望 4D [B,C,H,W]")



            pred_latents = self._unet.model(
                latent_batch,
                self._timesteps,
                encoder_hidden_states=audio_feature_batch
            ).sample

            pred_latents = pred_latents.to(device=self._device, dtype=next(self._vae.vae.parameters()).dtype)
            recon = self._vae.decode_latents(pred_latents)

            # recon 返回 iterable（B 张嘴部图），逐个 blend
            for k, res_frame in enumerate(recon):
                global_i = batch_idx[k]
                combine = self._blend_one_frame(res_frame, global_i)
                if combine is None:
                    combine = np.zeros((256, 256, 3), dtype=np.uint8)
                out_frames.append(combine)

        return out_frames

    def _blend_one_frame(self, res_frame: np.ndarray, global_i: int) -> Optional[np.ndarray]:
        bbox = self._coords_cycle[global_i % len(self._coords_cycle)]
        ori_frame = self._frames_cycle[global_i % len(self._frames_cycle)].copy()

        x1, y1, x2, y2 = bbox
        try:
            mouth = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        except Exception:
            return None

        mask = self._mask_cycle[global_i % len(self._mask_cycle)]
        mask_crop_box = self._mask_coords_cycle[global_i % len(self._mask_coords_cycle)]
        return self._get_image_blending(ori_frame, mouth, bbox, mask, mask_crop_box)

    def _estimate_size_bytes(self, frame_bgr: np.ndarray) -> int:
        if self.cfg.jpeg_quality is None:
            return int(self.cfg.constant_size_bytes)
        q = max(1, min(100, int(self.cfg.jpeg_quality)))
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return int(self.cfg.constant_size_bytes)
        return int(len(buf))

    @staticmethod
    def _uniform_sample(indices: list[int], k: int) -> list[int]:
        if k <= 0:
            return []
        if k >= len(indices):
            return indices
        out = []
        n = len(indices)
        for j in range(k):
            pos = int(round(j * (n - 1) / (k - 1))) if k > 1 else 0
            out.append(indices[pos])
        # 去重
        out2 = []
        seen = set()
        for x in out:
            if x not in seen:
                out2.append(x)
                seen.add(x)
        return out2

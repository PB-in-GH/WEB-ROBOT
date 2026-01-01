import argparse
import asyncio
from aiohttp import ClientSession
from loguru import logger

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRecorder


async def run(url: str, out_file: str, run_seconds: int):
    pc = RTCPeerConnection()
    recorder = MediaRecorder(out_file)

    @pc.on("track")
    def on_track(track):
        logger.info(f"track received: kind={track.kind}")
        recorder.addTrack(track)

    # 明确声明要接收音视频
    pc.addTransceiver("video", direction="recvonly")
    pc.addTransceiver("audio", direction="recvonly")

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    async with ClientSession() as session:
        async with session.post(url, json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}) as resp:
            ans = await resp.json()

    await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))

    await recorder.start()
    logger.info(f"recording to {out_file}, running {run_seconds}s...")

    # 每秒打印一次 stats（后续用于 Network Score / buffer 估计）
    async def stats_loop():
        for i in range(run_seconds):
            stats = await pc.getStats()
            # 只打印关键信息：inbound-rtp 的 jitter、packetsLost、framesDecoded 等
            inbound = [s for s in stats.values() if s.type == "inbound-rtp"]
            # inbound 里通常会有 audio 和 video 两条
            for s in inbound:
                kind = getattr(s, "kind", None) or getattr(s, "mediaType", None)
                jitter = getattr(s, "jitter", None)
                packetsLost = getattr(s, "packetsLost", None)
                packetsReceived = getattr(s, "packetsReceived", None)
                framesDecoded = getattr(s, "framesDecoded", None)
                bytesReceived = getattr(s, "bytesReceived", None)
                logger.info(
                    f"[stats][{kind}] jitter={jitter} lost={packetsLost} recv={packetsReceived} "
                    f"framesDecoded={framesDecoded} bytes={bytesReceived}"
                )
            await asyncio.sleep(1)

    stats_task = asyncio.create_task(stats_loop())

    await asyncio.sleep(run_seconds)

    stats_task.cancel()
    await recorder.stop()
    await pc.close()
    logger.info("done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080/offer")
    parser.add_argument("--out", default="recv_record.mp4")
    parser.add_argument("--seconds", type=int, default=25)
    args = parser.parse_args()

    asyncio.run(run(args.url, args.out, args.seconds))


if __name__ == "__main__":
    main()

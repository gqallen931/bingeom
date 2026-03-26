# Copyright 2024 Bingxin Ke, ETH Zurich. All rights reserved.
# 基于Apache License 2.0授权
# 实时深度生成修改版：修复图像尺寸不匹配问题

import argparse
import logging
import os
from pathlib import Path
import time

import numpy as np
import torch
import cv2  # 用于摄像头捕获
from tqdm.auto import tqdm
import einops
from omegaconf import OmegaConf

# 导入RollingDepth相关模块
from rollingdepth import (
    RollingDepthOutput,
    RollingDepthPipeline,
    write_video_from_numpy,
    concatenate_videos_horizontally_torch,
)
from src.util.colorize import colorize_depth_multi_thread
from src.util.config import str2bool


def main():
    # 配置日志输出级别为INFO，显示重要信息
    logging.basicConfig(level=logging.INFO)

    # -------------------- 命令行参数解析 --------------------
    parser = argparse.ArgumentParser(description="使用RollingDepth运行摄像头实时深度估计。")

    # 摄像头参数
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="摄像头设备ID，默认0（通常为内置摄像头）"
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=15,
        help="摄像头捕获帧率，默认15 FPS（根据硬件性能调整）"
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=540,
        help="显示窗口宽度，默认540"
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=540,
        help="显示窗口高度，默认540"
    )

    # 输出参数
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./realtime_output",
        help="保存结果的目录路径，默认：./realtime_output"
    )
    parser.add_argument(
        "--save-video",
        type=str2bool,
        nargs="?",
        default=False,
        help="是否保存输出视频，默认：False"
    )

    # 推理参数
    parser.add_argument(
        "-p",
        "--preset",
        type=str,
        choices=["fast", "fast1024", "full", "paper", "none", "ultrafast"],
        default="ultrafast",
        help="推理预设配置，实时推荐'ultrafast'"
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=r"D:\AnacondaProjects\BinGeo-LDM++\checkpoint\models--prs-eth--rollingdepth-v1-0\snapshots\45c3a33d1e0c0b60493fb7028d2324b5d7556460",
        help="模型检查点路径或Hugging Face标识符"
    )
    parser.add_argument(
        "--res",
        "--processing-resolution",
        type=int,
        default=256,  # 超快速模式使用更低分辨率
        help="处理分辨率，默认256（降低可提升速度）"
    )
    parser.add_argument(
        "--max-vae-bs",
        type=int,
        default=1,
        help="VAE最大批量大小，默认1"
    )
    parser.add_argument(
        "--refine-step",
        type=int,
        default=0,
        help="优化步骤数，默认0"
    )
    parser.add_argument(
        "--unload-snippet",
        type=str2bool,
        default=True,
        help="是否卸载片段以节省内存，默认True"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "fp32"],
        default="fp16",
        help="数据类型，默认fp16"
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="Spectral_r",
        help="深度可视化颜色映射，默认Spectral_r"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细日志"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，用于结果重现"
    )

    # -------------------- 配置参数 --------------------
    input_args = parser.parse_args()

    # 基础参数设置
    args = OmegaConf.create({
        "res": 256,
        "snippet_lengths": [2],
        "cap_dilation": True,
        "dtype": input_args.dtype,
        "refine_snippet_len": 2,
        "refine_start_dilation": 1,
        "dilations": [1],  # 仅使用相邻帧
        "refine_step": 0,
        "max_vae_bs": 1,
        "unload_snippet": True,
        "color_maps": [input_args.cmap],
        "save_sbs": True,
        "restore_res": False,
        "save_npy": False,
        "save_snippets": False,
        "output_fps": input_args.camera_fps,
        "resample_method": "BILINEAR",
        "verbose": input_args.verbose,
        "seed": input_args.seed,
        "checkpoint": input_args.checkpoint,
        "camera_id": input_args.camera_id,
        "camera_fps": input_args.camera_fps,
        "display_width": input_args.display_width,
        "display_height": input_args.display_height,
        "output_dir": input_args.output_dir,
        "save_video": input_args.save_video
    })

    # 应用预设配置
    if "none" != input_args.preset:
        logging.info(f"使用预设配置: {input_args.preset}")
        preset_args_dict = {
            "fast": OmegaConf.create({"dilations": [1, 5], "refine_step": 0}),
            "fast1024": OmegaConf.create({"res": 1024, "dilations": [1, 5], "refine_step": 0}),
            "full": OmegaConf.create({"res": 1024, "dilations": [1, 5, 10], "refine_step": 0}),
            "paper": OmegaConf.create(
                {"dilations": [1, 5, 10], "cap_dilation": False, "dtype": "fp32", "refine_step": 0}),
            "ultrafast": OmegaConf.create({"res": 256, "dilations": [1], "refine_step": 0})  # 超快速模式
        }
        args.update(preset_args_dict[input_args.preset])

    # 创建输出目录
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # -------------------- 设备配置 --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("使用GPU加速推理")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA不可用，使用CPU（速度会很慢）")
    logging.info(f"使用设备: {device}")

    # -------------------- 模型初始化 --------------------
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    logging.info(f"加载模型: {args.checkpoint}")
    pipe: RollingDepthPipeline = RollingDepthPipeline.from_pretrained(
        args.checkpoint, torch_dtype=dtype
    )

    # 启用内存优化
    try:
        pipe.enable_xformers_memory_efficient_attention()
        logging.info("已启用xformers内存优化")
    except ImportError:
        logging.warning("未安装xformers，尝试启用flash attention...")
        try:
            pipe.enable_flash_attention_2()
            logging.info("已启用flash attention")
        except Exception:
            logging.warning("无法启用注意力优化，性能可能受限")

    pipe = pipe.to(device)

    # -------------------- 摄像头初始化与测试 --------------------
    logging.info(f"打开摄像头 (ID: {args.camera_id})")
    cap = cv2.VideoCapture(args.camera_id, cv2.CAP_DSHOW)  # 使用DSHOW后端提高兼容性

    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.display_height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)

    # 验证摄像头是否正常工作并获取实际尺寸
    ret, test_frame = cap.read()
    if not ret or test_frame is None or test_frame.size == 0:
        logging.error("无法从摄像头获取有效帧！")
        cap.release()
        return

    # 获取实际的摄像头尺寸
    actual_height, actual_width = test_frame.shape[:2]
    logging.info(f"摄像头实际分辨率: {actual_width}x{actual_height}")
    logging.info(f"目标显示分辨率: {args.display_width}x{args.display_height}")

    # 如果实际尺寸与目标尺寸不符，更新目标尺寸
    if actual_width != args.display_width or actual_height != args.display_height:
        logging.warning(f"摄像头不支持目标分辨率，将使用实际分辨率: {actual_width}x{actual_height}")
        args.display_width = actual_width
        args.display_height = actual_height

    # 初始化视频写入器
    video_writer = None
    if args.save_video:
        combined_width = args.display_width * 2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = output_dir / "realtime_depth.mp4"
        video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            args.camera_fps,
            (combined_width, args.display_height)
        )
        logging.info(f"视频将保存到: {video_path}")

    # -------------------- 实时处理循环 --------------------
    logging.info("开始实时深度估计（按 'q' 键退出）")
    frame_buffer = []

    # 计算缓冲区大小
    min_snippet_length = min(args.snippet_lengths)
    max_dilation = max(args.dilations) if args.dilations else 1
    required_buffer_size = min_snippet_length * max_dilation * 2
    logging.info(f"缓冲区初始化中，需要至少 {required_buffer_size} 帧...")

    # 预填充缓冲区
    preload_progress = tqdm(range(required_buffer_size), desc="初始化帧缓冲区")
    for _ in preload_progress:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            logging.error("无法获取摄像头帧，检查摄像头连接")
            cap.release()
            return

        # 调整帧大小以匹配显示分辨率
        frame = cv2.resize(frame, (args.display_width, args.display_height))

        # 处理帧
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).float()
        frame_tensor = einops.rearrange(frame_tensor, "h w c -> 1 c h w")
        frame_tensor = (frame_tensor / 255.0) * 2.0 - 1.0
        frame_buffer.append(frame_tensor)

    logging.info("帧缓冲区初始化完成，开始处理...")
    fps_counter = []

    # 初始化显示窗口
    cv2.namedWindow("实时深度估计", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("实时深度估计", args.display_width * 2, args.display_height)

    try:
        while True:
            start_time = time.time()

            # 读取新帧
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                logging.warning("无法获取摄像头帧，退出")
                break

            # 确保帧大小与显示分辨率一致
            original_frame = cv2.resize(frame, (args.display_width, args.display_height))

            # 预处理新帧
            frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).float()
            frame_tensor = einops.rearrange(frame_tensor, "h w c -> 1 c h w")
            frame_tensor = (frame_tensor / 255.0) * 2.0 - 1.0

            # 更新缓冲区
            frame_buffer.append(frame_tensor)
            if len(frame_buffer) > required_buffer_size:
                frame_buffer.pop(0)

            # 准备输入
            input_frames = torch.cat(frame_buffer, dim=0)
            input_frames = einops.rearrange(input_frames, "n c h w -> 1 n c h w")
            input_frames = input_frames.to(device, dtype=dtype)

            # 深度估计
            depth_pred = None
            try:
                with torch.no_grad():
                    pipe_out: RollingDepthOutput = pipe.forward(
                        input_frames=input_frames,
                        dilations=list(args.dilations),
                        cap_dilation=args.cap_dilation,
                        snippet_lengths=list(args.snippet_lengths),
                        init_infer_steps=[1],
                        strides=[1],
                        coalign_kwargs=None,
                        refine_step=args.refine_step,
                        refine_snippet_len=args.refine_snippet_len,
                        refine_start_dilation=args.refine_start_dilation,
                        generator=None,
                        verbose=args.verbose,
                        max_vae_bs=args.max_vae_bs,
                        unload_snippet=args.unload_snippet,
                    )
                depth_pred = pipe_out.depth_pred[-1:]  # 取最后一帧
            except Exception as e:
                logging.error(f"深度估计出错: {str(e)}")
                # 创建错误提示帧
                error_frame = np.zeros((args.display_height, args.display_width, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"处理错误: {str(e)[:30]}...",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("实时深度估计", error_frame)
                cv2.waitKey(1)
                continue

            # 处理深度图
            if depth_pred is not None:
                colored_np = colorize_depth_multi_thread(
                    depth=depth_pred.cpu().numpy(),
                    valid_mask=None,
                    chunk_size=1,
                    num_threads=1,
                    color_map=args.color_maps[0],
                    verbose=False
                )[0]
                # 确保深度图尺寸与原始图像完全一致
                colored_bgr = cv2.cvtColor(colored_np, cv2.COLOR_RGB2BGR)
                colored_bgr = cv2.resize(colored_bgr, (args.display_width, args.display_height),
                                         interpolation=cv2.INTER_LINEAR)
            else:
                # 创建默认的黑色深度图
                colored_bgr = np.zeros((args.display_height, args.display_width, 3), dtype=np.uint8)
                cv2.putText(colored_bgr, "深度估计失败",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 验证尺寸匹配
            if original_frame.shape[0] != colored_bgr.shape[0] or original_frame.shape[1] != colored_bgr.shape[1]:
                logging.warning(
                    f"尺寸不匹配 - 原始图像: {original_frame.shape[:2]}, 深度图: {colored_bgr.shape[:2]}，强制调整...")
                # 强制调整深度图尺寸以匹配原始图像
                colored_bgr = cv2.resize(colored_bgr, (original_frame.shape[1], original_frame.shape[0]))

            # 组合并显示
            combined = np.hstack((original_frame, colored_bgr))

            # 添加FPS信息
            current_fps = 0
            if len(fps_counter) > 0:
                current_fps = sum(fps_counter) / len(fps_counter)
            cv2.putText(combined, f"FPS: {current_fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("实时深度估计", combined)

            # 保存视频
            if video_writer is not None:
                video_writer.write(combined)

            # 计算帧率
            elapsed = time.time() - start_time
            fps_counter.append(1.0 / elapsed if elapsed > 0 else 0)
            if len(fps_counter) > 10:
                fps_counter.pop(0)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info("用户按下 'q' 键，退出程序")
                break

    except KeyboardInterrupt:
        logging.info("用户中断程序")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        logging.info("程序已退出，资源已释放")


if __name__ == "__main__":
    main()

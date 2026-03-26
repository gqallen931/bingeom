# Copyright 2024 Bingxin Ke, ETH Zurich. All rights reserved.
# Last modified: 2024-12-09
#
# 基于Apache License 2.0授权
# 您可以在http://www.apache.org/licenses/LICENSE-2.0获取许可证完整文本
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 请查看许可证了解权限和限制的具体规定
# ---------------------------------------------------------------------------------
# 如果您觉得此代码有用，请在您的工作中引用我们的论文
# 参考文献格式请见: https://github.com/prs-eth/RollingDepth#-citation
# 有关该方法的更多信息请访问: https://rollingdepth.github.io
# ---------------------------------------------------------------------------------

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
import einops
from omegaconf import OmegaConf

# 导入RollingDepth相关模块
from rollingdepth import (
    RollingDepthOutput,
    RollingDepthPipeline,
    write_video_from_numpy,
    get_video_fps,
    concatenate_videos_horizontally_torch,
)
from src.util.colorize import colorize_depth_multi_thread
from src.util.config import str2bool

if "__main__" == __name__:
    # 配置日志输出级别为INFO，将显示重要信息
    logging.basicConfig(level=logging.INFO)

    # -------------------- 命令行参数解析 --------------------
    parser = argparse.ArgumentParser(
        description="使用RollingDepth运行视频深度估计。"
    )

    # 输入视频参数
    parser.add_argument(
        "-i",
        "--input-video",
        type=str,
        #required=True,
        default=r"D:\AnacondaProjects\BinGeo-LDM++\input\1",
        help=(
            "待处理的输入视频路径。支持："
            "- 单个视频文件路径（例如：'video.mp4'）"
            "- 包含视频路径列表的文本文件（每行一个路径）"
            "- 包含视频文件的目录路径"
            "为必填参数。"
        ),
        dest="input_video",
    )

    # 输出目录参数
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        #required=True,
        default=r"D:\AnacondaProjects\BinGeo-LDM++\output\1-1",
        help=(
            "处理结果的保存目录路径。"
            "如果目录不存在将自动创建。"
            "为必填参数。"
        ),
        dest="output_dir",
    )

    # 推理预设参数
    parser.add_argument(
        "-p",
        "--preset",
        type=str,
        choices=["fast", "fast1024", "full", "paper", "none"],
        default="fast",
        help="推理预设配置。",
    )

    # 起始帧参数
    parser.add_argument(
        "--start-frame",
        "--from",
        type=int,
        default=0,
        help=(
            "指定开始处理的帧索引。"
            "使用0表示从视频开头开始。"
            "默认值：0"
        ),
        dest="start_frame",
    )

    # 处理帧数参数
    parser.add_argument(
        "--frame-count",
        "--frames",
        type=int,
        default=0,
        help=(
            "起始帧之后要处理的帧数。"
            "设置为0表示处理到视频结束。"
            "默认值：0（处理所有帧）"
        ),
        dest="frame_count",
    )

    # 模型检查点参数
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=r"D:/AnacondaProjects/BinGeo-LDM++/checkpoint/models--prs-eth--rollingdepth-v1-0/snapshots/45c3a33d1e0c0b60493fb7028d2324b5d7556460",
        help=(
            "用于推理的模型检查点路径。可以是："
            "- 检查点文件的本地路径"
            "- Hugging Face模型库标识符（例如：'prs-eth/rollingdepth-v1-0'）"
            "默认值：'prs-eth/rollingdepth-v1-0'"
        ),
        dest="checkpoint",
    )

    # 处理分辨率参数
    parser.add_argument(
        "--res",
        "--processing-resolution",
        type=int,
        default=None,
        help=(
            "指定图像处理的最大分辨率（像素）。"
            "如果设置为None，使用预设配置值。"
            "如果设置为0，使用输入图像的原始分辨率处理。"
            "默认值：None"
        ),
        dest="res",
    )

    # VAE批量大小参数
    parser.add_argument(
        "--max-vae-bs",
        type=int,
        default=4,
        help=(
            "变分自编码器(VAE)处理的最大批量大小。"
            "值越高内存占用越大，但可能提高处理速度。"
            "如果遇到内存不足错误，请减小此值。"
            "默认值：4"
        ),
    )

    # 输出设置
    parser.add_argument(
        "--fps",
        "--output-fps",
        type=float,
        default=0,
        help=(
            "输出视频的帧率(FPS)。"
            "设置为0以匹配输入视频的帧率。"
            "默认值：0"
        ),
        dest="output_fps",
    )

    # 恢复分辨率参数
    parser.add_argument(
        "--restore-resolution",
        "--restore-res",
        type=str2bool,
        nargs="?",
        default=False,
        help=(
            "处理后是否将输出恢复到输入的原始分辨率。"
            "仅当处理期间输入已被缩放时适用。"
            "默认值：False"
        ),
        dest="restore_res",
    )

    # 保存并排视频参数
    parser.add_argument(
        "--save-sbs" "--save-side-by-side",
        type=str2bool,
        nargs="?",
        default=True,
        help=(
            "是否保存RGB和彩色深度视频并排显示。"
            "如果为True，将使用第一个颜色映射。"
            "默认值：True"
        ),
        dest="save_sbs",
    )

    # 保存npy文件参数
    parser.add_argument(
        "--save-npy",
        type=str2bool,
        nargs="?",
        default=True,
        help=(
            "是否将深度图保存为NumPy(.npy)文件。"
            "便于对原始深度数据进行进一步处理和分析。"
            "默认值：True"
        ),
    )

    # 保存片段参数
    parser.add_argument(
        "--save-snippets",
        type=str2bool,
        nargs="?",
        default=False,
        help=(
            "是否保存初始片段。"
            "有助于调试和质量评估。"
            "默认值：False"
        ),
    )

    # 颜色映射参数
    parser.add_argument(
        "--cmap",
        "--color-maps",
        type=str,
        nargs="+",
        default=["Spectral_r", "Greys_r"],
        help=(
            "用于深度可视化的一个或多个matplotlib颜色映射。"
            "可以指定多个映射以获得不同的可视化风格。"
            "常见选项：'Spectral_r'，'Greys_r'，'viridis'，'magma'。"
            "使用''(空字符串)跳过彩色化。"
            "默认值：['Spectral_r', 'Greys_r']"
        ),
        dest="color_maps",
    )

    # 推理设置
    parser.add_argument(
        "-d",
        "--dilations",
        type=int,
        nargs="+",
        default=None,
        help=(
            "时间分析的帧间距。"
            "设置为None以使用基于视频长度的预设配置。"
            "自定义配置："
            "`1 10 25`：最佳精度，较慢的处理速度"
            "`1 25`：速度和精度的平衡"
            "`1 10`：适用于短视频(<78帧)"
            "默认值：None（根据视频长度自动选择）"
        ),
        dest="dilations",
    )

    # 限制 dilation 参数
    parser.add_argument(
        "--cap-dilation",
        type=str2bool,
        default=None,
        help=(
            "是否为短视频自动减少dilation间距。"
            "设置为None使用预设配置。"
            "启用此选项可防止时间窗口超出视频长度。"
            "默认值：None（根据视频长度自动确定）"
        ),
        dest="cap_dilation",
    )

    # 数据类型参数
    parser.add_argument(
        "--dtype",
        "--data-type",
        type=str,
        choices=["fp16", "fp32", None],
        default=None,
        help=(
            "指定推理操作的浮点精度。"
            "选项：'fp16'(16位)，'fp32'(32位)，或None。"
            "如果为None，使用预设配置值。"
            "较低的精度(fp16)减少内存使用，但可能影响准确性。"
            "默认值：None"
        ),
        dest="dtype",
    )

    # 片段长度参数
    parser.add_argument(
        "--snip-len",
        "--snippet-lengths",
        type=int,
        nargs="+",
        choices=[2, 3, 4],
        default=None,
        help=(
            "每个时间窗口(片段)中要分析的帧数。"
            "设置为None使用预设值(3)。"
            "可以指定多个值对应不同的dilation率。"
            "示例：'--dilations 1 25 --snippet-length 2 3'表示"
            "对于dilation 1使用2帧，对于dilation 25使用3帧。"
            "允许值：2、3或4帧。"
            "默认值：None"
        ),
        dest="snippet_lengths",
    )

    # 优化步骤参数
    parser.add_argument(
        "--refine-step",
        type=int,
        default=None,
        help=(
            "用于提高准确性和细节的优化迭代次数。"
            "保持未设置(None)以使用预设配置。"
            "设置为0禁用优化。"
            "较高的值可能提高准确性，但增加处理时间。"
            "默认值：None"
        ),
        dest="refine_step",
    )

    # 优化片段长度参数
    parser.add_argument(
        "--refine-snippet-len",
        type=int,
        default=None,
        help=(
            "优化阶段使用的文本片段长度。"
            "指定一次处理的句子或段落数。"
            "如果未指定(None)，将使用系统定义的预设值。"
            "默认值：None"
        ),
    )

    # 优化起始dilation参数
    parser.add_argument(
        "--refine-start-dilation",
        type=int,
        default=None,
        help=(
            "从粗到精优化过程的初始dilation因子。"
            "控制优化步骤的起始粒度。"
            "值越高，初始搜索窗口越大。"
            "如果未指定(None)，使用系统默认值。"
            "默认值：None"
        ),
    )

    # 其他设置
    parser.add_argument(
        "--resample-method",
        type=str,
        choices=["BILINEAR", "NEAREST_EXACT", "BICUBIC"],
        default="BILINEAR",
        help="用于调整图像大小的重采样方法。",
    )

    # 卸载片段参数
    parser.add_argument(
        "--unload-snippet",
        type=str2bool,
        default=False,
        help=(
            "通过将处理的数据片段移至CPU来控制内存优化。"
            "启用时，以较慢的处理速度为代价减少GPU内存使用。"
            "适用于GPU内存有限或数据集较大的系统。"
            "默认值：False"
        ),
    )

    # 详细输出参数
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=("处理期间启用详细的进度和信息报告。"),
    )

    # 随机种子参数
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "用于重现性的随机数生成器种子(受计算随机性限制)。"
            "使用相同的种子值将在多次运行中产生相同的结果。"
            "如果未指定(None)，将使用随机种子。"
            "默认值：None"
        ),
    )

    # -------------------- 配置预设参数 --------------------
    input_args = parser.parse_args()

    # 基础参数设置
    args = OmegaConf.create(
        {
            "res": 768,  # 分辨率
            "snippet_lengths": [3],  # 片段长度
            "cap_dilation": True,  # 限制dilation
            "dtype": "fp16",  # 数据类型
            "refine_snippet_len": 3,  # 优化片段长度
            "refine_start_dilation": 6,  # 优化起始dilation
        }
    )

    # 不同预设的参数配置
    preset_args_dict = {
        "fast": OmegaConf.create(
            {
                "dilations": [1, 25],  # 快速模式：dilation值
                "refine_step": 0,  # 快速模式：不进行优化
            }
        ),
        "fast1024": OmegaConf.create(
            {
                "res": 1024,  # fast1024模式：更高分辨率
                "dilations": [1, 25],
                "refine_step": 0,
            }
        ),
        "full": OmegaConf.create(
            {
                "res": 1024,
                "dilations": [1, 10, 25],  # 完整模式：更多dilation值
                "refine_step": 10,  # 完整模式：10次优化
            }
        ),
        "paper": OmegaConf.create(
            {
                "dilations": [1, 10, 25],
                "cap_dilation": False,  # 论文模式：不限制dilation
                "dtype": "fp32",  # 论文模式：使用更高精度
                "refine_step": 10,
            }
        ),
    }

    # 应用预设配置（如果不是"none"）
    if "none" != input_args.preset:
        logging.info(f"使用预设配置: {input_args.preset}")
        args.update(preset_args_dict[input_args.preset])

    # 合并或覆盖参数：命令行参数优先于预设
    for key, value in vars(input_args).items():
        if key in args.keys():
            # 如果值已设置且与预设不同，则覆盖
            if value is not None and value != args[key]:
                logging.warning(f"覆盖参数: {key} = {value}")
                args[key] = value
        else:
            # 添加新参数
            args[key] = value
            # 参数检查
            assert value is not None or key in ["seed"], f"未定义的参数: {key}"

    # 日志输出参数信息
    msg = f"参数配置: {args}"
    if args.verbose:
        logging.info(msg)
    else:
        logging.debug(msg)

    # 参数检查：如果保存并排视频，必须提供颜色映射
    if args.save_sbs:
        assert (
                len(args.color_maps) > 0
        ), "未提供颜色映射，无法保存并排视频。"

    # 处理输入输出路径
    input_video = Path(args.input_video)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录（如果不存在）

    # -------------------- 设备配置 --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")  # 使用GPU
    else:
        device = torch.device("cpu")  # 使用CPU
        logging.warning("CUDA不可用。在CPU上运行会很慢。")
    logging.info(f"使用设备: {device}")

    # -------------------- 数据处理 --------------------
    # 根据输入类型获取视频列表
    if input_video.is_dir():
        # 如果输入是目录，获取目录中所有文件
        input_video_ls = os.listdir(input_video)
        input_video_ls = [input_video.joinpath(v_name) for v_name in input_video_ls]
    elif ".txt" == input_video.suffix:
        # 如果输入是文本文件，读取文件中的视频路径
        with open(input_video, "r") as f:
            input_video_ls = f.readlines()
        input_video_ls = [Path(s.strip()) for s in input_video_ls]
    else:
        # 否则视为单个视频文件
        input_video_ls = [Path(input_video)]
    input_video_ls = sorted(input_video_ls)  # 排序视频列表

    logging.info(f"找到 {len(input_video_ls)} 个视频。")

    # -------------------- 模型初始化 --------------------
    # 设置数据类型
    if "fp16" == args.dtype:
        dtype = torch.float16
    elif "fp32" == args.dtype:
        dtype = torch.float32
    else:
        raise ValueError(f"不支持的数据类型: {args.dtype}")

    # 从预训练模型加载RollingDepth管道
    pipe: RollingDepthPipeline = RollingDepthPipeline.from_pretrained(
        args.checkpoint, torch_dtype=dtype
    )  # type: ignore

    # 尝试启用xformers内存高效注意力机制
    try:
        pipe.enable_xformers_memory_efficient_attention()
        logging.info("已启用xformers")
    except ImportError:
        logging.warning("未使用xformers运行")

    # 将模型移动到指定设备
    pipe = pipe.to(device)

    # -------------------- 推理和结果保存 --------------------
    # 禁用梯度计算以节省内存并加速推理
    with torch.no_grad():
        # 根据详细模式选择是否显示进度条
        if args.verbose:
            video_iterable = tqdm(input_video_ls, desc="处理视频", leave=True)
        else:
            video_iterable = input_video_ls

        # 遍历所有视频进行处理
        for video_path in video_iterable:
            # 随机数生成器（用于可重复性）
            if args.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(args.seed)

            # 预测深度
            pipe_out: RollingDepthOutput = pipe(
                # 输入设置
                input_video_path=video_path,  # 视频路径
                start_frame=args.start_frame,  # 起始帧
                frame_count=args.frame_count,  # 处理帧数
                processing_res=args.res,  # 处理分辨率
                resample_method=args.resample_method,  # 重采样方法

                # 推理设置
                dilations=list(args.dilations),  # dilation值列表
                cap_dilation=args.cap_dilation,  # 是否限制dilation
                snippet_lengths=list(args.snippet_lengths),  # 片段长度
                init_infer_steps=[1],  # 初始推理步骤
                strides=[1],  # 步长
                coalign_kwargs=None,  # 共对齐参数
                refine_step=args.refine_step,  # 优化步骤数
                refine_snippet_len=args.refine_snippet_len,  # 优化片段长度
                refine_start_dilation=args.refine_start_dilation,  # 优化起始dilation

                # 其他设置
                generator=generator,  # 随机数生成器
                verbose=args.verbose,  # 是否详细输出
                max_vae_bs=args.max_vae_bs,  # VAE最大批量大小

                # 输出设置
                restore_res=args.restore_res,  # 是否恢复分辨率
                unload_snippet=args.unload_snippet,  # 是否卸载片段
            )

            # 获取深度预测结果 [N 1 H W]
            depth_pred = pipe_out.depth_pred

            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            # 保存预测结果为npy文件
            if args.save_npy:
                save_to = output_dir.joinpath(f"{video_path.stem}_pred.npy")
                if args.verbose:
                    logging.info(f"保存预测结果到 {save_to}")
                # 保存为 [N H W] 格式
                np.save(save_to, depth_pred.numpy().squeeze(1))

            # 保存中间片段
            if args.save_snippets and pipe_out.snippet_ls is not None:
                save_to = output_dir.joinpath(f"{video_path.stem}_snippets.npz")
                if args.verbose:
                    logging.info(f"保存片段到 {save_to}")
                snippet_dict = {}
                for i_dil, snippets in enumerate(pipe_out.snippet_ls):
                    dilation = args.dilations[i_dil]
                    # 保存为 [n_snip, snippet_len, H W] 格式
                    snippet_dict[f"dilation{dilation}"] = snippets.numpy().squeeze(2)
                np.savez_compressed(save_to, **snippet_dict)

            # 确定输出视频帧率
            if args.output_fps > 0:
                output_fps = args.output_fps
            else:
                output_fps = get_video_fps(video_path)  # 使用输入视频帧率

            # 对每个颜色映射生成彩色深度视频
            for i_cmap, cmap in enumerate(args.color_maps):
                if "" == cmap:  # 跳过空颜色映射
                    continue

                # 多线程彩色化深度图
                colored_np = colorize_depth_multi_thread(
                    depth=depth_pred.numpy(),
                    valid_mask=None,
                    chunk_size=4,
                    num_threads=4,
                    color_map=cmap,
                    verbose=args.verbose,
                )  # 输出格式: [n h w 3], 取值范围 [0, 255]

                # 保存彩色深度视频
                save_to = output_dir.joinpath(f"{video_path.stem}_{cmap}.mp4")
                write_video_from_numpy(
                    frames=colored_np,
                    output_path=save_to,
                    fps=output_fps,
                    crf=23,
                    preset="medium",
                    verbose=args.verbose,
                )

                # 保存RGB和深度并排视频（仅对第一个颜色映射）
                if args.save_sbs and 0 == i_cmap:
                    # 处理RGB图像（转换为0-255范围）
                    rgb = pipe_out.input_rgb * 255  # [N 3 H W]
                    # 调整彩色深度图的维度
                    colored_depth = einops.rearrange(
                        torch.from_numpy(colored_np), "n h w c -> n c h w"
                    )
                    # 水平拼接RGB和深度图，中间留10像素间隙
                    concat_video = (
                        concatenate_videos_horizontally_torch(
                            rgb, colored_depth, gap=10
                        )
                        .int()
                        .numpy()
                        .astype(np.uint8)
                    )
                    # 调整维度以适应视频写入
                    concat_video = einops.rearrange(concat_video, "n c h w -> n h w c")
                    # 保存并排视频
                    save_to = output_dir.joinpath(f"{video_path.stem}_rgbd.mp4")
                    write_video_from_numpy(
                        frames=concat_video,
                        output_path=save_to,
                        fps=output_fps,
                        crf=23,
                        preset="medium",
                        verbose=args.verbose,
                    )

        # 处理完成
        logging.info(
            f"处理完成。 {len(video_iterable)} 个预测结果已保存到 {output_dir}"
        )

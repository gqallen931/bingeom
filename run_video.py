# Copyright 2024 Bingxin Ke, ETH Zurich. All rights reserved.
# Last modified: 2024-12-09
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/RollingDepth#-citation
# More information about the method can be found at https://rollingdepth.github.io
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
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run video depth estimation using RollingDepth."
    )
    parser.add_argument(
        "-i",
        "--input-video",
        type=str,
        default=r"D:\AnacondaProjects\BinGeo-LDM++\input",
        help=(
            "Path to the input video(s) to be processed. Accepts: "
            "- Single video file path (e.g., 'video.mp4') "
            "- Text file containing a list of video paths (one per line) "
            "- Directory path containing video files "
            "Required argument."
        ),
        dest="input_video",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=r"D:\AnacondaProjects\BinGeo-LDM++\output",
        help=(
            "Directory path where processed outputs will be saved. "
            "Will be created if it doesn't exist. "
            "Required argument."
        ),
        dest="output_dir",
    )
    parser.add_argument(
        "-p",
        "--preset",
        type=str,
        choices=["fast", "fast1024", "full", "paper", "none"],
        default="fast",
        help="Inference presets.",
    )
    parser.add_argument(
        "--start-frame",
        "--from",
        type=int,
        default=0,
        help=(
            "Specifies the starting frame index for processing. "
            "Use 0 to start from the beginning of the video. "
            "Default: 0"
        ),
        dest="start_frame",
    )
    parser.add_argument(
        "--frame-count",
        "--frames",
        type=int,
        default=0,
        help=(
            "Number of frames to process after the starting frame. "
            "Set to 0 to process until the end of the video. "
            "Default: 0 (process all frames)"
        ),
        dest="frame_count",
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="D:/AnacondaProjects/RollingDepth-main/checkpoint/models--prs-eth--rollingdepth-v1-0/snapshots/45c3a33d1e0c0b60493fb7028d2324b5d7556460",
        help=(
            "Path to the model checkpoint to use for inference. Can be either: "
            "- A local path to checkpoint files "
            "- A Hugging Face model hub identifier (e.g., 'prs-eth/rollingdepth-v1-0') "
            "Default: 'prs-eth/rollingdepth-v1-0'"
        ),
        dest="checkpoint",
    )
    parser.add_argument(
        "--res",
        "--processing-resolution",
        type=int,
        default=None,
        help=(
            "Specifies the maximum resolution (in pixels) at which image processing will be performed. "
            "If set to None, uses the preset configuration value. "
            "If set to 0, processes at the original input image resolution. "
            "Default: None"
        ),
        dest="res",
    )
    parser.add_argument(
        "--max-vae-bs",
        type=int,
        default=4,
        help=(
            "Maximum batch size for the Variational Autoencoder (VAE) processing. "
            "Higher values increase memory usage but may improve processing speed. "
            "Reduce this value if encountering out-of-memory errors. "
            "Default: 4"
        ),
    )

    # Output settings
    parser.add_argument(
        "--fps",
        "--output-fps",
        type=float,
        default=0,
        help=(
            "Frame rate (FPS) for the output video. "
            "Set to 0 to match the input video's frame rate. "
            "Default: 0"
        ),
        dest="output_fps",
    )
    parser.add_argument(
        "--restore-resolution",
        "--restore-res",
        type=str2bool,
        nargs="?",
        default=False,
        help=(
            "Whether to restore the output to the original input resolution after processing. "
            "Only applies when input has been resized during processing. "
            "Default: False"
        ),
        dest="restore_res",
    )
    parser.add_argument(
        "--save-sbs" "--save-side-by-side",
        type=str2bool,
        nargs="?",
        default=True,
        help=(
            "Whether to save RGB and colored depth videos side-by-side. "
            "If True, the first color map will be used. "
            "Default: True"
        ),
        dest="save_sbs",
    )
    parser.add_argument(
        "--save-npy",
        type=str2bool,
        nargs="?",
        default=True,
        help=(
            "Whether to save depth maps as NumPy (.npy) files. "
            "Enables further processing and analysis of raw depth data. "
            "Default: True"
        ),
    )
    parser.add_argument(
        "--save-snippets",
        type=str2bool,
        nargs="?",
        default=False,
        help=(
            "Whether to save initial snippets. "
            "Useful for debugging and quality assessment. "
            "Default: False"
        ),
    )
    parser.add_argument(
        "--cmap",
        "--color-maps",
        type=str,
        nargs="+",
        default=["Spectral_r", "Greys_r"],
        help=(
            "One or more matplotlib color maps for depth visualization. "
            "Multiple maps can be specified for different visualization styles. "
            "Common options: 'Spectral_r', 'Greys_r', 'viridis', 'magma'. "
            "Use '' (empty string) to skip colorization. "
            "Default: ['Spectral_r', 'Greys_r']"
        ),
        dest="color_maps",
    )

    # Inference setting
    parser.add_argument(
        "-d",
        "--dilations",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Spacing between frames for temporal analysis. "
            "Set to None to use preset configurations based on video length. "
            "Custom configurations: "
            "`1 10 25`: Best accuracy, slower processing "
            "`1 25`: Balanced speed and accuracy "
            "`1 10`: For short videos (<78 frames) "
            "Default: None (auto-select based on video length)"
        ),
        dest="dilations",
    )
    parser.add_argument(
        "--cap-dilation",
        type=str2bool,
        default=None,
        help=(
            "Whether to automatically reduce dilation spacing for short videos. "
            "Set to None to use preset configuration. "
            "Enabling this prevents temporal windows from extending beyond video length. "
            "Default: None (automatically determined based on video length)"
        ),
        dest="cap_dilation",
    )
    parser.add_argument(
        "--dtype",
        "--data-type",
        type=str,
        choices=["fp16", "fp32", None],
        default=None,
        help=(
            "Specifies the floating-point precision for inference operations. "
            "Options: 'fp16' (16-bit), 'fp32' (32-bit), or None. "
            "If None, uses the preset configuration value. "
            "Lower precision (fp16) reduces memory usage but may affect accuracy. "
            "Default: None"
        ),
        dest="dtype",
    )
    parser.add_argument(
        "--snip-len",
        "--snippet-lengths",
        type=int,
        nargs="+",
        choices=[2, 3, 4],
        default=None,
        help=(
            "Number of frames to analyze in each temporal window (snippet). "
            "Set to None to use preset value (3). "
            "Can specify multiple values corresponding to different dilation rates. "
            "Example: '--dilations 1 25 --snippet-length 2 3' uses "
            "2 frames for dilation 1 and 3 frames for dilation 25. "
            "Allowed values: 2, 3, or 4 frames. "
            "Default: None"
        ),
        dest="snippet_lengths",
    )
    parser.add_argument(
        "--refine-step",
        type=int,
        default=None,
        help=(
            "Number of refinement iterations to improve accuracy and details. "
            "Leave as unset (None) to use preset configuration. "
            "Set to 0 to disable refinement. "
            "Higher values may improve accuracy but increase processing time. "
            "Default: None"
        ),
        dest="refine_step",
    )
    parser.add_argument(
        "--refine-snippet-len",
        type=int,
        default=None,
        help=(
            "Length of text snippets used during the refinement phase. "
            "Specifies the number of sentences or segments to process at once. "
            "If not specified (None), system-defined preset values will be used. "
            "Default: None"
        ),
    )
    parser.add_argument(
        "--refine-start-dilation",
        type=int,
        default=None,
        help=(
            "Initial dilation factor for the coarse-to-fine refinement process. "
            "Controls the starting granularity of the refinement steps. "
            "Higher values result in larger initial search windows. "
            "If not specified (None), uses system default. "
            "Default: None"
        ),
    )

    # Other settings
    parser.add_argument(
        "--resample-method",
        type=str,
        choices=["BILINEAR", "NEAREST_EXACT", "BICUBIC"],
        default="BILINEAR",
        help="Resampling method used to resize images.",
    )
    parser.add_argument(
        "--unload-snippet",
        type=str2bool,
        default=False,
        help=(
            "Controls memory optimization by moving processed data snippets to CPU. "
            "When enabled, reduces GPU memory usage at the cost of slower processing. "
            "Useful for systems with limited GPU memory or large datasets. "
            "Default: False"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=("Enable detailed progress and information reporting during processing. "),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random number generator seed for reproducibility (up to computational randomness). "
            "Using the same seed value will produce identical results across runs. "
            "If not specified (None), a random seed will be used. "
            "Default: None"
        ),
    )

    # -------------------- Config preset arguments --------------------
    input_args = parser.parse_args()

    args = OmegaConf.create(
        {
            "res": 768,
            "snippet_lengths": [3],
            "cap_dilation": True,
            "dtype": "fp16",
            "refine_snippet_len": 3,
            "refine_start_dilation": 6,
        }
    )
    preset_args_dict = {
        "fast": OmegaConf.create(
            {
                "dilations": [1, 25],
                "refine_step": 0,
            }
        ),
        "fast1024": OmegaConf.create(
            {
                "res": 1024,
                "dilations": [1, 25],
                "refine_step": 0,
            }
        ),
        "full": OmegaConf.create(
            {
                "res": 1024,
                "dilations": [1, 10, 25],
                "refine_step": 10,
            }
        ),
        "paper": OmegaConf.create(
            {
                "dilations": [1, 10, 25],
                "cap_dilation": False,
                "dtype": "fp32",
                "refine_step": 10,
            }
        ),
    }
    if "none" != input_args.preset:
        logging.info(f"Using preset: {input_args.preset}")
        args.update(preset_args_dict[input_args.preset])

    # Merge or overwrite arguments
    for key, value in vars(input_args).items():
        if key in args.keys():
            # overwrite if value is set and different from preset
            if value is not None and value != args[key]:
                logging.warning(f"Overwritting argument: {key} = {value}")
                args[key] = value
        else:
            # add argument
            args[key] = value
            # sanity check
            assert value is not None or key in ["seed"], f"Undefined argument: {key}"

    msg = f"arguments: {args}"
    if args.verbose:
        logging.info(msg)
    else:
        logging.debug(msg)

    # Argument check
    if args.save_sbs:
        assert (
            len(args.color_maps) > 0
        ), "No color map is given, can not save side-by-side videos."

    input_video = Path(args.input_video)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    if input_video.is_dir():
        input_video_ls = os.listdir(input_video)
        input_video_ls = [input_video.joinpath(v_name) for v_name in input_video_ls]
    elif ".txt" == input_video.suffix:
        with open(input_video, "r") as f:
            input_video_ls = f.readlines()
        input_video_ls = [Path(s.strip()) for s in input_video_ls]
    else:
        input_video_ls = [Path(input_video)]
    input_video_ls = sorted(input_video_ls)

    logging.info(f"Found {len(input_video_ls)} videos.")

    # -------------------- Model --------------------
    if "fp16" == args.dtype:
        dtype = torch.float16
    elif "fp32" == args.dtype:
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    pipe: RollingDepthPipeline = RollingDepthPipeline.from_pretrained(
        args.checkpoint, torch_dtype=dtype
    )  # type: ignore

    try:
        pipe.enable_xformers_memory_efficient_attention()
        logging.info("xformers enabled")
    except ImportError:
        logging.warning("Run without xformers")

    pipe = pipe.to(device)

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        if args.verbose:
            video_iterable = tqdm(input_video_ls, desc="Processing videos", leave=True)
        else:
            video_iterable = input_video_ls
        for video_path in video_iterable:
            # Random number generator
            if args.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(args.seed)

            # Predict depth
            pipe_out: RollingDepthOutput = pipe(
                # input setting
                input_video_path=video_path,
                start_frame=args.start_frame,
                frame_count=args.frame_count,
                processing_res=args.res,
                resample_method=args.resample_method,
                # infer setting
                dilations=list(args.dilations),
                cap_dilation=args.cap_dilation,
                snippet_lengths=list(args.snippet_lengths),
                init_infer_steps=[1],
                strides=[1],
                coalign_kwargs=None,
                refine_step=args.refine_step,
                refine_snippet_len=args.refine_snippet_len,
                refine_start_dilation=args.refine_start_dilation,
                # other settings
                generator=generator,
                verbose=args.verbose,
                max_vae_bs=args.max_vae_bs,
                # output settings
                restore_res=args.restore_res,
                unload_snippet=args.unload_snippet,
            )

            depth_pred = pipe_out.depth_pred  # [N 1 H W]

            os.makedirs(output_dir, exist_ok=True)

            # Save prediction as npy
            if args.save_npy:
                save_to = output_dir.joinpath(f"{video_path.stem}_pred.npy")
                if args.verbose:
                    logging.info(f"Saving predictions to {save_to}")
                np.save(save_to, depth_pred.numpy().squeeze(1))  # [N H W]

            # Save intermediate snippets
            if args.save_snippets and pipe_out.snippet_ls is not None:
                save_to = output_dir.joinpath(f"{video_path.stem}_snippets.npz")
                if args.verbose:
                    logging.info(f"Saving snippets to {save_to}")
                snippet_dict = {}
                for i_dil, snippets in enumerate(pipe_out.snippet_ls):
                    dilation = args.dilations[i_dil]
                    snippet_dict[f"dilation{dilation}"] = snippets.numpy().squeeze(
                        2
                    )  # [n_snip, snippet_len, H W]
                np.savez_compressed(save_to, **snippet_dict)

            # Colorize results
            if args.output_fps > 0:
                output_fps = args.output_fps
            else:
                output_fps = get_video_fps(video_path)

            for i_cmap, cmap in enumerate(args.color_maps):
                if "" == cmap:
                    continue
                colored_np = colorize_depth_multi_thread(
                    depth=depth_pred.numpy(),
                    valid_mask=None,
                    chunk_size=4,
                    num_threads=4,
                    color_map=cmap,
                    verbose=args.verbose,
                )  # [n h w 3], in [0, 255]
                save_to = output_dir.joinpath(f"{video_path.stem}_{cmap}.mp4")

                write_video_from_numpy(
                    frames=colored_np,
                    output_path=save_to,
                    fps=output_fps,
                    crf=23,
                    preset="medium",
                    verbose=args.verbose,
                )

                # Save side-by-side videos
                if args.save_sbs and 0 == i_cmap:
                    rgb = pipe_out.input_rgb * 255  # [N 3 H W]
                    colored_depth = einops.rearrange(
                        torch.from_numpy(colored_np), "n h w c -> n c h w"
                    )
                    concat_video = (
                        concatenate_videos_horizontally_torch(
                            rgb, colored_depth, gap=10
                        )
                        .int()
                        .numpy()
                        .astype(np.uint8)
                    )
                    concat_video = einops.rearrange(concat_video, "n c h w -> n h w c")
                    save_to = output_dir.joinpath(f"{video_path.stem}_rgbd.mp4")
                    write_video_from_numpy(
                        frames=concat_video,
                        output_path=save_to,
                        fps=output_fps,
                        crf=23,
                        preset="medium",
                        verbose=args.verbose,
                    )

        logging.info(
            f"Finished. {len(video_iterable)} predictions are saved to {output_dir}"
        )

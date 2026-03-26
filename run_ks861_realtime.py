# Copyright 2024 基于Apache License 2.0授权
# KS861相机专用实时深度估计程序

import argparse
import logging
import os
from pathlib import Path
import time
import threading
from queue import Queue, Empty

import numpy as np
import torch
import cv2  # 用于摄像头捕获
from tqdm.auto import tqdm
import einops
from omegaconf import OmegaConf

# 导入项目相关模块
from bingeo_ldm_plus_plus import KS861Pipeline
from rollingdepth import RollingDepthOutput
from src.util.colorize import colorize_depth_multi_thread
from src.util.config import str2bool

# 确保中文显示正常
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class PipelineStage:
    """
    流水线处理阶段基类
    """
    def __init__(self, input_queue=None, output_queue=None):
        self.input_queue = input_queue or Queue(maxsize=10)
        self.output_queue = output_queue or Queue(maxsize=10)
        self.running = False
        self.thread = None
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        return self
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def run(self):
        """子类需要实现此方法"""
        pass

class ImageCaptureStage(PipelineStage):
    """
    图像采集阶段 - 专为KS861单USB复合双目摄像头设计
    KS861摄像头输出的是水平拼接的左右目图像，需要分割成左右两个摄像头的图像
    """
    def __init__(self, camera_id=0, camera_fps=15, display_width=640, display_height=480, **kwargs):
        super().__init__(**kwargs)
        self.camera_id = camera_id
        self.camera_fps = camera_fps
        self.display_width = display_width  # 左右图像总宽度
        self.display_height = display_height
        self.cap = None
    
    def run(self):
        logging.info(f"尝试初始化KS861摄像头，设备ID: {self.camera_id}")
        # 初始化摄像头
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)  # 使用DSHOW后端提高兼容性
        
        if not self.cap.isOpened():
            logging.error(f"无法打开KS861摄像头，设备ID: {self.camera_id}")
            return
        
        logging.info(f"KS861摄像头已打开，设备ID: {self.camera_id}")
        
        # 设置摄像头参数
        # KS861需要设置为1280x720分辨率来获取左右拼接的图像
        width_set = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width * 2)  # 设置为期望宽度的2倍，用于左右分割
        height_set = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
        fps_set = self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        
        logging.info(f"尝试设置摄像头参数: 宽度={self.display_width*2}, 高度={self.display_height}, FPS={self.camera_fps}")
        logging.info(f"参数设置结果: 宽度={width_set}, 高度={height_set}, FPS={fps_set}")
        
        # 获取实际设置的参数
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"实际设置的摄像头参数: 宽度={actual_width}, 高度={actual_height}, FPS={actual_fps}")
        
        # 验证摄像头是否正常工作
        logging.info("尝试读取测试帧...")
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None or test_frame.size == 0:
            logging.error(f"无法从KS861摄像头获取有效帧！读取结果: ret={ret}, 帧数据={test_frame is not None}")
            if self.cap:
                self.cap.release()
            return
        
        logging.info("成功获取测试帧！")
        
        # 获取实际的摄像头尺寸
        frame_height, frame_width = test_frame.shape[:2]
        logging.info(f"KS861摄像头实际分辨率: {frame_width}x{frame_height}")
        
        # 计算左右图像的宽度
        left_width = frame_width // 2
        right_width = frame_width - left_width
        logging.info(f"KS861摄像头左右图像分割尺寸: 左{left_width}x{frame_height}, 右{right_width}x{frame_height}")
        
        try:
            frame_count = 0
            consecutive_failures = 0
            max_consecutive_failures = 10
            
            while self.running:
                try:
                    ret, frame = self.cap.read()
                    frame_count += 1
                    
                    if frame_count % 10 == 0:
                        logging.debug(f"已处理{frame_count}帧，队列中当前有{self.output_queue.qsize()}项")
                    
                    if not ret or frame is None or frame.size == 0:
                        consecutive_failures += 1
                        logging.warning(f"无法获取KS861摄像头帧 (第{frame_count}帧)，连续失败次数: {consecutive_failures}")
                        
                        # 如果连续失败次数超过阈值，尝试重新初始化摄像头
                        if consecutive_failures >= max_consecutive_failures:
                            logging.warning(f"连续{max_consecutive_failures}次无法获取摄像头帧，尝试重新初始化...")
                            try:
                                self.cap.release()
                                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
                                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width * 2)
                                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
                                self.cap.set(cv2.CAP_PROP_FPS, 15)
                                consecutive_failures = 0
                                logging.info("摄像头重新初始化完成")
                            except Exception as e:
                                logging.error(f"摄像头重新初始化失败: {str(e)}")
                        
                        time.sleep(0.01)  # 短暂休眠避免CPU占用过高
                        continue
                    
                    # 重置连续失败计数
                    consecutive_failures = 0
                    
                    # KS861摄像头图像分割：将一帧图像分为左右两部分
                    left_frame = frame[:, :left_width, :]  # 左目图像
                    right_frame = frame[:, right_width:, :]  # 右目图像
                    
                    # 调整尺寸到指定的显示宽度和高度
                    left_frame = cv2.resize(left_frame, (self.display_width, self.display_height))
                    right_frame = cv2.resize(right_frame, (self.display_width, self.display_height))
                    
                    # 放入输出队列，包含左右目图像
                    timestamp = time.time()
                    if self.output_queue.full():
                        self.output_queue.get()  # 移除最旧的帧
                    
                    try:
                        self.output_queue.put((left_frame, right_frame, timestamp), block=False)
                        if frame_count % 50 == 0:
                            logging.info(f"成功放入第{frame_count}帧数据到队列")
                    except Exception as e:
                        logging.error(f"放入队列失败: {str(e)}")
                    
                except Exception as e:
                    logging.error(f"图像采集过程发生错误: {str(e)}")
                    time.sleep(0.01)
                
        finally:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                logging.info("KS861摄像头资源已释放")

class PreprocessingStage(PipelineStage):
    """
    预处理阶段 - 处理KS861摄像头的左右目图像对
    直接在目标设备上生成Half张量，避免跨设备传输导致的类型转换
    """
    def __init__(self, device=torch.device('cuda'), **kwargs):
        super().__init__(**kwargs)
        self.device = device
    
    def run(self):
        while self.running:
            try:
                # 从输入队列获取左右目帧
                left_frame, right_frame, timestamp = self.input_queue.get(timeout=0.1)
                
                # 处理左目图像 - 直接转换为half类型并移动到目标设备
                # 从numpy数组直接生成Half张量，避免中间的Float32转换
                left_rgb = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
                # 创建Float32张量后立即转换为Half，避免中间操作滞留Float32
                left_tensor = torch.from_numpy(left_rgb).float().to(self.device).half()
                left_tensor = einops.rearrange(left_tensor, "h w c -> 1 c h w")
                left_tensor = (left_tensor / 255.0) * 2.0 - 1.0  # 归一化到[-1, 1]
                # 确保结果仍然是Half类型
                left_tensor = left_tensor.half()
                
                # 处理右目图像 - 直接转换为half类型并移动到目标设备
                # 从numpy数组直接生成Half张量，避免中间的Float32转换
                right_rgb = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
                # 创建Float32张量后立即转换为Half，避免中间操作滞留Float32
                right_tensor = torch.from_numpy(right_rgb).float().to(self.device).half()
                right_tensor = einops.rearrange(right_tensor, "h w c -> 1 c h w")
                right_tensor = (right_tensor / 255.0) * 2.0 - 1.0  # 归一化到[-1, 1]
                # 确保结果仍然是Half类型
                right_tensor = right_tensor.half()
                
                # 放入输出队列，包含左右目张量和原始帧
                if self.output_queue.full():
                    self.output_queue.get()
                self.output_queue.put((left_tensor, right_tensor, left_frame, right_frame, timestamp))
                
            except Exception as e:
                # 处理队列超时或其他异常
                if not self.running:
                    break
                time.sleep(0.01)

class DepthEstimationStage(PipelineStage):
    """
    深度估计阶段 - 处理KS861摄像头的左右目图像对
    """
    def __init__(self, pipe, buffer_size=5, **kwargs):
        super().__init__(**kwargs)
        self.pipe = pipe
        self.buffer_size = buffer_size
        self.device = next(self.pipe.unet.parameters()).device  # 缓存设备信息
        # 缓存模型数据类型
        self.model_dtype = next(self.pipe.unet.parameters()).dtype
        self.left_frame_buffer = []  # 左目帧缓冲区
        self.right_frame_buffer = []  # 右目帧缓冲区
        self.original_left_buffer = []  # 原始左目帧缓冲区
        self.original_right_buffer = []  # 原始右目帧缓冲区
        
        # 确保模型在CUDA设备上
        if not self.device.type == 'cuda':
            logging.warning(f"DepthEstimationStage: 模型不在CUDA设备上，当前设备: {self.device}")
        
        # 记录模型数据类型
        logging.debug(f"DepthEstimationStage: 模型数据类型: {self.model_dtype}")
    
    def run(self):
        while self.running:
            try:
                # 从输入队列获取预处理后的左右目帧
                try:
                    left_tensor, right_tensor, left_frame, right_frame, timestamp = self.input_queue.get(timeout=0.1)
                    logging.debug(f"DepthEstimationStage: 成功获取预处理帧，当前队列大小: {self.input_queue.qsize()}")
                except Empty:
                    # 队列空的情况，不报错只记录
                    logging.debug("DepthEstimationStage: 输入队列为空")
                    time.sleep(0.01)
                    continue
                
                # 三重类型验证：输入队列 - 确保张量在正确的设备上且为half类型
                left_tensor = left_tensor.to(self.device).half()
                right_tensor = right_tensor.to(self.device).half()
                
                # 三重类型验证：缓冲区 - 锁定缓冲区类型，确保所有帧均为Half类型
                self.left_frame_buffer.append(left_tensor)
                self.right_frame_buffer.append(right_tensor)
                self.original_left_buffer.append(left_frame)
                self.original_right_buffer.append(right_frame)
                
                # 保持缓冲区大小
                if len(self.left_frame_buffer) > self.buffer_size:
                    self.left_frame_buffer.pop(0)
                    self.right_frame_buffer.pop(0)
                    self.original_left_buffer.pop(0)
                    self.original_right_buffer.pop(0)
                
                # 缓冲区大小设为1，适配KS861的15fps硬件限制
                min_frames_required = 1
                if len(self.left_frame_buffer) < min_frames_required:
                    logging.debug(f"DepthEstimationStage: 缓冲区帧数不足 ({len(self.left_frame_buffer)}/{min_frames_required})，继续积累...")
                    continue
                
                try:
                    # 准备输入（使用当前可用的帧）
                    # 三重类型验证：推理前 - 添加输入-模型dtype一致性校验
                    # 使用模型实际的数据类型，而不是硬编码的half
                    left_frames = torch.cat(self.left_frame_buffer, dim=0).to(dtype=self.model_dtype)
                    right_frames = torch.cat(self.right_frame_buffer, dim=0).to(dtype=self.model_dtype)
                    
                    left_frames = einops.rearrange(left_frames, "n c h w -> 1 n c h w")
                    right_frames = einops.rearrange(right_frames, "n c h w -> 1 n c h w")
                    
                    # 确保输入是3通道格式
                    if left_frames.shape[2] > 3:
                        left_frames = left_frames[:, :, :3, :, :]  # 只保留前3个通道
                    if right_frames.shape[2] > 3:
                        right_frames = right_frames[:, :, :3, :, :]  # 只保留前3个通道
                    
                    # 确保推理前张量类型正确且与模型一致
                    left_frames = left_frames.to(self.device).to(dtype=self.model_dtype).contiguous()
                    right_frames = right_frames.to(self.device).to(dtype=self.model_dtype).contiguous()
                    
                    # 显式校验输入与模型dtype是否一致
                    if left_frames.dtype != self.model_dtype:
                        logging.warning(f"DepthEstimationStage: 输入dtype({left_frames.dtype})与模型dtype({self.model_dtype})不匹配，强制转换")
                    left_frames = left_frames.to(dtype=self.model_dtype)
                    right_frames = right_frames.to(dtype=self.model_dtype)
                    
                    # 深度估计
                    with torch.no_grad():
                        pipe_out = self.pipe.forward(
                            left_frames=left_frames,
                            right_frames=right_frames,
                            dilations=[1],
                            verbose=False
                        )
                    
                    depth_pred = pipe_out.depth_pred[-1:]  # 取最后一帧
                    confidence = pipe_out.confidence[-1:] if hasattr(pipe_out, 'confidence') else None
                    
                    # 获取对应的原始帧（使用左目图像作为显示用）
                    latest_original_frame = self.original_left_buffer[-1]
                    
                    # 放入输出队列
                    if self.output_queue.full():
                        self.output_queue.get()  # 移除最旧的项
                    
                    try:
                        self.output_queue.put((latest_original_frame, depth_pred, confidence, timestamp), block=False)
                        logging.debug(f"DepthEstimationStage: 成功放入深度估计结果到输出队列")
                    except Exception as queue_error:
                        logging.error(f"DepthEstimationStage: 放入输出队列失败: {str(queue_error)}")
                except Exception as inference_error:
                    logging.error(f"DepthEstimationStage: 推理过程发生错误: {str(inference_error)}")
                    # 发生推理错误时，仍然传递原始帧，但标记为失败
                    latest_original_frame = self.original_left_buffer[-1]
                    if self.output_queue.full():
                        self.output_queue.get()
                    self.output_queue.put((latest_original_frame, None, None, timestamp))
                
            except Exception as e:
                if not self.running:
                    break
                import traceback
                logging.warning(f"深度估计出错: {str(e)}")
                logging.warning(f"错误详情: {traceback.format_exc()}")
                time.sleep(0.01)

class PostprocessingStage(PipelineStage):
    """
    后处理阶段
    """
    def __init__(self, color_map='Spectral_r', **kwargs):
        super().__init__(**kwargs)
        self.color_map = color_map
    
    def run(self):
        frame_counter = 0
        queue_empty_count = 0
        max_queue_empty_warn = 10  # 最多显示10次队列空警告
        
        while self.running:
            try:
                # 从输入队列获取结果
                try:
                    original_frame, depth_pred, confidence, timestamp = self.input_queue.get(timeout=0.1)
                    frame_counter += 1
                    queue_empty_count = 0  # 重置队列空计数
                    
                    if frame_counter % 10 == 0:
                        logging.debug(f"PostprocessingStage: 成功处理第{frame_counter}帧，队列中当前有{self.input_queue.qsize()}项")
                except Empty:
                    queue_empty_count += 1
                    # 只在开始时或连续多次空时显示警告
                    if queue_empty_count <= max_queue_empty_warn or queue_empty_count % 100 == 0:
                        logging.debug(f"PostprocessingStage: 输入队列为空 (第{queue_empty_count}次)")
                    time.sleep(0.01)
                    continue
                
                # 处理深度图
                if depth_pred is not None:
                    try:
                        colored_np = colorize_depth_multi_thread(
                            depth=depth_pred.cpu().numpy(),
                            valid_mask=None,
                            chunk_size=1,
                            num_threads=1,
                            color_map=self.color_map,
                            verbose=False
                        )[0]
                        # 确保深度图尺寸与原始图像一致
                        colored_bgr = cv2.cvtColor(colored_np, cv2.COLOR_RGB2BGR)
                        colored_bgr = cv2.resize(colored_bgr, 
                                              (original_frame.shape[1], original_frame.shape[0]),
                                              interpolation=cv2.INTER_LINEAR)
                    except Exception as depth_error:
                        logging.error(f"后处理: 深度图着色失败: {str(depth_error)}")
                        # 创建默认的深度图
                        colored_bgr = np.zeros((original_frame.shape[0], original_frame.shape[1], 3), dtype=np.uint8)
                        cv2.putText(colored_bgr, "深度图着色失败",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # 创建默认的黑色深度图
                    colored_bgr = np.zeros((original_frame.shape[0], original_frame.shape[1], 3), dtype=np.uint8)
                    cv2.putText(colored_bgr, "深度估计失败",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 放入输出队列
                if self.output_queue.full():
                    self.output_queue.get()  # 移除最旧的项
                
                try:
                    self.output_queue.put((original_frame, colored_bgr, timestamp), block=False)
                except Exception as queue_error:
                    logging.error(f"后处理: 放入输出队列失败: {str(queue_error)}")
                
            except Exception as e:
                if not self.running:
                    break
                logging.warning(f"后处理出错: {str(e)}")
                # 只在调试模式下显示完整异常信息
                if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
                    import traceback
                    logging.debug(f"后处理错误详情: {traceback.format_exc()}")
                time.sleep(0.01)

class DisplayStage(PipelineStage):
    """
    显示阶段
    """
    def __init__(self, display_width=640, display_height=720, save_video=False, output_dir="./realtime_output", **kwargs):
        super().__init__(**kwargs)
        self.display_width = display_width
        self.display_height = display_height
        self.save_video = save_video
        self.output_dir = output_dir
        self.video_writer = None
        self.fps_counter = []
        self.window_name = "KS861实时深度估计"
    
    def initialize_video_writer(self):
        if not self.save_video:
            return
        
        # 创建输出目录
        output_dir = Path(self.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化视频写入器
        combined_width = self.display_width * 2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = output_dir / "ks861_realtime_depth.mp4"
        
        self.video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            15,  # 目标帧率
            (combined_width, self.display_height)
        )
        logging.info(f"视频将保存到: {video_path}")
    
    def run(self):
        # 初始化显示窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width * 2, self.display_height)
        
        # 初始化视频写入器
        self.initialize_video_writer()
        
        # 初始化空帧计数器
        empty_frame_count = 0
        max_empty_display = 10  # 最多显示10次空帧警告
        
        try:
            while self.running:
                try:
                    # 从输入队列获取处理后的帧
                    try:
                        original_frame, colored_bgr, timestamp = self.input_queue.get(timeout=0.1)
                        empty_frame_count = 0  # 重置空帧计数
                        
                        # 组合并显示
                        combined = np.hstack((original_frame, colored_bgr))
                        
                        # 添加FPS信息
                        current_time = time.time()
                        fps = 1.0 / (current_time - timestamp + 1e-8)
                        self.fps_counter.append(fps)
                        if len(self.fps_counter) > 10:
                            self.fps_counter.pop(0)
                        avg_fps = sum(self.fps_counter) / len(self.fps_counter) if self.fps_counter else 0
                        
                        cv2.putText(combined, f"FPS: {avg_fps:.1f}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # 显示结果
                        cv2.imshow(self.window_name, combined)
                        
                        # 保存视频
                        if self.video_writer is not None:
                            self.video_writer.write(combined)
                        
                        # 检查按键
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            logging.info("用户按下 'q' 键，退出程序")
                            self.running = False
                            break
                    except Empty:
                        empty_frame_count += 1
                        # 只在开始时或连续多次空时显示警告
                        if empty_frame_count <= max_empty_display or empty_frame_count % 100 == 0:
                            logging.debug(f"DisplayStage: 输入队列为空 (第{empty_frame_count}次)")
                        
                        # 在连续空队列时显示等待画面，避免窗口空白
                        if empty_frame_count <= max_empty_display * 2:
                            # 创建等待画面
                            waiting_frame = np.zeros((self.display_height, self.display_width * 2, 3), dtype=np.uint8)
                            if empty_frame_count <= max_empty_display:
                                cv2.putText(waiting_frame, "等待图像数据...",
                                            (self.display_width // 2 - 100, self.display_height // 2),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                            cv2.imshow(self.window_name, waiting_frame)
                            cv2.waitKey(1)  # 确保画面刷新
                        
                        time.sleep(0.01)
                except Exception as e:
                    if not self.running:
                        break
                    logging.warning(f"显示出错: {str(e)}")
                    # 只在调试模式下显示完整异常信息
                    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
                        import traceback
                        logging.debug(f"显示错误详情: {traceback.format_exc()}")
                    time.sleep(0.01)
            
        finally:
            # 释放资源
            if self.video_writer is not None:
                self.video_writer.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)

def main():
    # 配置日志输出级别为DEBUG，显示详细调试信息
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # -------------------- 命令行参数解析 --------------------
    parser = argparse.ArgumentParser(description="使用BinGeo-LDM++运行KS861相机实时深度估计。")

    # 摄像头参数
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="摄像头设备ID，默认0"
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=15,
        help="摄像头捕获帧率，默认15 FPS（KS861硬件限制）"
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=640,
        help="显示窗口宽度，默认640"
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=480,
        help="显示窗口高度，默认480（KS861摄像头实际高度）"
    )

    # 输出参数
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./output/realtime_output",
        help="保存结果的目录路径，默认：./output/realtime_output"
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
        default=256,
        help="处理分辨率，默认256（平衡速度和精度）"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "fp32"],
        default="fp16",
        help="数据类型，默认fp16（提高速度）"
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

    # -------------------- 配置参数 --------------------
    args = parser.parse_args()

    # -------------------- 设备配置 --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("使用GPU加速推理")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA不可用，使用CPU（速度会很慢）")
    logging.info(f"使用设备: {device}")

    # -------------------- 模型初始化 --------------------
    logging.info(f"加载模型: {args.checkpoint}")
    
    # 创建额外参数字典
    custom_kwargs = {
        'config_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'ks861.yaml'),
        'verbose': args.verbose
    }
    
    # 确定目标数据类型
    target_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    
    # 初始化KS861Pipeline，确保模型和预处理阶段使用相同设备和dtype
    # 只使用balanced策略，这是diffusers库唯一支持的device_map策略
    pipe = KS861Pipeline.from_pretrained(
        args.checkpoint,
        torch_dtype=target_dtype,
        device_map='balanced' if device.type == 'cuda' else None,
        kwargs=custom_kwargs
    )
    
    # 强制所有组件为指定dtype，确保模型dtype全局生效
    # 先重置设备映射策略，因为使用了balanced device_map后不能直接调用to()
    pipe.reset_device_map()
    pipe = pipe.to(device, dtype=target_dtype)
    
    # 显式遍历所有模型参数，强制转换为目标dtype
    if args.dtype == "fp16":
        for param in pipe.unet.parameters():
            param.data = param.data.half()
        for param in pipe.vae.parameters():
            param.data = param.data.half()
    
    pipe.enable_optimizations()

    # -------------------- 初始化流水线 --------------------
    # 创建队列
    capture_queue = Queue(maxsize=5)
    preprocess_queue = Queue(maxsize=5)
    inference_queue = Queue(maxsize=5)
    postprocess_queue = Queue(maxsize=5)
    display_queue = Queue(maxsize=5)
    
    # 创建流水线阶段
    capture_stage = ImageCaptureStage(
        camera_id=args.camera_id,
        camera_fps=args.camera_fps,
        display_width=args.display_width,
        display_height=args.display_height,
        output_queue=capture_queue
    )
    
    preprocess_stage = PreprocessingStage(
        device=device,
        input_queue=capture_queue,
        output_queue=preprocess_queue
    )
    
    inference_stage = DepthEstimationStage(
        pipe=pipe,
        buffer_size=1,
        input_queue=preprocess_queue,
        output_queue=inference_queue
    )
    
    postprocess_stage = PostprocessingStage(
        color_map=args.cmap,
        input_queue=inference_queue,
        output_queue=postprocess_queue
    )
    
    display_stage = DisplayStage(
        display_width=args.display_width,
        display_height=args.display_height,
        save_video=args.save_video,
        output_dir=args.output_dir,
        input_queue=postprocess_queue
    )
    
    try:
        # 启动流水线
        logging.info("启动流水线处理...")
        
        # 先启动摄像头捕获阶段
        capture_stage.start()
        logging.info("等待摄像头初始化完成...")
        
        # 等待摄像头初始化完成（最多等待5秒）
        max_wait_time = 5.0  # 最大等待时间（秒）
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            # 检查是否有图像进入预处理队列
            if not capture_queue.empty():
                logging.info("摄像头已成功捕获图像，开始启动其他处理阶段...")
                break
            time.sleep(0.1)
        
        # 如果摄像头初始化超时，打印警告但继续尝试启动其他阶段
        if time.time() - start_time >= max_wait_time:
            logging.warning(f"摄像头初始化超时（{max_wait_time}秒），但仍尝试启动其他处理阶段...")
        
        # 启动其他处理阶段
        processing_stages = [preprocess_stage, inference_stage, postprocess_stage, display_stage]
        for stage in processing_stages:
            stage.start()
            time.sleep(0.1)  # 短暂延迟确保阶段正确启动
        
        # 主循环等待直到显示阶段结束
        while display_stage.running:
            time.sleep(0.1)
                
    except KeyboardInterrupt:
        logging.info("用户中断程序")
    finally:
        # 停止所有阶段
        logging.info("停止流水线处理，释放资源...")
        stages = [capture_stage, preprocess_stage, inference_stage, postprocess_stage, display_stage]
        for stage in reversed(stages):  # 反向停止以避免死锁
            stage.stop()
        
        logging.info("程序已退出，所有资源已释放")

if __name__ == "__main__":
    main()
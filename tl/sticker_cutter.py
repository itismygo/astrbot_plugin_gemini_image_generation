"""主体+附件吸附分割算法（可选项）

核心流程：
1. 去除白底并提取前景掩码
2. 识别主体与附件，按距离和方向吸附附件
3. 合并后裁剪并转为透明背景
"""

from __future__ import annotations

import cv2
import numpy as np

class BlackLineCutter:
    """基于黑色分割线的切分器 (Otsu + Peak Detection + Recursive Border Trimming)"""
    
    def __init__(self, black_threshold: int = 50, padding: int = 2):
        self.black_threshold = black_threshold
        self.padding = padding

    def _find_split_line(self, proj: np.ndarray, perpendicular_len: int) -> tuple[int, int]:
        """
        寻找分割线的位置和宽度
        proj: 投影数组 (黑色像素计数)
        perpendicular_len: 垂直方向的长度 (用于归一化)
        Returns: (start, end)
        """
        total_len = len(proj)
        center = total_len // 2
        # 在中间 40% 区域搜索
        search_radius = int(total_len * 0.2)
        start_search = max(0, center - search_radius)
        end_search = min(total_len, center + search_radius)
        
        region = proj[start_search:end_search]
        if len(region) == 0:
            return center, center
            
        # 找到峰值
        max_val = np.max(region)
        local_idx = np.argmax(region)
        global_idx = start_search + local_idx
        
        # 判定是否为有效线
        # 如果峰值太低 (说明这一行没多少黑色)，可能不是线
        # 比如：至少要有 30% 的像素是黑色的
        if max_val < perpendicular_len * 0.3:
            # 没找到明显的线，默认中心切割，宽度为0
            return center, center
            
        # 确定线的宽度 (Plateau Detection)
        # 向左右扩展，直到数值下降到峰值的一半以下
        # 或者低于 30% 的覆盖率
        threshold = max(max_val * 0.5, perpendicular_len * 0.3)
        
        l, r = global_idx, global_idx
        
        # 向左找边缘
        while l > 0 and proj[l] > threshold:
            l -= 1
            
        # 向右找边缘
        while r < total_len - 1 and proj[r] > threshold:
            r += 1
            
        # 稍微向外扩一点点，确保切干净
        return l, r + 1

    def _trim_black_borders(self, img: np.ndarray) -> np.ndarray:
        """
        递归切除四周的黑色线条，直到边缘不再是黑色
        改进版：允许跳过外围的白色噪点，寻找最内侧的黑线边界
        """
        if img is None or img.size == 0:
            return img
            
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 定义什么是"黑线"：一行的平均亮度低于某个阈值
        LINE_THRESHOLD = 80
        # 搜索范围限制在 25% 以内，防止切掉主体
        h_limit = int(h * 0.25)
        w_limit = int(w * 0.25)
        
        row_means = np.mean(gray, axis=1)
        col_means = np.mean(gray, axis=0)
        
        # Top: 找范围内最后一个黑线的位置
        top_cut = 0
        for i in range(min(h_limit, h)):
            if row_means[i] < LINE_THRESHOLD:
                top_cut = i + 1
        
        # Bottom: 找范围内(从下往上)最后一个黑线的位置
        bottom_cut = h
        for i in range(min(h_limit, h)):
            idx = h - 1 - i
            if row_means[idx] < LINE_THRESHOLD:
                bottom_cut = idx
                
        # Left
        left_cut = 0
        for i in range(min(w_limit, w)):
            if col_means[i] < LINE_THRESHOLD:
                left_cut = i + 1
                
        # Right
        right_cut = w
        for i in range(min(w_limit, w)):
            idx = w - 1 - i
            if col_means[idx] < LINE_THRESHOLD:
                right_cut = idx
                
        # 安全检查
        if top_cut >= bottom_cut or left_cut >= right_cut:
            return img
            
        return img[top_cut:bottom_cut, left_cut:right_cut]

    def _trim_whitespace(self, img: np.ndarray, pad: int | None = None) -> np.ndarray:
        """
        去除图片四周的空白区域 (白底)
        """
        if img is None or img.size == 0:
            return img
            
        if pad is None:
            pad = self.padding

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用 Otsu 二值化
        try:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        except:
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        coords = cv2.findNonZero(binary)
        if coords is None:
            return img # 全白
            
        x, y, w, h = cv2.boundingRect(coords)
        
        # 稍微留一点点边距
        margin = pad
        h_img, w_img = img.shape[:2]
        
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w_img, x + w + margin)
        y2 = min(h_img, y + h + margin)
        
        return img[y1:y2, x1:x2]

    def _normalize_padding(self, img: np.ndarray, pad: int = 4) -> np.ndarray:
        """
        标准化内边距：
        1. 找到内容最小包围盒 (Otsu)
        2. 创建新画布，大小为 内容尺寸 + 2*pad
        3. 将内容居中放置
        这样无论原来白边多少，最终都只有固定的 pad 像素
        """
        if img is None or img.size == 0:
            return img
            
        h_img, w_img = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Otsu 二值化找内容
        try:
            # THRESH_BINARY_INV: 背景白(255)->0, 前景黑(0)->255
            # 这样 findNonZero 找的就是前景
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        except:
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            
        coords = cv2.findNonZero(binary)
        if coords is None:
            # 全白图片，返回一个固定大小的白块或者原图
            return img
            
        x, y, w, h = cv2.boundingRect(coords)
        
        # 提取纯内容
        content = img[y:y+h, x:x+w]
        
        # 创建新画布
        new_h = h + 2 * pad
        new_w = w + 2 * pad
        canvas = np.full((new_h, new_w, 3), 255, dtype=np.uint8)
        
        # 粘贴内容
        canvas[pad:pad+h, pad:pad+w] = content
        
        return canvas

    def process_image(self, img: np.ndarray, debug: bool = False) -> tuple[list[np.ndarray] | None, np.ndarray | None]:
        """
        检测黑色直线并切割图片
        Returns:
            (crops, debug_img)
        """
        if img is None:
            return None, None
            
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. 全局二值化 (找黑线)
        # 使用 Otsu 自动寻找最佳阈值，应对不同程度的"黑"
        _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 2. 投影
        row_proj = np.sum(binary_inv > 0, axis=1) # (H,)
        col_proj = np.sum(binary_inv > 0, axis=0) # (W,)
        
        # 3. 寻找分割线 (Y轴和X轴)
        y_split_start, y_split_end = self._find_split_line(row_proj, w)
        x_split_start, x_split_end = self._find_split_line(col_proj, h)
        
        # 4. 定义四个区域
        # 如果没找到线 (start == end == center)，就相当于中心切割
        regions = [
            (0, y_split_start, 0, x_split_start),          # Top-Left
            (0, y_split_start, x_split_end, w),            # Top-Right
            (y_split_end, h, 0, x_split_start),            # Bottom-Left
            (y_split_end, h, x_split_end, w)               # Bottom-Right
        ]
        
        crops = []
        debug_boxes = []
        
        for (y1, y2, x1, x2) in regions:
            # 避免无效区域
            if y2 <= y1 or x2 <= x1:
                continue
                
            crop = img[y1:y2, x1:x2]
            
            # 5. 三步走策略：
            # Step 1: 先去除外围的白底 (pad=0)，确保黑边暴露在最外层
            crop = self._trim_whitespace(crop, pad=0)
            
            # Step 2: 强力切除黑边
            crop = self._trim_black_borders(crop)

            # 额外收缩：去黑边后强制向内收缩 2 像素，防止残留黑边
            if crop.size > 0:
                h_c, w_c = crop.shape[:2]
                if h_c > 4 and w_c > 4:
                    crop = crop[2:h_c-2, 2:w_c-2]
            
            # Step 3: 标准化内边距 (User Request: 最小矩形框住 + 4px 白边)
            # 这步替代了之前的 _trim_whitespace 和 强制 shrink
            final_crop = self._normalize_padding(crop, pad=4)
            
            if final_crop.size > 0:
                crops.append(final_crop)
                debug_boxes.append((x1, y1, x2-x1, y2-y1))

        debug_img = None
        if debug:
            debug_img = img.copy()
            # 画出分割线
            cv2.rectangle(debug_img, (0, y_split_start), (w, y_split_end), (0, 0, 255), -1)
            cv2.rectangle(debug_img, (x_split_start, 0), (x_split_end, h), (0, 0, 255), -1)
            
            for (x, y, w_box, h_box) in debug_boxes:
                cv2.rectangle(debug_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            
        return crops, debug_img

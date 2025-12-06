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
    
    def __init__(self, black_threshold: int = 20, padding: int = 2):
        self.black_threshold = black_threshold
        self.padding = padding

    def _merge_gaps(self, line_pixels: np.ndarray, max_gap: int = 5) -> np.ndarray:
        """
        缝合线段中的小间隙
        line_pixels: 二值化的一行/一列像素 (0/255)
        max_gap: 允许缝合的最大间隙像素数
        """
        # 找到所有黑色段 (255)
        padded = np.concatenate(([0], line_pixels, [0]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff > 0)[0]
        ends = np.where(diff < 0)[0]
        
        if len(starts) < 2:
            return line_pixels
            
        merged = line_pixels.copy()
        # 检查相邻线段之间的间隙
        for i in range(len(starts) - 1):
            gap_start = ends[i]
            gap_end = starts[i+1]
            gap_len = gap_end - gap_start
            
            if gap_len <= max_gap:
                # 缝合间隙
                merged[gap_start:gap_end] = 255
                
        return merged

    def _find_split_line(self, proj: np.ndarray, perpendicular_len: int, search_start: int, search_end: int, 
                        binary_img: np.ndarray | None = None, is_horizontal: bool = True, division_count: int = 2) -> tuple[int, int]:
        """
        寻找分割线的位置和宽度
        proj: 投影数组 (黑色像素计数)
        perpendicular_len: 垂直方向的长度 (用于归一化)
        search_start: 搜索起始位置
        search_end: 搜索结束位置
        binary_img: 二值化图像 (用于连贯性检查)
        is_horizontal: 是否正在寻找水平线 (用于从 binary_img 提取像素)
        division_count: 切割的份数 (用于计算合理的连贯性阈值)
        Returns: (start, end)
        """
        if search_start >= search_end:
            center = (search_start + search_end) // 2
            return center, center

        region = proj[search_start:search_end]
        if len(region) == 0:
            center = (search_start + search_end) // 2
            return center, center
            
        # 找到峰值
        max_val = np.max(region)
        local_idx = np.argmax(region)
        global_idx = search_start + local_idx
        
        # 判定是否为有效线
        # 1. 总量检查：降低到 40%
        if max_val < perpendicular_len * 0.4:
            center = (search_start + search_end) // 2
            return center, center

        # 2. 连贯性检查
        if binary_img is not None:
            if is_horizontal:
                line_pixels = binary_img[global_idx, :]
            else:
                line_pixels = binary_img[:, global_idx]
            
            # 计算最长连续黑色段 (255)
            padded = np.concatenate(([0], line_pixels, [0]))
            diff = np.diff(padded.astype(int))
            starts = np.where(diff > 0)[0]
            ends = np.where(diff < 0)[0]
            
            max_len = 0
            if len(starts) > 0:
                max_len = np.max(ends - starts)
                
            # 连贯性阈值：40%
            if max_len < perpendicular_len * 0.4:
                center = (search_start + search_end) // 2
                return center, center
            
        # 确定线的宽度 (Plateau Detection)
        threshold = max(max_val * 0.5, perpendicular_len * 0.4)
        
        l, r = global_idx, global_idx
        
        # 向左找边缘
        while l > 0 and proj[l] > threshold:
            l -= 1
            
        # 向右找边缘
        while r < len(proj) - 1 and proj[r] > threshold:
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
        # 使用 self.black_threshold (默认50) 而不是硬编码的 80，避免切掉深色主体
        LINE_THRESHOLD = self.black_threshold
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



    def _get_content_bounds(self, img: np.ndarray, threshold: int = 30, padding: int = 2) -> tuple[int, int, int, int]:
        """
        仿照 JS 算法：去除四周的黑/白边框
        """
        if img is None:
            return 0, 0, 0, 0
            
        h, w = img.shape[:2]
        
        # 检查黑色 (所有通道 < threshold)
        is_black = np.max(img, axis=2) < threshold
        
        # 检查白色 (所有通道 > 255 - threshold)
        is_white = np.min(img, axis=2) > (255 - threshold)
        
        # 认为是背景的像素
        is_border = is_black | is_white
        
        # 内容像素 (非背景)
        is_content = ~is_border
        
        coords = cv2.findNonZero(is_content.astype(np.uint8))
        if coords is None:
            return 0, 0, w, h # 没找到内容，返回全图
            
        x, y, w_content, h_content = cv2.boundingRect(coords)
        
        # 应用内缩 padding (去除残留边框)
        # 注意：boundingRect 返回的是 (x, y, w, h)
        # 我们需要调整 x, y 和 w, h
        
        new_x = x + padding
        new_y = y + padding
        new_w = w_content - 2 * padding
        new_h = h_content - 2 * padding
        
        # 边界检查
        if new_w <= 0 or new_h <= 0:
            return x, y, w_content, h_content
            
        return new_x, new_y, new_w, new_h

    def process_image(self, img: np.ndarray, rows: int = 2, cols: int = 2, debug: bool = False) -> tuple[list[np.ndarray] | None, np.ndarray | None]:
        """
        检测黑色直线并切割图片
        Returns:
            (crops, debug_img)
        """
        if img is None:
            return None, None

        # Step 0: 预处理 - 裁剪掉四周的黑/白边框 (借鉴 JS 算法)
        # 这能极大提高后续网格切割的准确性，因为去除了干扰的外部边框
        # 且如果去边后就是完美网格，后续的切割会非常准
        cx, cy, cw, ch = self._get_content_bounds(img)
        img_cropped = img[cy:cy+ch, cx:cx+cw]
        
        if img_cropped.size == 0:
            img_cropped = img # 回退
            
        h, w = img_cropped.shape[:2]
        gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        
        # 1. 全局二值化 (找黑线)
        # 使用固定阈值而不是 Otsu，避免深色主体被误判为黑线
        # Otsu 会自动寻找最佳阈值，如果图片整体偏暗，阈值会很高，导致深色主体变成"黑色"
        # 我们只关心真正的黑线，所以使用 self.black_threshold (默认20)
        _, binary_inv = cv2.threshold(gray, self.black_threshold, 255, cv2.THRESH_BINARY_INV)
        
        # 2. 投影
        row_proj = np.sum(binary_inv > 0, axis=1) # (H,)
        col_proj = np.sum(binary_inv > 0, axis=0) # (W,)
        
        # 3. 寻找分割线 (Y轴和X轴)
        # 寻找 rows-1 条水平分割线
        y_splits = [0]
        for i in range(1, rows):
            # 预期位置
            expected_y = int(h * i / rows)
            # 搜索范围：限制在单元格高度的 5% 以内，只进行微调
            # 既然已经去除了边框，理论上分割线就在均分位置附近
            cell_h = h / rows
            search_radius = max(5, int(cell_h * 0.05))
            
            start_search = max(0, expected_y - search_radius)
            end_search = min(h, expected_y + search_radius)
            
            # 注意：寻找水平线时，投影是 row_proj (H,)，垂直长度是 w
            start, end = self._find_split_line(row_proj, w, start_search, end_search, binary_inv, is_horizontal=True, division_count=cols)
            y_splits.append(start)
            y_splits.append(end)
        y_splits.append(h)
        
        # 寻找 cols-1 条垂直分割线
        x_splits = [0]
        for i in range(1, cols):
            # 预期位置
            expected_x = int(w * i / cols)
            # 搜索范围：限制在单元格宽度的 5% 以内，只进行微调
            cell_w = w / cols
            search_radius = max(5, int(cell_w * 0.05))
            
            start_search = max(0, expected_x - search_radius)
            end_search = min(w, expected_x + search_radius)
            
            # 注意：寻找垂直线时，投影是 col_proj (W,)，垂直长度是 h
            start, end = self._find_split_line(col_proj, h, start_search, end_search, binary_inv, is_horizontal=False, division_count=rows)
            x_splits.append(start)
            x_splits.append(end)
        x_splits.append(w)
        
        # 4. 定义区域
        regions = []
        for r in range(rows):
            for c in range(cols):
                # y_splits 结构: [0, split1_start, split1_end, split2_start, split2_end, ..., h]
                # 第 r 行的区域对应 y_splits[2*r] 到 y_splits[2*r+1]
                # 比如 rows=2: [0, s1, e1, h]
                # r=0: 0 -> s1 (idx 0 -> 1)
                # r=1: e1 -> h (idx 2 -> 3)
                
                y1 = y_splits[2*r]
                y2 = y_splits[2*r+1]
                
                x1 = x_splits[2*c]
                x2 = x_splits[2*c+1]
                
                regions.append((y1, y2, x1, x2))
        
        crops = []
        debug_boxes = []
        
        for (y1, y2, x1, x2) in regions:
            # 避免无效区域
            if y2 <= y1 or x2 <= x1:
                continue
                
            crop = img_cropped[y1:y2, x1:x2]
            
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
            debug_img = img_cropped.copy()
            # 画出分割线
            # 绘制水平分割线 (红色)
            for i in range(1, len(y_splits)-1, 2):
                s, e = y_splits[i], y_splits[i+1]
                cv2.rectangle(debug_img, (0, s), (w, e), (0, 0, 255), -1)
                
            # 绘制垂直分割线 (蓝色)
            for i in range(1, len(x_splits)-1, 2):
                s, e = x_splits[i], x_splits[i+1]
                cv2.rectangle(debug_img, (s, 0), (e, h), (255, 0, 0), -1)
                
            # 画出最终的裁剪框 (绿色)
            for (dx, dy, dw, dh) in debug_boxes:
                cv2.rectangle(debug_img, (dx, dy), (dx+dw, dy+dh), (0, 255, 0), 2)
                
        return crops, debug_img

"""主体+附件吸附分割算法（可选项）

核心流程：
1. 去除白底并提取前景掩码
2. 识别主体与附件，按距离和方向吸附附件
3. 合并后裁剪并转为透明背景
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# (x1, y1, x2, y2)
Box = tuple[int, int, int, int]


@dataclass
class Region:
    """表示一个检测到的区域"""

    box: Box
    area: int
    center: tuple[float, float]
    is_main: bool = False


class StickerCutter:
    """表情包分割器，侧重主体与附件吸附"""

    def __init__(
        self,
        min_area: int = 300,
        main_ratio_range: tuple[float, float] = (0.4, 2.5),
        main_area_threshold: float = 0.35,
        attach_max_gap: int = 60,
        attach_prefer_left: float = 1.3,
        attach_prefer_up: float = 1.1,
        max_width_ratio: float = 2.5,
        margin: int = 8,
        white_threshold: int = 248,
    ):
        self.min_area = min_area
        self.main_ratio_range = main_ratio_range
        self.main_area_threshold = main_area_threshold
        self.attach_max_gap = attach_max_gap
        self.attach_prefer_left = attach_prefer_left
        self.attach_prefer_up = attach_prefer_up
        self.max_width_ratio = max_width_ratio
        self.margin = margin
        self.white_threshold = white_threshold

    def _prepare_foreground(self, img: np.ndarray) -> np.ndarray:
        """去除白底，提取前景掩码"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(img)
        white_mask = (r > self.white_threshold) & (g > self.white_threshold) & (
            b > self.white_threshold
        )
        fg_color = (~white_mask).astype(np.uint8) * 255

        bin_inv = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            15,
        )
        edges = cv2.Canny(gray, 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

        fg = cv2.bitwise_or(fg_color, bin_inv)
        fg = cv2.bitwise_or(fg, edges_dilated)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        return fg

    def _find_all_regions(self, fg: np.ndarray) -> list[Region]:
        """找到所有前景区域，必要时对粘连区域做简单切分"""
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: list[Region] = []
        all_areas = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area > self.min_area:
                all_areas.append(area)

        split_threshold = float("inf")
        median_area = 0.0
        if all_areas:
            median_area = float(np.median(all_areas))
            split_threshold = median_area * 2.5

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area < self.min_area:
                continue

            # 粘连时按长边方向切割
            if area > split_threshold and median_area > 0:
                if cw > 1.8 * ch:
                    num_splits = max(2, int(np.sqrt(area / median_area) + 0.5))
                    split_width = cw // num_splits
                    for i in range(num_splits):
                        sx = x + i * split_width
                        sw = split_width if i < num_splits - 1 else (cw - i * split_width)
                        if sw > 50:
                            box = (sx, y, sx + sw, y + ch)
                            center = (sx + sw / 2, y + ch / 2)
                            regions.append(Region(box=box, area=sw * ch, center=center))
                    continue
                if ch > 1.8 * cw:
                    num_splits = max(2, int(np.sqrt(area / median_area) + 0.5))
                    split_height = ch // num_splits
                    for i in range(num_splits):
                        sy = y + i * split_height
                        sh = split_height if i < num_splits - 1 else (ch - i * split_height)
                        if sh > 50:
                            box = (x, sy, x + cw, sy + sh)
                            center = (x + cw / 2, sy + sh / 2)
                            regions.append(Region(box=box, area=cw * sh, center=center))
                    continue

            box = (x, y, x + cw, y + ch)
            center = (x + cw / 2, y + ch / 2)
            regions.append(Region(box=box, area=area, center=center))

        return regions

    def _classify_regions(self, regions: list[Region]) -> tuple[list[Region], list[Region]]:
        """区分主体与附件"""
        if not regions:
            return [], []

        areas = [r.area for r in regions]
        median_area = float(np.median(areas))

        mains: list[Region] = []
        attaches: list[Region] = []
        for r in regions:
            x1, y1, x2, y2 = r.box
            w = x2 - x1
            h = y2 - y1
            ratio = h / w if w > 0 else 1.0
            is_main = (
                r.area >= self.main_area_threshold * median_area
                and self.main_ratio_range[0] <= ratio <= self.main_ratio_range[1]
            )
            r.is_main = is_main
            if is_main:
                mains.append(r)
            else:
                attaches.append(r)

        if not mains and regions:
            sorted_regions = sorted(regions, key=lambda r: r.area, reverse=True)
            top_area = sorted_regions[0].area
            for r in sorted_regions:
                if r.area >= 0.3 * top_area:
                    r.is_main = True
                    mains.append(r)
                else:
                    attaches.append(r)

        return mains, attaches

    def _box_distance(self, box1: Box, box2: Box) -> float:
        """计算两个框的最小间距"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        if x1_1 < x2_2 and x2_1 > x1_2 and y1_1 < y2_2 and y2_1 > y1_2:
            return 0.0

        horiz_gap = max(0, max(x1_2 - x2_1, x1_1 - x2_2))
        vert_gap = max(0, max(y1_2 - y2_1, y1_1 - y2_2))
        return float(np.hypot(horiz_gap, vert_gap))

    def _attach_regions(self, mains: list[Region], attaches: list[Region]) -> list[Box]:
        """将附件吸附到主体，返回合并后的框"""
        if not mains:
            return []

        main_boxes = [list(m.box) for m in mains]
        widths = [m.box[2] - m.box[0] for m in mains]
        typical_width = float(np.median(widths)) if widths else 100.0

        attaches_sorted = sorted(attaches, key=lambda a: a.area)
        for attach in attaches_sorted:
            a_box = attach.box
            ax1, ay1, ax2, ay2 = a_box
            a_cx, a_cy = attach.center

            best_idx = None
            best_score = float("inf")
            for i, m_box in enumerate(main_boxes):
                mx1, my1, mx2, my2 = m_box
                dist = self._box_distance(a_box, tuple(m_box))
                if dist > self.attach_max_gap:
                    continue

                m_cx = (mx1 + mx2) / 2
                m_cy = (my1 + my2) / 2

                score = dist
                h_overlap = min(ax2, mx2) - max(ax1, mx1)
                is_above = a_cy < m_cy
                attach_overlaps_main = h_overlap > 0
                attach_near_right_edge = ax2 <= mx2 + typical_width * 0.3

                if attach_overlaps_main:
                    score *= 0.3
                elif attach_near_right_edge and ax1 >= mx1:
                    score *= 0.5

                if ax1 > mx2:
                    score *= self.attach_prefer_left * 1.5
                elif a_cx > m_cx:
                    score *= self.attach_prefer_left

                if a_cy > m_cy:
                    score *= self.attach_prefer_up

                if is_above and attach_overlaps_main:
                    score *= 0.5

                if score < best_score:
                    best_score = score
                    best_idx = i

            if best_idx is not None:
                m_box = main_boxes[best_idx]
                new_x1 = min(m_box[0], ax1)
                new_y1 = min(m_box[1], ay1)
                new_x2 = max(m_box[2], ax2)
                new_y2 = max(m_box[3], ay2)

                new_width = new_x2 - new_x1
                if new_width > self.max_width_ratio * typical_width:
                    max_w = int(self.max_width_ratio * typical_width)
                    if ax1 < m_box[0]:
                        new_x1 = min(m_box[0], ax1)
                        new_x2 = new_x1 + max_w
                    else:
                        new_x2 = max(m_box[2], ax2)
                        new_x1 = new_x2 - max_w

                main_boxes[best_idx] = [new_x1, new_y1, new_x2, new_y2]

        return [tuple(b) for b in main_boxes]

    def _suppress_overlapping(self, boxes: list[Box], iou_threshold: float = 0.3) -> list[Box]:
        """简单非极大值抑制，去掉重叠过多的框"""
        if not boxes:
            return boxes

        def area(b: Box) -> int:
            return (b[2] - b[0]) * (b[3] - b[1])

        def iou(b1: Box, b2: Box) -> float:
            x1 = max(b1[0], b2[0])
            y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2])
            y2 = min(b1[3], b2[3])
            iw = max(0, x2 - x1)
            ih = max(0, y2 - y1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            return inter / float(area(b1) + area(b2) - inter)

        def overlap_ratio(b1: Box, b2: Box) -> float:
            x1 = max(b1[0], b2[0])
            y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2])
            y2 = min(b1[3], b2[3])
            iw = max(0, x2 - x1)
            ih = max(0, y2 - y1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            return inter / float(area(b1))

        boxes_sorted = sorted(boxes, key=area, reverse=True)
        kept: list[Box] = []
        for b in boxes_sorted:
            discard = False
            for kb in kept:
                if overlap_ratio(b, kb) > 0.5 or iou(b, kb) > iou_threshold:
                    discard = True
                    break
            if not discard:
                kept.append(b)
        return kept

    def _clean_edge_artifacts(self, crop: np.ndarray, edge_threshold: int = 3) -> np.ndarray:
        """清理裁剪边缘的贴边残留"""
        h, w = crop.shape[:2]
        if h < 20 or w < 20:
            return crop

        b, g, r = cv2.split(crop)
        white_mask = (r > self.white_threshold) & (g > self.white_threshold) & (
            b > self.white_threshold
        )
        fg_mask = (~white_mask).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = crop.copy()
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if area < 50:
                continue

            touching_left = x <= edge_threshold
            touching_right = (x + cw) >= (w - edge_threshold)
            touching_top = y <= edge_threshold
            touching_bottom = (y + ch) >= (h - edge_threshold)
            if not (touching_left or touching_right or touching_top or touching_bottom):
                continue

            area_ratio = area / float(h * w)
            if area_ratio > 0.08:
                continue

            should_remove = False
            width_threshold = 0.15
            height_threshold = 0.15
            if touching_left and x == 0 and cw < w * width_threshold:
                should_remove = True
            if touching_right and (x + cw) >= w - 1 and cw < w * width_threshold:
                should_remove = True
            if touching_top and y == 0 and ch < h * height_threshold:
                should_remove = True
            if touching_bottom and (y + ch) >= h - 1 and ch < h * height_threshold:
                should_remove = True
            if area_ratio < 0.03 and (touching_left or touching_right or touching_top or touching_bottom):
                should_remove = True
            corners = (touching_left or touching_right) and (touching_top or touching_bottom)
            if corners and area_ratio < 0.05:
                should_remove = True

            if should_remove:
                cv2.drawContours(result, [cnt], -1, (255, 255, 255), -1)
                dilated_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(dilated_mask, [cnt], -1, 255, -1)
                dilated_mask = cv2.dilate(dilated_mask, kernel, iterations=2)
                result[dilated_mask > 0] = [255, 255, 255]

        return result

    def _trim_edges(self, img: np.ndarray, pixels: int = 2) -> np.ndarray:
        """直接裁掉边缘若干像素，进一步去残留"""
        h, w = img.shape[:2]
        if h <= pixels * 2 or w <= pixels * 2:
            return img
        return img[pixels : h - pixels, pixels : w - pixels].copy()

    def _convert_to_transparent(self, img: np.ndarray) -> np.ndarray:
        """将纯色背景转换为透明，保留前景"""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(fg_mask, contours, -1, 255, -1)

        corner_colors = [img[0, 0], img[0, w - 1], img[h - 1, 0], img[h - 1, w - 1]]
        bg_color = np.mean(corner_colors, axis=0).astype(np.uint8)
        color_diff = np.sqrt(np.sum((img.astype(float) - bg_color.astype(float)) ** 2, axis=2))
        bg_by_color = color_diff < 30
        fg_by_color = ~bg_by_color

        final_fg = (fg_mask > 0) | fg_by_color
        final_fg = final_fg.astype(np.uint8) * 255
        final_fg = cv2.morphologyEx(final_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        final_fg = cv2.morphologyEx(final_fg, cv2.MORPH_OPEN, kernel, iterations=1)

        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        bg_mask = final_fg == 0
        bgra[bg_mask, 3] = 0

        fg_float = final_fg.astype(float) / 255.0
        fg_blur = cv2.GaussianBlur(fg_float, (3, 3), 0)
        edge_region = (fg_blur > 0.1) & (fg_blur < 0.9)
        alpha_edge = (fg_blur * 255).astype(np.uint8)
        bgra[edge_region, 3] = alpha_edge[edge_region]
        return bgra

    def process_image(self, img: np.ndarray, debug: bool = False):
        """处理单张图片，返回裁剪结果与调试图"""
        h, w = img.shape[:2]
        fg = self._prepare_foreground(img)
        regions = self._find_all_regions(fg)
        mains, attaches = self._classify_regions(regions)
        merged_boxes = self._attach_regions(mains, attaches)
        final_boxes = self._suppress_overlapping(merged_boxes)
        final_boxes = sorted(final_boxes, key=lambda b: (b[1] // 50, b[0]))

        crops: list[np.ndarray] = []
        for box in final_boxes:
            x1 = max(0, box[0] - self.margin)
            y1 = max(0, box[1] - self.margin)
            x2 = min(w, box[2] + self.margin)
            y2 = min(h, box[3] + self.margin)
            crop = img[y1:y2, x1:x2].copy()
            crop = self._clean_edge_artifacts(crop)
            crop = self._trim_edges(crop, pixels=3)
            crop = self._convert_to_transparent(crop)
            crops.append(crop)

        debug_img = None
        if debug:
            debug_img = img.copy()
            for m in mains:
                x1, y1, x2, y2 = m.box
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            for a in attaches:
                x1, y1, x2, y2 = a.box
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            for box in final_boxes:
                x1 = max(0, box[0] - self.margin)
                y1 = max(0, box[1] - self.margin)
                x2 = min(w, box[2] + self.margin)
                y2 = min(h, box[3] + self.margin)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return crops, debug_img


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

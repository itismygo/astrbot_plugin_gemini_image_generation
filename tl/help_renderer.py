"""
帮助页面渲染模块
支持三种渲染模式：html (t2i)、local (Pillow)、text (纯文本)
"""

import io
import os
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from astrbot.api import logger


def get_template_path(
    templates_dir: str | Path,
    theme_settings: dict,
    extension: str = ".html",
) -> Path:
    """根据主题配置获取模板路径"""
    mode = theme_settings.get("mode", "cycle")
    cycle_config = theme_settings.get("cycle_config", {})
    single_config = theme_settings.get("single_config", {})

    template_filename = "help_template_light"

    if mode == "single":
        template_filename = single_config.get("template_name", "help_template_light")
    else:
        day_start = cycle_config.get("day_start", 6)
        day_end = cycle_config.get("day_end", 18)
        day_template = cycle_config.get("day_template", "help_template_light")
        night_template = cycle_config.get("night_template", "help_template_dark")

        current_hour = datetime.now().hour
        if day_start <= current_hour < day_end:
            template_filename = day_template
        else:
            template_filename = night_template

    if not template_filename.endswith(extension):
        template_filename += extension

    template_path = Path(templates_dir) / template_filename

    if not template_path.exists():
        logger.warning(f"模板文件不存在: {template_path}，回退到默认模板")
        template_filename = f"help_template_light{extension}"
        template_path = Path(templates_dir) / template_filename

    return template_path


def render_text(template_data: dict) -> str:
    """纯文本渲染"""
    return f"""🎨 {template_data.get("title", "Gemini 图像生成插件")}

基础指令:
• /生图 [描述] - 生成图像
• /快速 [预设] [描述] - 快速模式
• /改图 [描述] - 修改图像
• /换风格 [风格] - 风格转换
• /生图帮助 - 显示帮助

预设选项: 头像/海报/壁纸/卡片/手机/手办化

当前配置:
• 模型: {template_data.get("model", "N/A")}
• 分辨率: {template_data.get("resolution", "N/A")}
• API密钥: {template_data.get("api_keys_count", 0)}个
• LLM工具超时: {template_data.get("tool_timeout", 60)}秒

系统状态:
• 搜索接地: {template_data.get("grounding_status", "✗ 禁用")}
• 自动头像: {template_data.get("avatar_status", "✗ 禁用")}
• 智能重试: {template_data.get("smart_retry_status", "✗ 禁用")}"""


def _load_font(size: int):
    """加载字体"""
    builtin_font = Path(__file__).parent / "NotoSansSC-Regular.ttf"
    font_paths = [
        str(builtin_font),
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_local_pillow(
    templates_dir: str | Path,
    theme_settings: dict,
    template_data: dict,
) -> bytes:
    """使用 Pillow 本地渲染帮助页为图片（类似HTML样式）"""
    # 判断深色/浅色主题
    mode = theme_settings.get("mode", "cycle")
    is_dark = False
    if mode == "single":
        is_dark = "dark" in theme_settings.get("single_config", {}).get(
            "template_name", ""
        )
    else:
        cycle_config = theme_settings.get("cycle_config", {})
        day_start = cycle_config.get("day_start", 6)
        day_end = cycle_config.get("day_end", 18)
        current_hour = datetime.now().hour
        is_dark = not (day_start <= current_hour < day_end)

    # 颜色配置
    if is_dark:
        bg_color = (22, 27, 34)
        card_bg = (33, 38, 45)
        border_color = (48, 54, 61)
        text_primary = (230, 237, 243)
        text_secondary = (125, 133, 144)
        accent_color = (88, 166, 255)
    else:
        bg_color = (246, 248, 250)
        card_bg = (255, 255, 255)
        border_color = (208, 215, 222)
        text_primary = (31, 35, 40)
        text_secondary = (101, 109, 118)
        accent_color = (9, 105, 218)

    # 字体
    title_font = _load_font(24)
    section_font = _load_font(16)
    text_font = _load_font(14)

    # 布局参数
    width = 520
    padding = 24
    section_gap = 20
    line_height = 24
    section_title_height = 32

    # 准备内容
    title = template_data.get("title", "Gemini 图像生成插件")
    config_items = [
        f"模型: {template_data.get('model', 'N/A')}",
        f"分辨率: {template_data.get('resolution', 'N/A')}",
        f"API密钥: {template_data.get('api_keys_count', 0)}个",
        f"搜索接地: {template_data.get('grounding_status', '-')}",
        f"自动头像: {template_data.get('avatar_status', '-')}",
        f"智能重试: {template_data.get('smart_retry_status', '-')}",
        f"LLM超时: {template_data.get('tool_timeout', 60)}秒",
    ]
    commands = [
        "/生图 [描述] - 生成图像",
        "/改图 [描述] - 修改图像",
        "/换风格 [风格] - 风格转换",
        "/切图 - 切割表情包",
        "/生图帮助 - 显示帮助",
    ]
    quick_modes = [
        "/快速 头像 - 1K 1:1",
        "/快速 海报 - 2K 16:9",
        "/快速 壁纸 - 4K 16:9",
        "/快速 手办化 - 2K 3:2",
    ]

    # 计算高度
    total_height = padding * 2 + 50  # 标题区
    total_height += section_title_height + len(config_items) * line_height + section_gap
    total_height += section_title_height + len(commands) * line_height + section_gap
    total_height += section_title_height + len(quick_modes) * line_height + padding

    # 创建图片
    img = Image.new("RGB", (width, total_height), bg_color)
    draw = ImageDraw.Draw(img)

    # 绘制卡片背景（圆角矩形）
    card_margin = 12
    draw.rounded_rectangle(
        [card_margin, card_margin, width - card_margin, total_height - card_margin],
        radius=12,
        fill=card_bg,
        outline=border_color,
    )

    y = padding + card_margin

    # 标题
    draw.text((padding + card_margin, y), title, font=title_font, fill=text_primary)
    y += 40

    # 分隔线
    draw.line(
        [(padding + card_margin, y), (width - padding - card_margin, y)],
        fill=border_color,
        width=1,
    )
    y += section_gap

    def draw_section(section_title: str, items: list[str]):
        nonlocal y
        # 标题栏
        draw.rectangle(
            [padding + card_margin, y, padding + card_margin + 4, y + 16],
            fill=accent_color,
        )
        draw.text(
            (padding + card_margin + 12, y - 2),
            section_title,
            font=section_font,
            fill=text_primary,
        )
        y += section_title_height
        # 内容
        for item in items:
            draw.text(
                (padding + card_margin + 12, y),
                item,
                font=text_font,
                fill=text_secondary,
            )
            y += line_height
        y += section_gap // 2

    draw_section("当前配置", config_items)
    draw_section("基础指令", commands)
    draw_section("快速模式", quick_modes)

    # 输出为 PNG bytes
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()

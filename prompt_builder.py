"""
Explicit prompt builder for T2I - No implicit semantic injection
All visual decisions controlled via Control Pad
"""

# Detail Level Mapping - 控制信息密度
DETAIL_LEVELS = {
    "low": "",
    "medium": "",
    "high": ""
}

def build_prompt(
    concept_keywords: list,
    style: str = "",
    detail_level: str = "medium",
    staff_suffix: str = "",
    staff_negative: str = None,
    reference_images: list | None = None,
) -> dict:
    """
    构建显式可控的提示词（无隐式推断）

    Args:
        concept_keywords: SLM提供的概念关键词（只读，"画什么"）
        style: 自由文本风格描述（由人员输入）
        detail_level: 细节层次 low/medium/high（教学抽象度与标注密度）
        staff_suffix: 工作人员控制的正向后缀（"怎么画"）
        staff_negative: 工作人员控制的完整负向提示词（可完全覆盖；空则不注入）
        reference_images: 参考图路径列表（相对仓库）

    Returns:
        {
            "positive": str,
            "negative": str,
            "structure": dict
        }
    """
    style_base = style.strip()
    concept_str = ", ".join(concept_keywords) if concept_keywords else ""
    detail_str = DETAIL_LEVELS.get(detail_level, DETAIL_LEVELS["medium"])
    staff_str = staff_suffix.strip() if staff_suffix else ""

    positive_parts = [
        style_base,           # ① Style (自由文本)
        concept_str,          # ② SLM Concepts
        detail_str,           # ③ Detail Level
        staff_str             # ④ Staff Suffix
    ]
    positive = ", ".join(filter(None, positive_parts))

    # 负向提示词：空则不注入
    negative = staff_negative.strip() if staff_negative else ""

    return {
        "positive": positive,
        "negative": negative,
        "structure": {
            "style": style_base,
            "concept_keywords": concept_keywords,
            "detail_level": detail_level,
            "staff_suffix": staff_suffix,
            "staff_negative": negative,
            "reference_images": reference_images or []
        }
    }

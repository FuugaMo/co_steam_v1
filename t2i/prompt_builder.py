"""
Explicit prompt builder for T2I - No implicit semantic injection
All visual decisions controlled via Control Pad
"""

def build_prompt(
    concept_keywords: list,
    style: str = "",
    staff_suffix: str = "",
    staff_negative: str = None,
    reference_images: list | None = None,
) -> dict:
    """
    构建显式可控的提示词（无隐式推断）

    Args:
        concept_keywords: SLM提供的概念关键词（只读，"画什么"）
        style: 自由文本风格描述（由人员输入）
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
    staff_str = staff_suffix.strip() if staff_suffix else ""

    positive_parts = [
        style_base,           # ① Style (自由文本)
        concept_str,          # ② SLM Concepts
        staff_str             # ③ Staff Suffix
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
            "staff_suffix": staff_suffix,
            "staff_negative": negative,
            "reference_images": reference_images or []
        }
    }

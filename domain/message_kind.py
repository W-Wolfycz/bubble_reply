from __future__ import annotations


def is_noise_message(
    *,
    message_type_value: str,
    post_type: str,
    has_poke: bool,
) -> bool:
    """判断入站事件是否为“非对话消息”（系统事件 / notice / request / 戳一戳）。

    这类消息本身不会造成回复指向歧义，智能引用观察器默认不计入打断序号；
    是否因“累积过多”而升级为打断由 SmartReplyTracker 单独判定。
    """
    if message_type_value == "OtherMessage":
        return True
    if post_type in {"notice", "request"}:
        return True
    return has_poke

import re

from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm import lm_utils
from app.conf.lm_config import lm_config
from app.utils.sse_utils import push_to_session
from app.utils.task_utils import set_task_result
from app.clients.mongo_history_utils import save_chat_message


def _push_answer(session_id: str, answer: str, is_stream: bool):
    """将答案输出：流式走 SSE，非流式走 task_result。"""
    if is_stream:
        push_to_session(session_id, "final_answer", {"answer": answer})
    else:
        set_task_result(session_id, "answer", answer)


def _extract_image_urls(docs: list[dict]) -> list[str]:
    """从文档文本中提取图片 URL。"""
    urls = []
    for d in docs:
        text = d.get("text", "")
        urls.extend(re.findall(r'https?://[^\s]+(?:png|jpe?g|gif|webp|bmp|svg)[^\s]*', text, re.IGNORECASE))
    return urls


def _generate_answer_from_docs(state: dict, session_id: str, is_stream: bool):
    """根据 reranked_docs 生成答案并推送。"""
    docs = state.get("reranked_docs", []) or []
    context = "\n\n".join(
        f"source: {d.get('source', '')}\nchunk_id: {d.get('chunk_id', '')}\nurl: {d.get('url', '')}\ntitle: {d.get('title', '')}\nscore: {d.get('rerank_score', d.get('rrf_score', ''))}\ntext: {d.get('text', '')}"
        for d in docs[:5]
    )
    query = state.get("rewritten_query") or state.get("original_query", "")
    history = state.get("history", [])
    history_text = "\n".join(
        f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('text', '')}"
        for m in (history[-4:] if history else [])
    )
    item_names = ", ".join(state.get("item_names", [])) or "未知"
    prompt = load_prompt("answer_out", context=context, history=history_text, item_names=item_names, question=query)
    llm = lm_utils.get_llm_client(lm_config.llm_model, json_mode=False)

    if is_stream:
        full_answer = ""
        for chunk in llm.stream([{"role": "user", "content": prompt}]):
            delta = chunk.content or ""
            if delta:
                full_answer += delta
                push_to_session(session_id, "delta", {"delta": delta})
        answer = full_answer.strip()
        state["answer"] = answer
        _push_answer(session_id, answer, True)
    else:
        resp = llm.invoke([{"role": "user", "content": prompt}])
        answer = resp.content.strip()
        state["answer"] = answer
        _push_answer(session_id, answer, False)

    image_urls = _extract_image_urls(docs)
    state["image_urls"] = image_urls
    if image_urls and is_stream:
        push_to_session(session_id, "final", {"image_urls": image_urls})


def node_answer_output(state):
    """输出节点：生成并推送答案。"""
    session_id = state.get("session_id")
    answer = state.get("answer", "")
    is_stream = state.get("is_stream", False)

    if answer:
        _push_answer(session_id, answer, is_stream)
        return state

    _generate_answer_from_docs(state, session_id, is_stream)
    answer = state.get("answer", "")

    if answer:
        save_chat_message(session_id, "assistant", answer,
                          item_names=state.get("item_names"),
                          image_urls=state.get("image_urls"))
    return state


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_answer_output 本地测试")
    print("=" * 50)

    # 1. 构造模拟数据
    # 模拟重排序后的文档列表 (reranked_docs)
    # 包含：本地文档（带Markdown图片）、联网结果（带URL字段）、纯文本文档
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "source": "local",
            "title": "HAK 180 烫金机操作手册_v2.pdf",
            "score": 0.95,
            "text": """
            HAK 180 烫金机的操作面板位于机器正前方。
            开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
            具体的操作面板布局请参考下图：
            ![操作面板布局图](http://local-server/images/panel_view.jpg)

            如果是进行局部烫金，请调节侧面的旋钮。
            ![侧面旋钮细节](http://local-server/images/knob_detail.png)
            """
        },
        {
            "chunk_id": None,
            "source": "web",
            "title": "HAK 180 常见故障排除 - 官网",
            "score": 0.88,
            "url": "http://example.com/hak180_troubleshooting.jpeg",  # 这是一个直接指向图片的URL（虽然少见，但用于测试提取）
            "text": "如果机器无法加热，请检查保险丝是否熔断..."
        },
        {
            "chunk_id": "local_102",
            "source": "local",
            "title": "安全注意事项",
            "score": 0.82,
            "text": "操作时请务必佩戴隔热手套，避免高温烫伤。"
        }
    ]

    # 模拟历史记录
    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？"},
        {"role": "assistant", "text": "您好！请问您具体指的是哪一款机器？"},
        {"role": "user", "text": "HAK 180 烫金机"}
    ]

    # 模拟输入状态
    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK 180 烫金机"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,  # 测试非流式
        # "is_stream": True, # 若要测试流式，需确保 SSE 环境或 mock 相关函数
        "answer": None  # 初始无答案
    }

    try:
        # 运行节点
        result = node_answer_output(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")

        # 1. 验证 Prompt 构建
        if "prompt" in result:
            print(f"[PASS] Prompt 构建成功 (长度: {len(result['prompt'])})")
            # print(f"Prompt 预览:\n{result['prompt'][:200]}...")
        else:
            print("[FAIL] Prompt 未构建")

        # 2. 验证答案生成
        answer = result.get("answer")
        if answer and len(answer) > 10:
            print(f"[PASS] 答案生成成功 (长度: {len(answer)})")
            print(f"答案预览: {answer[:50]}...")
        else:
            print(f"[WARN] 答案生成可能异常 (Content: {answer})")

        # 3. 验证图片提取
        # 我们期望提取到 3 张图片：
        # 1. http://local-server/images/panel_view.jpg (来自 local_101)
        # 2. http://local-server/images/knob_detail.png (来自 local_101)
        # 3. http://example.com/hak180_troubleshooting.jpeg (来自 web 结果的 url 字段)

        # 注意：这里我们没办法直接从 result state 里拿到 image_urls，因为它是作为 SSE 推送出去的，或者存库了
        # 但我们可以通过日志观察 _extract_images_from_docs 的输出
        # 如果需要验证，可以临时修改 node_answer_output 返回 image_urls
        print("\n[INFO] 请检查上方日志中是否包含 '图片提取完成' 及以下 URL:")
        print(" - http://local-server/images/panel_view.jpg")
        print(" - http://local-server/images/knob_detail.png")
        print(" - http://example.com/hak180_troubleshooting.jpeg")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
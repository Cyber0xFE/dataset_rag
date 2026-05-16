import json

from app.core.logger import logger
from app.import_process.agent.main_graph import app
from app.import_process.agent.state import create_default_state

logger.info("===== 开始测试 =====")

initial_state = create_default_state(local_file_path="E:/BaiduNetdiskDownload/05_项目_掌柜智库/掌柜智库项目/资料/doc/万用表RS-12的使用.pdf",
                                     is_pdf_read_enabled=True)
final_state = None

# 只输出更最终的状态值（字典形式），不包含节点名称、执行日志、元数据等额外信息
for event in app.stream(initial_state):
    for key, value in event.items():
        logger.info(f"节点: {key}")
        final_state = value

# 格式化输出最终状态
logger.info(f"最终状态: {json.dumps(final_state, indent=4, ensure_ascii=False)}")

logger.info("图结构:")
# uv add grandalf
app.get_graph().print_ascii()

logger.info("===== 测试结束 =====")

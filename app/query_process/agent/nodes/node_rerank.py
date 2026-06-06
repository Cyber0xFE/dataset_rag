import sys

from app.core.logger import logger
from app.utils.task_utils import add_running_task, add_done_task
from app.lm.reranker_utils import get_reranker_model


def _merge_docs(rrf_chunks: list, web_docs: list) -> list[dict]:
    """将 rrf_chunks 和 web_search_docs 合并为统一结构。"""
    merged = []
    for c in rrf_chunks:
        merged.append({
            "text": c.get("content", ""),
            "doc_id": c.get("chunk_id", ""),
            "chunk_id": c.get("chunk_id", ""),
            "title": c.get("title", ""),
            "url": "",
            "source": "local",
        })
    for d in web_docs:
        merged.append({
            "text": d.get("snippet", "") or d.get("content", ""),
            "doc_id": d.get("url", "") or d.get("chunk_id", ""),
            "chunk_id": d.get("url", "") or d.get("chunk_id", ""),
            "title": d.get("title", ""),
            "url": d.get("url", ""),
            "source": "web",
        })
    return merged


def _rerank_docs(docs: list[dict], query: str):
    """使用重排序模型对文档列表进行重排序。"""
    if not query or not docs:
        return
    reranker = get_reranker_model()
    scores = reranker.compute_score([(query, doc["text"]) for doc in docs], normalize=True)
    for doc, score in zip(docs, scores):
        doc["rerank_score"] = round(float(score), 4)
    docs.sort(key=lambda d: d["rerank_score"], reverse=True)
    logger.info(f"rerank 重排序完成，输入 {len(docs)} 条")


def node_rerank(state):
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    merged = _merge_docs(
        state.get("rrf_chunks", []) or [],
        state.get("web_search_docs", []) or [],
    )

    # 使用重排序模型对merged进行重排序
    query = state.get("rewritten_query") or state.get("original_query", "")
    _rerank_docs(merged, query)

    state["reranked_docs"] = merged

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rerank 本地测试")
    print("=" * 50)

    # 1. 模拟数据
    # 1.1 RRF 本地文档数据
    mock_rrf_chunks = [{'chunk_id': 466582515052979214, 'content': '使用方法：旋转拨盘选择测量档位，将表笔接入被测电路。',
                        'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf',
                        'rrf_score': 0.0328, 'title': '使用方法'},
                       {'chunk_id': 466582515052979220, 'content': '使用方法：旋转拨盘选择测量档位，将表笔接入被测电路。',
                        'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf',
                        'rrf_score': 0.0318, 'title': '使用方法'},
                       {'chunk_id': 466582515052979218, 'content': '产品概述：万用表RS-12是一款高精度数字万用表。',
                        'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf',
                        'rrf_score': 0.0311, 'title': '产品概述'},
                       {'chunk_id': 466582515052979219, 'content': '技术参数：直流电压量程0-600V，交流电压量程0-600V。',
                        'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf',
                        'rrf_score': 0.0306, 'title': '技术参数'},
                       {'chunk_id': 466582515052979221, 'content': '安全警告：测量电压时请勿超过额定输入值，防止触电。',
                        'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf',
                        'rrf_score': 0.0305, 'title': '安全警告'},
                       {'chunk_id': 466582515052979226, 'content': '使用方法：旋转拨盘选择测量档位，将表笔接入被测电路。',
                        'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf',
                        'rrf_score': 0.0159, 'title': '使用方法'},
                       {'chunk_id': 466582515052979232, 'content': '使用方法：旋转拨盘选择测量档位，将表笔接入被测电路。',
                        'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf',
                        'rrf_score': 0.0156, 'title': '使用方法'},
                       {'chunk_id': 466582515052979239, 'content': '使用方法：旋转拨盘选择测量档位，将表笔接入被测电路。',
                        'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf',
                        'rrf_score': 0.0154, 'title': '使用方法'},
                       {'chunk_id': 466582515052979224, 'content': '产品概述：万用表RS-12是一款高精度数字万用表。',
                        'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf',
                        'rrf_score': 0.0145, 'title': '产品概述'},
                       {'chunk_id': 466582515052979225, 'content': '技术参数：直流电压量程0-600V，交流电压量程0-600V。',
                        'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf',
                        'rrf_score': 0.0143, 'title': '技术参数'}]

    # 1.2 MCP 联网搜索数据
    mock_web_docs = [{'hostlogo': '', 'hostname': '无',
                      'snippet': '不同使用需求下万用表的使用方法有所区别,以下针对常见的几种使用场景进行了万用表的使用方法介绍:电压测量:并联接入,分清交直流 电压测量是最常用场景,如检测电池电压、家用插座电压,核心原则是“并联接入电路”。直流电压(如电池、电路板直流供电):将旋钮转...”档,红表笔插“vΩma”孔;大电流(>200ma)转至对应大电流档,红表笔插“10a”孔。将万用表串联接入断开处:红表笔接电路电流流出的一端(靠近电源正极),黑表笔接电流流入的一端(靠近电源负极),接通电源后,屏幕显示数值即为电流值。万用表的使用方法',
                      'title': '万用表', 'url': 'https://rsonline.cn/web/p/multimeters/7904104/'}, {
                         'hostlogo': 'http://t8.baidu.com/it/u=3027949672,2003644166&fm=195&app=88&f=JPEG?w=200&h=200&s=76D1A8764C706C920AEEFAEE0200501D',
                         'hostname': '无',
                         'snippet': '为了您的安全,请在使用本仪表之前仔细阅读该手册: 使用本表时,将输入的测量值超出其所允许的量程范围. 输入量程 功能最大输入 交/直流电压直流/交流电压600v 直流/交流电压直流/交流电压600v,200vrms用于200mv量程 ma直流200m...0\uf06da 直流电流 200ma100\uf06da\uf0b1(1.2%reading+2digits) •)))蜂鸣指示 二极管测试指示10a10ma\uf0b1(2.0%reading+2digits) 200\uf0570.1\uf057',
                         'title': '数字万用表使用与安全指南.pdf 21页VIP',
                         'url': 'https://m.book118.com/html/2026/0528/5144343140013221.shtm'},
                     {'hostlogo': 'https://m.book118.com/favicon.ico', 'hostname': 'E书联盟',
                      'snippet': '万用表使用说明书 1、适用于1至10安培的沟通和 1、适用于1至10安培的沟通和 直流电电流测量的输入端子。 、适用于至400毫安的 安测量的输入端子。 3 、适用于全部测式的公 CAT1 、适用于电压、电阻、 通断性、二极管、电容 测量的输入端子 ...要的电路测试点,测量电阻。 〔四〕通断性的推断:中选中了电阻模式,按两次黄色按钮可启动通断性蜂鸣器。假设电阻不超过50欧母,蜂鸣器会发出连续音,说明短路。假设电表读数为0,刚表示是开路。 测量二极管 在测量二极管时,同样必需将电源关闭,以即电容器放电。',
                      'title': '万用表使用说明书.docx 6页',
                      'url': 'https://max.book118.com/html/2025/0529/5021003100012213.shtm'}, {
                         'hostlogo': 'https://mbs1.bdstatic.com/searchbox/mappconsole/image/20221223/5a9c40b3-d1a4-400d-8af8-85383c768ef4.png',
                         'hostname': '汽车之家',
                         'snippet': '电压测量档是最常用的功能之一。在检测电瓶电压时,应选择直流电压档(DCV),量程通常设为20V。将红表笔接触电瓶正极,黑表笔接触负极,正常值应在12.6V至14.4V之间。若读数低于12V,可能表明电瓶存电不足;若高于15V,则需检查发电机调节器工作...使用过程中,请始终确认档位与被测参数匹配,切勿在电流档位下直接测量电压,否则极易造成仪表内部电路烧毁。建议每次测量前,先估算数值范围,选择合适的量程,避免过载。养成断电测量电阻、表笔插孔正确、量程由大到小调试的良好习惯,是确保测量准确与设备安全的关键。',
                         'title': '汽修数字万用表档位怎么用?', 'url': 'https://www.autohome.com.cn/ask/25254123.html'},
                     {
                         'hostlogo': 'https://ss2.baidu.com/6ONYsjip0QIZ8tyhnq/it/u=2005731947,4139443793&fm=195&app=88&f=JPEG?w=200&h=200',
                         'hostname': 'CSDN软件开发网',
                         'snippet': '本文介绍了万用表的基本结构和使用方法,包括表笔的插入、量程的选择以及测量的正确姿势。强调了在测量电流、电压和电阻时如何选择合适的量程,以及安全操作的注意事项,如不带电测量电阻和电容放电。此外,还提供了万用表使用口诀,帮助读者更好地掌握使用技巧。  哈...流档(A~)、直流电流档(A-)、交流电压档(V~)、直流电压档(V-)、电阻档(Ω)、蜂鸣档。 拨动到要使用的档位以后,还要选择更精确的量程。以交流电压档为例,又分为2V,20V,200V,750V四种,选择的原则是:选大于且最接近与实际示数的量程。',
                         'title': '电工基础知识分享(四):图文并茂,史上最全万用表使用手册',
                         'url': 'https://blog.csdn.net/m0_61241991/article/details/129322956'}, {
                         'hostlogo': 'https://mbs1.bdstatic.com/searchbox/mappconsole/image/20230210/1990c335-3550-4d29-bd75-54f81b936fb0.png',
                         'hostname': '无',
                         'snippet': '万用表使用说明 万用表使用说明书 万用表的详细使用方法 万用表使用方法 万用表简单使用方法 万用表操作规程 万用表操作规程 万用表的使用方法详细介绍 一键批量下载以上文档 万用表操作说明 万用表使用说明 万用表使用说明书 万用表的详细使用方法 万用表使用方法 万用表简单使用方法 页数 9页 立即下载',
                         'title': '万用表操作说明', 'url': 'https://abg.baidu.com/view/1a81c460ddccda38376bafa6'}, {
                         'hostlogo': 'https://mbs1.bdstatic.com/searchbox/mappconsole/image/20230210/1990c335-3550-4d29-bd75-54f81b936fb0.png',
                         'hostname': '无',
                         'snippet': '万用表使用说明书 个人求职简历 简历通用求职简历 求职简历模板 应届生求职简历模板 销售工作总结 评分 4.58 大小 827kb 页数 5页 立即下载 点赞 收藏 分享 举报',
                         'title': '万用表使用说明书', 'url': 'https://abg.baidu.com/view/03eb0e0cf12d2af90242e61d'},
                     {'hostlogo': '', 'hostname': '福禄克官方网站',
                      'snippet': 'Fluke 15B MAX 经济型数字万用表 全新升级15B+ 误操作报警,特尖表笔,任意键唤醒 FLK-EV KIT 新能源工具套装 新能源工具套装,应用于新能源汽车售后维修或三电比如动力电池维修等场景。帮助现场工具专业化、标准化、精益化。 Flu...用于电气及机电设备的巡检,外出检测服务,暖通空调检查,产品研发及品质管控等,大大提高检测效率。 Fluke 963A/963B/963C/963D 联网型温湿度记录仪 4G/WIFI 双版本可选,无需组网,独立工作。长期记录,可多平台远程读取实时数据。',
                      'title': 'Fluke 12E+ 多功能万用表',
                      'url': 'https://www.fluke.com.cn/product/电气测试/数字万用表/fluke-12e-plus'}, {
                         'hostlogo': 'https://ss0.baidu.com/6ONWsjip0QIZ8tyhnq/it/u=493147230,3096476255&fm=195&app=88&f=JPEG?w=200&h=200',
                         'hostname': '知乎',
                         'snippet': '一、电压的测量 1、直流电压的测量,如电池、随身听电源等。首先将黑表笔插进“com”孔,红表笔插进“V Ω”。把旋钮选到比估计值大的量程(注意:表盘上的数值均为最大量程,“V-”表示直流电压档,“V～”表示交流电压档,“A”是电流档),接着把表笔接电...管……测量时,表笔位置与电压测量一样,将旋钮旋到“ ”档;用红表笔接二极管的正极,黑表笔接负极,这时会显示二极管的正向压降。肖特基二极管的压降是0.2V左右,普通硅整流管(1N4000、1N5400系列等)约为0.7V,发光二极管约为1.8～2.3V。',
                         'title': '万用表怎么使用?',
                         'url': 'http://www.zhihu.com/question/294494104/answer/2518404606'},
                     {'hostlogo': 'https://baijiahao.baidu.com/favicon.ico', 'hostname': '百家号',
                      'snippet': '一、电阻测量原理:理解欧姆定律的实践应用 万用表电阻档的核心原理基于欧姆定律(R=V/I),通过内部电路构建恒流源或固定电压源,结合表头电阻与被测电阻的串联/并联关系,推导出未知电阻值。 1. 数字万用表:恒流源+电压测量 数字万用表内部集成恒流源电...为电源,内部标准电阻(Rₛ)与表头线圈构成闭合回路。接入被测电阻(Rₓ)后,回路总电阻变为Rₛ+Rₓ,电流减小导致表针偏转角度降低。表盘欧姆刻度根据电流大小非线性标注,直接指示电阻值。例如,×1k档下,指针指在15位置,实际电阻为15×1k=15kΩ。',
                      'title': '万用表电阻档使用方法全解析:从基础操作到故障诊断',
                      'url': 'https://baijiahao.baidu.com/s?id=1848331429506804548&wfr=spider&for=pc'}]

    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？",  # 查询意图：想了解这两个算法
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_rerank(mock_state)
        reranked = result.get("reranked_docs", [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入文档总数: {len(mock_rrf_chunks) + len(mock_web_docs)}")
        print(f"输出文档总数: {len(reranked)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(reranked, 1):
            print(f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:20]}...")

        # 验证逻辑：
        # 预期 "local_1", "local_2", "Rerank技术详解" 分数较高
        # 预期 "local_3", "无关网页" 分数较低，可能被截断或排在最后

        top1_score = reranked[0].get("score")
        if top1_score > 0:
            print("\n[PASS] Rerank 打分正常")
        else:
            print("\n[FAIL] Rerank 打分异常 (均为0或负数)")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")

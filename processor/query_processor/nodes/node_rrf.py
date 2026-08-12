from typing import List, Tuple, Any, Dict

from processor.import_processor.base import BaseNode
from processor.query_processor.logger import logger
from processor.query_processor.state import QueryGraphState
from utils.json_format_utils import format_json


class NodeRrf(BaseNode):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web）进行加权融合排序。
    """

    name : str = "node_rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:

        logger.info(f"【{self.name}】节点逻辑")
        # 1.获取各路的搜索结果（排除网络搜索：rerank节点做）
        embedding_search_list = [
            doc.get("entity") for doc in state.get("embedding_chunks" ,[]) if isinstance(doc,dict)
            ]
        hyde_embedding_search_list = [
            doc.get("entity") for doc in state.get("hyde_embedding_chunks",[]) if isinstance(doc,dict)
        ]

        # 2. 为不同路的搜索结果设置不同发权重
        rrf_inputs = [
            (embedding_search_list,1.0),
            (hyde_embedding_search_list,1.0)
        ]

        # 3. 利用RRF的计算公式取获取到所有路查询到chunk对应的score
        rrf_merge_results = self._rrf_merge(rrf_inputs)

        # 4. 获取rrf_chunks(只取文档，不要分数)
        rrf_chunks = [doc for doc,_ in rrf_merge_results]

        # 5. 更新state
        state["rrf_chunks"] = rrf_chunks

        # 6. 返回state
        return state

    def _rrf_merge(self, rrf_inputs,k:int=60,max_results:int=None) -> List[Tuple[Dict[str,Any],float]]:
        """
            利用 RRF 公式计算每一个文档的总得分
            :param rrf_inputs:  列表，每个元素是(各路的搜索结果列表, 权重)的元组
            :param k:           平滑参数(RFF常数)，通常取 60
            :param max_results: 合并完之后返回的文档数，None 表示全部
            :return:            合并以及排序后的文档列表，[(元素, RRF 得分), ...] 按得分降序
        """
        chunk_scores = {}   #存放所有chunk的RRF计算后的分数值
        chunk_data = {}   #存放所有chunk的文档数据

        for rrf_input, weight in rrf_inputs:
            #第一路：embedding_search_list,第几路
            for rank ,doc in enumerate(rrf_input,start=1):
                # rank：当前这一路里面的排名,从1开始
                chunk_id = doc.get("chunks_id")
                if not chunk_id:
                    continue  # 跳过没有id的文档
                #RRF 公式：score +=weight / (k+rank)
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id,0.0) + weight/(k+rank)
                # 使用 setdefault 保留首次遇到的文档版本(只记录第一次)
                chunk_data.setdefault(chunk_id,doc)

        #按得分降序排序
        #(chunk_data[cid] chunk_id对应的文档chunks
        unsorted_results = [(chunk_data[cid],score) for cid,score in chunk_scores.items()]
        # 排序时看每个元素的第 2 个值（也就是分数）,倒序
        sorted_results = sorted(unsorted_results,key=lambda x:x[1],reverse=True)

        return sorted_results[:max_results] if max_results else sorted_results


if __name__ =="__main__":
    # 模拟两路检索结果
    mock_state = {
        "embedding_chunks": [
            {  "entity": {
                "chunks_id": 468191558938394830,
                "content": "\n•\t本设备通过 AC 220 V-240 V 50/60 Hz 电源供电。\n\n请勿将本设备连接到直流电源或逆变器（直流交流变换器）。存在火灾或触电的风险。\n\n•\t请勿用湿手触摸插头。这样可能导致触电。如果不确定您拥有哪种类型的电源，请联系合格的电工。\n\n•\t始终确保插头已完全插入。如果电源线磨损或损坏，请勿使用设备或用手触摸电源线。\n\n•\t设备内部有高压电极。\n\n先拔掉电源线，再清洁设备内部。拔出电源线时，不要拉电线，而是捏住插头往外拔。存在发生火灾、触电或设备故障的风险。\n\n•\t请勿将任何物体压在电源线上。\n\n•\t请勿将本设备放在人们可能踏过电源线的位置。\n\n•\t请勿将本设备放置在会使得拉伸或拉紧电源线的位置，否则电源线可能会磨损或损坏。\n\n•\t始终确保插头已完全插入。如果电源线磨损或损坏，请勿使用设备或用手触摸电源线。如果拔出设备的电源插头，请勿触摸损坏 磨损的部分。\n\n•\t请勿让设备压在电源线上。\n\n•\t请勿在雷暴天气期间使用本设备。存在闪电导致触电的潜在风险。\n\n•\t请勿使用任何非指定的电缆。否则可能导致火灾或人员受伤。必须按照 正确安装。\n\n•\t请勿让任何金属硬件或任何类型的液体落在设备的电源插头上。否则可能导致触电或火灾。\n\n•\tBrother 强烈建议您不要使用任何类型的延长线。\n\n•\t定期拔出电源插头进行清洁。使用干布清洁插头插脚根部以及插脚之间的位置。如果电源插头长时间插入在电源插座中，灰尘会堆积在插头插脚周围，这可能会导致短路，从而引起火灾。\n\n•\t本设备装有接地的插头。此插头只能插入接地的电源插座中。这是一项安全功能。如果您无法将插头插入到插座中，请让电工更换过时的插座。请勿试图破坏接地插头的作用。\n\n不遵守说明和警告可能导致人员中度或严重受伤。遵守这些指引以避免人员受伤。\n",
                "item_name": "HAK180烫金机"
            }},
            {"entity": {
                "chunks_id": 468191558938394831,
                "content": "\n•\t将本设备放置在平整、水平且稳定的表面上（如桌面），避免震动和冲击。\n\n•\t将本设备放置在通风良好的环境中。\n\n•\t为了防止人员受伤，请谨慎操作，避免将手指放置在图中所示的区域中。\n\n![禁止将手指伸入设备内部齿轮/传动机构区域（图中箭头所指部位）](http://192.168.204.130:9002/think-tank/upload-images/hak180产品安全手册/c61a7f4e923881679f747508ae309c39dc221685344b068009256b1b3a40cc00.jpg)\n\n![禁止将手指伸入设备顶部开口区域（如进纸/出纸口或盖板下方）](http://192.168.204.130:9002/think-tank/upload-images/hak180产品安全手册/5067b2891ca4f761e2874921e0eb433aa742afbf38ca8dc509afecbf0aa6a6b5.jpg)\n",
                "item_name": "HAK180烫金机"
            }},
            {"entity": {
                "chunks_id": 468191558938394826,
                "content": "![HAK 180 烫金机产品安全手册封面条形码（含型号 D01WD7001-00 及品牌 SCHN）](http://192.168.204.130:9002/think-tank/upload-images/hak180产品安全手册/677a08ee041965bbbdb6b483d6c17d5aaa36a26b6dc96870a2019f0307b8616f.jpg)  \nD01WD7001-00\n\nSCHN\n",
                "item_name": "HAK180烫金机"
            }},
        ],
        "hyde_embedding_chunks": [
            { "entity": {
                "item_name": "HAK180烫金机",
                "chunks_id": 468191558938394829,
                "content": "\n•\t请先阅读这本手册，再尝试操作本设备或尝试进行任何维护。不按照这些说明操作可能会提高发生人员受伤或财产损坏（包括火灾、触电、烧伤或窒息所致）的风险。对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。\n\n•\t请勿在未去除所有包装材料的情况下使用本设备，包括本设备内部的任何附加的包装材料。否则可能会产生火灾的风险。\n\n•\t请勿拆解本设备。拆解本设备可能会导致火灾或触电。\n\n•\t请勿尝试自行维修本设备。打开或拆下盖子可能使您接触到危险电压点以及带来其他风险，并且可能使您的保修失效。对于所有维修事宜，请联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t请在以下环境使用本设备：温度保持在 10 °C 和 32 °C 之间，湿度保持在 20% 和 80% 之间，无冷凝。\n\n•\t请勿使本设备受到阳光直射、过热、接触明火、腐蚀性气体、湿气或灰尘。否则可能产生触电、短路或火灾的风险，从而导致损坏设备和/或导致设备无法运行。\n\n•\t请勿将设备放在加热器、空调、电风扇或水附近。\n\n否则当水（包括加热 空调 通风设备所产生的冷凝水）接触本设备时可能产生短路或火灾的风险。\n\n•\t如果设备变得异常高温、冒烟、产生任何强烈味道，或者如果您意外在设备上倒入任何液体，请立即从电源插座拔掉设备的插头。请联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n如果设备跌落或者已损坏，则有触电的可能性。请从电源插座中拔掉设备的插头，然后联系 呼叫中心或您当地的 经销商。\n\n•\t如果水、其他液体或金属物体进入设备内部，请立即从电源插座中拔掉设备的插头，然后联系 Brother 呼叫中心或您当地的 Brother经销商。\n\n•\t请勿在卡纸或有纸张散落在设备内部的情况下尝试使用本设备。纸张与定影单元长时间接触可能导致火灾。\n\n请勿使用任何易燃物品、任何类型的喷雾剂包含酒精或氨水的有机溶剂/液体来清洁本设备的内部或外部。否则可能导致火灾。请改用无绒干抹布。有关如何清洁本设备的说明，请参阅 。\n\n•\t请勿将本设备放在化学品附近，或者将本设备放置在可能会泼溅到化学品的位置。万一化学品接触本设备，则存在火灾或触电的风险。特别是有机溶剂或液体（如苯、油漆稀释剂、抛光剂或除臭剂）可能导致塑料盖和/或电缆溶解或分解，从而产生火灾或触电的风险。这些化学品或其他化学品可能导致本设备故障或褪色。\n\n•\t本设备的包装中使用了塑料袋。塑料袋并不是玩具。为避免窒息的危险，请将这些塑料袋远离婴儿和儿童，并正确弃置这些塑料袋。\n\n•\t对于使用起搏器的用户：\n\n本设备可能会产生弱磁场。如果您在本设备附近感觉到起搏器工作不正常，请远离本设备，并立即咨询医生。\n\n•\t使用本设备之后短时间内，本设备的一些内部零件仍然处于极热状态。打开前盖时，请勿触摸以灰色标记的区域。存在烧伤的风险。先等待设备冷却下来，再触摸设备的内部零件。\n\n![⚠️ **高温警示：设备内部零件冷却前勿触碰，防止烧伤**  （图示说明：使用后设备内部部件仍达170°C/338°F，灰色标记区域为高温区；严禁在未冷却时打开前盖或触摸内部零件）](http://192.168.204.130:9002/think-tank/upload-images/hak180产品安全手册/f3349cded08d6686a93d0a81b9a64ec1e50d9a82cbb88541b37027f085813a15.jpg)  \n儎⑟ഴḽ䆜઀ᛞ࠽व䀜᪮儎⑟Ⲻ䇴༽䜞ԬȾ\n\n![**图：设备内部结构示意图——手部操作部件（电源线安全警示前页）**](http://192.168.204.130:9002/think-tank/upload-images/hak180产品安全手册/501bb8d2d681e4502d87badb15a68939eadfa086d309c3599f1c36b0bc559177.jpg)\n"
            }},
            {"entity": {
                "item_name": "HAK180烫金机",
                "chunks_id": 468191558938394827,
                "content": "\n产品安全手册（简体中文）\n\n感谢您购买 HAK 180 烫金机。\n\n在使用本设备之前，请先阅读本手册，包括所有预防措施。阅读本手册后，请妥善保管。\n\n有关使用本设备的更多信息，请参阅使用说明书，其可在兄弟 (中国)商业有限公司技术服务支持网站 http://www.95105369.com/Web/Manuals.aspx 上找到。建议您先通读使用说明书，再使用本设备。\n\n如需获得常见问题解答、故障排除和说明书，请访问\n\nhttp://www.95105369.com。\n\n对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。\n\n•\t对于保养、调整或维修事宜，请联系 Brother 呼叫中心或您当地的Brother 经销商。\n\n•\t如果本设备工作不正常或发生任何错误，请关闭本设备，拔下所有电缆，然后联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t本文档中提供的信息可能会随时更改，恕不另行通知。\n\n•\t严禁未经授权擅自复制或重制本文档的任何部分或全部内容。\n\n•\t请注意，对于使用通过本设备制作的产品造成的任何损坏或利润损失，或者故障、维修导致的数据消失或更改，或者第三方提出的任何索赔，我们不承担任何责任。\n"
            }},
            {"entity": {
                "item_name": "HAK180烫金机",
                "chunks_id": 468191558938394834,
                "content": "\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整操作说明中涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合格技术进行全面工作以将本设备恢复到正常操作。Brother不建议使用 Brother 正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备的任何零件，由此导致的任何维修可能不在保修范围内。\n"
            }},
        ]
    }
    node_rrf = NodeRrf()
    result = node_rrf(mock_state)
    logger.info(format_json(result))





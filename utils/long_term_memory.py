"""
长期记忆模块
跨会话存储用户偏好、重要事实、设备信息等。
基于 MongoDB，按 user_id 隔离（user_id 前端生成后永久存储，不随新对话清除），
支持按重要性筛选、自动提取。
"""
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING

load_dotenv()

logger = logging.getLogger(__name__)
# 偏好子类型：只对 memory_type=preference 有效，用于细分覆盖范围
PREF_SUBTYPE_STYLE = "style"      # 回答风格：简洁/详细、正式/口语、语言等
PREF_SUBTYPE_CONTENT = "content"  # 内容要求：配图、步骤、参数表、引用来源等
PREF_SUBTYPE_OTHER = "other"      # 其他偏好
PREF_SUBTYPES = {PREF_SUBTYPE_STYLE, PREF_SUBTYPE_CONTENT, PREF_SUBTYPE_OTHER}

# 矛盾检测时最多检测的旧偏好条数（防止旧偏好太多时 LLM 调用过多）
MAX_CONFLICT_CHECK_PREFS = 10

# LLM 记忆提取提示词：通用识别 + 短句压缩 + 类型/重要性标注，一步完成
LLTM_EXTRACT_TEMPLATE = """你是长期记忆提取器，根据对话提取值得跨会话记住的用户信息。

规则：
1. 只提取跨会话有价值的信息：用户偏好、重要事实、涉及的产品/设备型号、未完成事项
2. 每条记忆必须是简短短句（不超过30字），语义精炼，去掉口语和冗余
3. 类型取值：preference(偏好) / fact(事实) / device(设备) / history(重要历史)
4. 如果类型是 preference，必须同时给出 preference_subtype（偏好子类型）：
   - style：回答风格相关（简洁/详细、正式/口语、用中文/英文、语气等）
   - content：内容要求相关（要配图、要分步骤、要参数表、要引用来源、要标注型号等）
   - other：不属于以上两类的偏好
5. fact/device/history 类型不需要 preference_subtype 字段
6. 不要提取一次性问答的普通内容；设备型号无论是否常见、是否是新型号都要提取

用户提问：{user_query}
助手回答：{assistant_answer}

直接返回JSON数组，不要输出其他任何内容：
[{{"content": "简短短句", "type": "preference", "preference_subtype": "style", "importance": 7}}]
"""


class LongTermMemory:
    """长期记忆管理：MongoDB 存储，按 user_id 隔离（跨会话生效）"""

    def __init__(self):
        self.mongo_url = os.getenv("MONGO_URL")
        self.db_name = os.getenv("MONGO_DB_NAME")
        self.client = MongoClient(self.mongo_url)
        self.db = self.client[self.db_name]
        self.collection = self.db["long_term_memory"]
        # 索引：user_id + importance（按重要性查询）
        self.collection.create_index([("user_id", 1), ("importance", -1)])
        self.collection.create_index([("user_id", 1), ("created_at", -1)])
        # 去重索引：同 user + 同内容哈希唯一
        self.collection.create_index([("user_id", 1), ("content_hash", 1)], unique=True)
        logger.info("LongTermMemory 初始化完成")

    def save_memory(
        self,
        user_id: str,
        content: str,
        memory_type: str = "fact",
        importance: int = 5,
        metadata: Optional[Dict] = None,
        preference_subtype: str = PREF_SUBTYPE_OTHER,
    ) ->str:
        """
        保存一条长期记忆（内容哈希去重：同 user + 同内容 → 更新而非重复插入）
        :param user_id: 用户标识（前端生成，跨会话不变）
        :param content: 记忆内容
        :param memory_type: 类型（preference偏好 / fact事实 / device设备 / history历史）
        :param importance: 重要性 1-10，越高越优先加载
        :param metadata: 附加元数据
        :return: 记录ID
        """
        import hashlib
        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        now = datetime.now().timestamp()

        # 偏好类记忆精准覆盖：只降级「同子类型 + 真正矛盾」的旧偏好
        # 1. 先按子类型粗筛：不同子类型的偏好天然不冲突（如风格偏好 vs 内容要求）
        # 2. 同子类型内再用 LLM 精筛：只降级和新偏好矛盾的旧偏好
        # device/fact/history 类型不处理
        if memory_type == "preference":
            #规范化子类型
            sub = preference_subtype if preference_subtype in PREF_SUBTYPES else PREF_SUBTYPE_OTHER
            # 查询同子类型的旧偏好（按时间倒序，只取最近的 N 条做检测）
            old_prefs = list(
                self.collection.find({
                    "user_id" : user_id,
                    "memory_type" : "preference",
                    "preference_subtype" : sub,
                }).sort("created_at",-1).limit(MAX_CONFLICT_CHECK_PREFS)
            )
            downgraded = 0
            for old in old_prefs:
                old_content = old.get("content","")
                # 快速判断：内容完全相同视为重复，直接降级
                if old_content.strip()==content.strip():
                    self.collection.update_one({"_id":old["_id"]},{"$set":{"importance":1}})
                    downgraded+=1
                    continue
                # LLM 矛盾检测：只降级真正矛盾的旧偏好
                if self.is_preference_conflict(old_content,content):
                    self.collection.update_one({"_id": old["_id"]},{"$set":{"importance":1}})
                    downgraded+=1
                    logger.info(f"偏好矛盾降级: 旧='{old_content[:40]}' 新='{content[:40]}'")
                    logger.info(f"长期记忆偏好精准覆盖: 子类型={sub}, 检测{len(old_prefs)}条旧偏好, 降级{downgraded}条")

        # 去重：同 user + 同内容哈希已存在 → 刷新时间并提升重要性，避免重复堆叠
        existing = self.collection.find_one({
            "user_id": user_id,
            "content_hash": content_hash,
        })
        if existing:
            self.collection.update_one(
                {"_id": existing["_id"]},
                {"$set": {
                    "created_at": now,
                    "importance": max(existing.get("importance", 5), importance),
                }},
            )
            logger.info(f"长期记忆去重更新: type={memory_type}, content={content[:40]}")
            return str(existing["_id"])

        doc = {
            "user_id": user_id,
            "content": content,
            "content_hash": content_hash,
            "memory_type": memory_type,
            "importance": max(1, min(10, importance)),
            "metadata": metadata or {},
            "created_at": now,
        }
        # 只对 preference 类型存储子类型
        if memory_type=="preference":
            doc["preference_subtype"] = preference_subtype if preference_subtype in PREF_SUBTYPES else PREF_SUBTYPE_OTHER
        result = self.collection.insert_one(doc)
        logger.info(f"保存长期记忆: type={memory_type}, importance={importance}, content={content[:50]}")
        return str(result.inserted_id)

    def is_preference_conflict(self,old_content:str,new_content:str)->bool:
        """
        用 LLM 判断两条偏好是否矛盾（无法同时满足）。
        失败时保守返回 False（不降级），避免误伤。
        """
        try:
            import  json
            from utils.llm_utils import get_llm_client
            from config.lm_config import lm_config
            from langchain_core.messages import HumanMessage
            llm = get_llm_client(model=lm_config.item_model,json_mode=True)
            prompt = f"""判断以下两条用户偏好是否互相矛盾（无法同时满足）。
            只返回 JSON，不要输出其他内容：{{"conflict": true}} 或 {{"conflict": false}}

            旧偏好：{old_content}
            新偏好：{new_content}
            """
            resp = llm.invoke([HumanMessage(content=prompt)])
            data = json.loads(resp.content)
            return bool(data.get("conflict",False))
        except Exception as  e:
            logger.warning(f"偏好矛盾检测失败，保守返回不矛盾: {e}")
            return False

    def get_memories(
        self,
        user_id: str,
        min_importance: int = 3,
        max_tokens: int = 800,
        max_days: int = 180,
    ) -> List[Dict[str, Any]]:
        """
        加载长期记忆：
        - 按重要性降序、时间降序排列
        - 时间衰减：超过 max_days 的记忆不再注入
        - token 预算：累计内容长度超过预算即停止，防止 prompt 膨胀
        :param user_id: 用户标识（前端生成，跨会话不变）
        :param min_importance: 最低重要性阈值
        :param max_tokens: 记忆区最大 token 预算（保守估 1 token ≈ 2 字符）
        :param max_days: 记忆有效天数，超期自动失效
        :return: 记忆列表
        """
        now = datetime.now().timestamp()
        cutoff = now - max_days * 86400
        query = {
            "user_id": user_id,
            "importance": {"$gte": min_importance},
            "created_at": {"$gte": cutoff},
        }
        cursor = (
            self.collection.find(query)
            .sort([("importance", -1), ("created_at", -1)])
        )
        memories = []
        total_chars = 0
        budget_chars = max_tokens * 2
        for m in cursor:
            content_len = len(m.get("content", ""))
            if total_chars + content_len > budget_chars:
                break
            memories.append(m)
            total_chars += content_len
        logger.info(f"加载长期记忆: user={user_id}, count={len(memories)}"
                    f"(预算{max_tokens}tokens, 有效期{max_days}天)")
        return memories

    def format_memories_for_prompt(self, user_id: str, min_importance: int = 3) -> str:
        """
        将长期记忆格式化为提示词文本
        """
        memories = self.get_memories(user_id, min_importance=min_importance)
        if not memories:
            return ""
        lines = []
        type_labels = {
            "preference": "用户偏好",
            "fact": "重要事实",
            "device": "设备信息",
            "history": "历史记录",
        }
        subtype_labels = {
            PREF_SUBTYPE_STYLE: "风格",
            PREF_SUBTYPE_CONTENT: "内容",
            PREF_SUBTYPE_OTHER: "偏好",
        }
        for m in memories:
            mtype = m.get("memory_type", "fact")
            if mtype=="preference":
                sub = m.get("preference_subtype",PREF_SUBTYPE_OTHER)
                label = f"用户偏好-{subtype_labels.get(sub, '偏好')}"
            else:
                label = type_labels.get(mtype, mtype)
            lines.append(f"- [{label}] {m.get('content', '')}")
        return "\n".join(lines)

    def extract_and_save(
        self,
        user_id: str,
        user_query: str,
        assistant_answer: str,
    ) -> int:
        """
        从一轮对话中提取值得长期记忆的信息并保存。
        用 LLM 提取：通用识别任何型号/偏好 + 短句压缩 + 类型/重要性标注，一步完成。
        不依赖写死的正则/关键词，后期新增文件无需改代码。
        提取失败不阻塞主流程，直接跳过。
        :return: 新保存/更新的记忆条数
        """
        try:
            memories = self._llm_extract(user_query, assistant_answer)
        except Exception as e:
            logger.warning(f"LLM 提取长期记忆失败，跳过: {e}")
            return 0

        saved = 0
        for m in memories or []:
            if isinstance(m, dict):
                content = str(m.get("content", "")).strip()
            else:
                continue
            if not content:
                continue
            mtype = str(m.get("type", "fact"))
            try:
                importance = int(m.get("importance", 5))
            except (TypeError, ValueError):
                importance = 5
            pre_subtype = str(m.get("preference_subtype", PREF_SUBTYPE_OTHER))
            if pre_subtype not in PREF_SUBTYPES:
                pre_subtype = PREF_SUBTYPE_OTHER
            self.save_memory(user_id, content, mtype, importance,preference_subtype=pre_subtype)
            saved += 1
        if saved > 0:
            logger.info(f"从对话提取并保存了 {saved} 条长期记忆")
        return saved

    def _llm_extract(
        self,
        user_query: str,
        assistant_answer: str,
    ) -> List[Dict[str, Any]]:
        """
        调用 LLM（qwen-flash，便宜快）提取记忆，返回 [{content,type,importance}]。
        json_mode=True 强制 JSON 输出，用 json.loads 解析。
        """
        import json
        from utils.llm_utils import get_llm_client
        from config.lm_config import lm_config
        from langchain_core.messages import HumanMessage

        llm = get_llm_client(model=lm_config.item_model, json_mode=True)
        prompt = LLTM_EXTRACT_TEMPLATE.format(
            user_query=(user_query or "")[:500],
            assistant_answer=(assistant_answer or "")[:1500],
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content
        if not content:
            return []
        data = json.loads(content)
        if isinstance(data, dict) and "memories" in data:
            return data["memories"]
        if isinstance(data, list):
            return data
        return []

    def prune_user(self, user_id: str, max_items: int = 50) -> int:
        """
        单用户记忆总量控制：超过 max_items 时，删除最旧、重要性最低的超额部分。
        防止记忆无限膨胀（MongoDB 只增不减）。
        :return: 删除的记忆条数
        """
        count = self.collection.count_documents({"user_id": user_id})
        if count <= max_items:
            return 0
        overflow = count - max_items
        cursor = (
            self.collection.find({"user_id": user_id})
            .sort([("importance", 1), ("created_at", 1)])
            .limit(overflow)
        )
        ids = [m["_id"] for m in cursor]
        if ids:
            self.collection.delete_many({"_id": {"$in": ids}})
        logger.info(f"长期记忆淘汰: user={user_id}, 删除 {len(ids)} 条超额记忆")
        return len(ids)

    def clear_user(self, user_id: str) -> int:
        """清空指定用户的所有长期记忆"""
        result = self.collection.delete_many({"user_id": user_id})
        logger.info(f"清空长期记忆: user={user_id}, count={result.deleted_count}")
        return result.deleted_count


# 单例
_memory_instance: Optional[LongTermMemory] = None


def get_long_term_memory() -> LongTermMemory:
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = LongTermMemory()
    return _memory_instance


if __name__ == "__main__":
    ltm = get_long_term_memory()
    uid = "test_user_001"
    ltm.clear_user(uid)

    # 测试保存
    ltm.save_memory(uid, "用户喜欢简洁的回答", "preference", 8)
    ltm.save_memory(uid, "用户的设备是 HAK180 烫金机", "device", 6)

    # 测试加载
    formatted = ltm.format_memories_for_prompt(uid)
    print("格式化后的长期记忆:")
    print(formatted)

    # 测试提取
    ltm.extract_and_save(uid, "我喜欢简洁回答", "好的，我会简洁回答")
    print("\n提取后记忆:")
    print(ltm.format_memories_for_prompt(uid))

    ltm.clear_user(uid)
    print("\n测试完成，已清理")

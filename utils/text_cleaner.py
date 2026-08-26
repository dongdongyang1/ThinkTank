"""
文本清理工具：将 Markdown 格式转为纯文本
作为 LLM 不遵守 prompt 时的兜底方案
"""
import re


def clean_markdown(text: str) -> str:
    """
    将 Markdown 文本清理为纯文本
    - 去掉 ### 标题标记
    - 去掉 **加粗**、*斜体*
    - 去掉 --- 分隔线
    - [文字](链接) → 文字（链接）
    - - 列表 → 保留文字（去掉前缀符号）
    - > 引用 → 去掉 >
    - 代码块 ``` → 去掉
    """
    if not text:
        return text

    # 1. 去掉代码块标记 ```
    text = re.sub(r'```[\w]*\n?', '', text)
    text = re.sub(r'\n?```', '', text)

    # 2. 去掉行内代码 `code` → code
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # 3. [文字](链接) → 文字（链接）
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1（\2）', text)

    # 4. 去掉图片标记 ![alt](url) → 空（图片由系统单独渲染）
    text = re.sub(r'!\[.*?\]\([^)]+\)', '', text)

    # 5. 去掉标题标记 ### / ## / #
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

    # 6. 去掉加粗 **text** → text
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)

    # 7. 去掉斜体 *text* → text（注意不要匹配到列表符号）
    text = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'\1', text)

    # 8. 去掉分隔线 ---（单独一行）
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

    # 9. 去掉引用 > 
    text = re.sub(r'^\s*>\s?', '', text, flags=re.MULTILINE)

    # 10. 列表符号 - / * / + → 保留文字（去掉前缀）
    # 注意：这一步在加粗/斜体之后，避免把 *text* 误判为列表
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)

    # 11. 清理多余空行（连续3个以上换行 → 2个）
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 12. 去掉行首尾空格
    lines = text.split('\n')
    lines = [line.rstrip() for line in lines]
    text = '\n'.join(lines)

    return text.strip()


if __name__ == "__main__":
    test = """### ✅ 方法一：使用管理员密码

如果曾设置过 **BIOS 管理员密码**，可通过该密码进入 BIOS：

1. 开机时按 `F2` 进入 BIOS
2. 使用管理员密码登录
3. 进入 **Security** 选项卡

> 注意：仅当您记得管理员密码时才可使用此方法。

---

### 方法二：联系技术支持

- 官方客服：400-830-8300
- 在线支持：[华为官网](https://support.huawei.com)

```python
print("hello")
```
"""
    print("原始:")
    print(test)
    print("\n" + "=" * 50 + "\n")
    print("清理后:")
    print(clean_markdown(test))

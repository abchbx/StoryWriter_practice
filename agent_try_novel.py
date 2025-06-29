# 注意：根据您的要求，所有函数名和变量名均保持原样。
# 仅修改了内部的 prompt (提示)，将AI的功能从“教材编写”切换为“短篇小说创作”。
# 因此，您可能会在日志输出中看到“教材”、“小节”等词语，但其实际执行的是小说创作任务。

import copy
import json
import os
import re
from typing import Dict, List, Tuple

import autogen
from autogen import AssistantAgent, ConversableAgent
from autogen.agentchat.contrib.capabilities import transform_messages
from openai import OpenAI, APIError # 导入 APIError 以便更精确地捕获错误

# ==============================================================================
# 1. 配置信息 (Configuration)
# ==============================================================================
# 【安全建议】: 请不要将 API Key 直接写入代码中。
# 推荐使用环境变量或配置文件来管理您的密钥。

# --- 主要 LLM 配置 ---
PRIMARY_API_KEY = "ca3bcb7f94e64b36408aa77e2db8ff93.LelLOG0PmmJxZB9Q" # <--- 请在这里填入您的主 API Key
PRIMARY_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
PRIMARY_MODEL = "glm-4-flash-250414"

# --- 备用 LLM 配置 (请替换成您的备用模型信息) ---
BACKUP_API_KEY = "hunyuan-lite"  # <--- 【请修改】您的备用 API Key
BACKUP_BASE_URL = "https://api.hunyuan.cloud.tencent.com/v1" # <--- 【请修改】您的备用 Base URL
BACKUP_MODEL = "hunyuan-lite" # <--- 【请修改】您的备用模型名称


# 将主备 LLM 添加到配置列表
# AutoGen 会按照列表顺序尝试连接，如果第一个失败，会自动尝试下一个。
config_list = [
    {
        "model": PRIMARY_MODEL,
        "base_url": PRIMARY_BASE_URL,
        "api_key": PRIMARY_API_KEY,
        "api_type": "openai",
        "price":[1,1]
    },
    {
        "model": BACKUP_MODEL,
        "base_url": BACKUP_BASE_URL,
        "api_key": BACKUP_API_KEY,
        "api_type": "openai",
        "price":[1,1]
    },
]

# 用于文本压缩的配置（同样支持主备）
# 注意：文本压缩建议使用成本较低的轻量级模型
summarizer_config_list = [
    {
        "model": PRIMARY_MODEL,
        "base_url": PRIMARY_BASE_URL,
        "api_key": PRIMARY_API_KEY,
        "api_type": "openai",
        "price":[1,1]
    },
    {
        "model": BACKUP_MODEL,
        "base_url": BACKUP_BASE_URL,
        "api_key": BACKUP_API_KEY,
        "api_type": "openai",
        "price":[1,1]
    },
]


# 用于缓存压缩后的消息，避免重复计算
condensed_cache = {}


# ==============================================================================
# 2. 上下文压缩与消息转换 (Context Compression & Message Transformation)
# ==============================================================================
class MessageRedact:
    """
    一个消息转换类，用于压缩长对话历史，节省 token。
    【已更新】:此类现在支持备用 LLM。
    """
    def __init__(self, config_list_for_compression: List[Dict]):
        self.condensed_cache = {}
        # 传入用于压缩的LLM配置列表
        self.config_list = config_list_for_compression
        # 初始化一个客户端，后续会根据需要动态更新其配置
        self.client = OpenAI(
            api_key=self.config_list[0]["api_key"],
            base_url=self.config_list[0]["base_url"],
        )
        self.prompt_template = """
你是一位专业的文本摘要师。你的任务是将以下对话内容压缩成一段高度精炼、保留核心信息的摘要。

**压缩要求:**
1.  **目标长度**: 将内容压缩至其原始长度的20%左右。
2.  **保留核心**: 必须保留最关键的信息、关键决策、主要冲突和最终的解决方案或结论。
3.  **省略细节**: 移除不影响核心逻辑的场景描述、次要对话和修饰性语言。
4.  **确保连贯**: 压缩后的文本必须逻辑清晰、语句通顺，能独立构成一段有意义的摘要。

请对以下内容进行处理：
"""

    def _call_llm_with_fallback(self, messages: List[Dict]) -> str:
        """
        尝试使用配置列表中的每个LLM来获取摘要，直到成功为止。
        """
        last_exception = None
        for config in self.config_list:
            try:
                print(f"INFO: Attempting to compress message using model: {config.get('model')} at {config.get('base_url')}")
                # 动态更新客户端配置
                client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
                completion = client.chat.completions.create(
                    model=config["model"],
                    messages=messages,
                ).choices[0].message.content
                return completion
            except Exception as e:
                print(f"WARNING: Compression with model '{config.get('model')}' failed. Reason: {e}")
                last_exception = e
                continue # 尝试下一个配置
        
        # 如果所有配置都失败了，则抛出最后一个捕获到的异常
        raise last_exception if last_exception else APIError("All LLM configurations for summarization failed.", code=500)


    def apply_transform(self, messages: List[Dict]) -> List[Dict]:
        temp_messages = copy.deepcopy(messages)

        if len(temp_messages) > 3:
            for message in temp_messages[1:-2]:
                content = message.get("content")
                if isinstance(content, str) and len(content) > 150:
                    message_id = hash(content)
                    if message_id in self.condensed_cache:
                        print(f"INFO: Using cached condensed message for role: {message['role']}")
                        message["content"] = self.condensed_cache[message_id]
                    else:
                        print(f"INFO: Compressing message for role: {message['role']}...")
                        try:
                            full_prompt = self.prompt_template + content
                            # 使用新的带回退逻辑的方法
                            completion = self._call_llm_with_fallback(
                                messages=[{"role": "user", "content": full_prompt}]
                            )
                            
                            message["content"] = completion
                            self.condensed_cache[message_id] = completion
                            print("INFO: Compression successful.")
                        except Exception as e:
                            print(f"ERROR: Could not compress message after trying all fallbacks. Final error: {e}")
                            pass
                            
        return temp_messages

    def get_logs(self, pre_transform_messages: List[Dict], post_transform_messages: List[Dict]) -> Tuple[str, bool]:
        for i in range(len(pre_transform_messages)):
            if pre_transform_messages[i].get("content") != post_transform_messages[i].get("content"):
                return "Message content was condensed.", True
        return "No message content was changed.", False

# 实例化消息转换处理器，并传入用于压缩的配置列表
redact_handling = transform_messages.TransformMessages(transforms=[MessageRedact(summarizer_config_list)])


# ==============================================================================
# 3. 核心功能函数 (Core Generation Functions)
# ==============================================================================

def event_generate(premise: str, id: str, output_prefix: str) -> str:
    """
    **小说模式**: 根据给定的主题（premise）生成结构化的故事大纲。
    此函数现在负责创建故事的整体情节结构。
    """
    outline_creator = ConversableAgent(
        name="OutlineCreator",
        system_message=f"""
你是一位想象力丰富、擅长结构化叙事的情节构思师。
你的核心任务是根据下面提供的【故事主题 (Premise)】，设计一份引人入胜、逻辑严谨的短篇小说大纲。

**核心主题 (Premise):**
`{premise}`

**你的输出必须严格遵循以下 Markdown 格式，以便后续处理:**
# [小说标题]
## [一句话故事梗概]
### 1. 开端：[介绍主角、背景，埋下核心冲突的种子]
### 2. 上升：[冲突展开，主角遭遇第一个主要障碍]
### 3. 高潮：[故事最紧张的时刻，主角与核心冲突正面对决]
### 4. 下降：[高潮过后的余波，展示对决的直接后果]
### 5. 结局：[冲突解决，主角达成（或失去）目标，故事收尾]

**设计要求:**
1.  **戏剧性**: 确保情节有明确的冲突、转折和高潮，避免平铺直叙。
2.  **逻辑性**: 人物动机要合理，情节发展要符合逻辑，经得起推敲。
3.  **创意性**: 在主题框架内，构思新颖的人物、情节或世界观。
4.  **结构化**: 严格遵守上述的 Markdown 标题层级和编号格式。
""",
        llm_config={"config_list": config_list, "temperature": 0.7},
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
    )

    outline_auditor = ConversableAgent(
        name="OutlineAuditor",
        system_message=f"""
你是一位经验丰富的故事编辑和审核员，负责评估故事大纲的潜力。
**核心主题 (Premise):**
`{premise}`

**评估标准:**
1.  **吸引力 (最高优先级)**: 大纲是否有趣？是否有足够的悬念和冲突来吸引读者？
2.  **逻辑结构**: 情节转折是否生硬？人物动机是否可信？
3.  **完整性**: 是否包含了故事必备的开端、发展、高潮和结局？
4.  **格式规范**: 是否严格遵循了要求的 Markdown 格式？

**指令:**
- 如果大纲在所有方面都表现优秀，请直接回复 `TERMINATE`。
- 如果大纲存在问题（如：情节平淡、逻辑不通、缺乏创意），请回复 `重新生成`，并明确指出需要修改的核心问题。
""",
        llm_config={"config_list": config_list, "temperature": 0.3},
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
    )

    try:
        print(f"INFO: [小说创作模式] 正在根据主题 '{premise}' 生成故事大纲...")
        outline_auditor.initiate_chat(outline_creator, message="请根据我提供给你的核心主题开始设计一份短篇小说大纲。", max_turns=5)
        print("INFO: 故事大纲生成流程结束。")
    except Exception as e:
        print(f"ERROR: 大纲生成失败，已尝试所有备用模型。错误: {e}")
        return "None"

    # 提取并保存结果
    messages = [entry['content'] for entries in outline_auditor.chat_messages.values() for entry in entries]
    output_dir = os.path.join(output_prefix, "textbook_outline")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"outline_{id}.json")
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(messages, file, ensure_ascii=False, indent=4)
        
    final_outline = ""
    chat_history = outline_auditor.chat_messages.get(outline_creator)
    if chat_history:
        # 寻找最后一个非TERMINATE的有效回复
        for msg in reversed(chat_history):
            content = msg.get("content", "")
            if content and "TERMINATE" not in content:
                final_outline = content
                break
    
    if not final_outline:
        print("WARNING: 未能从对话历史中提取出有效的大纲。")
        
    return final_outline


def event_extract(text_data: str, id: str, output_prefix: str) -> str:
    """
    **小说模式**: 从生成的大纲文本中，精确抽取出所有情节要点。
    此函数负责解析大纲，为下一步的故事创作做准备。
    """
    if not isinstance(text_data, str) or not text_data:
        print("ERROR: event_extract 需要一个非空的字符串输入。")
        return ""
        
    print("INFO: [小说创作模式] 正在从大纲中提取所有情节要点...")
    # 正则表达式，用于匹配 '### 1.' 或 '### 2.' 等格式的情节要点标题
    section_pattern = r"^\s*#{3,}\s*[\d\.]+\s*(.*)"
    sections = re.findall(section_pattern, text_data, re.MULTILINE)

    if not sections:
        print("WARNING: 在大纲中未能找到符合格式 '### 数字. ...' 的情节要点标题。")
        # 尝试匹配更宽松的格式，如以 '### ' 开头的行
        section_pattern_loose = r"^\s*#{3,}\s*(.*)"
        sections = re.findall(section_pattern_loose, text_data, re.MULTILINE)
        if sections:
            print(f"INFO: 已使用宽松模式找到 {len(sections)} 个情节要点。")
        else:
            print("WARNING: 宽松模式下也未能找到任何情节要点。")
            return ""
    
    print(f"INFO: 成功提取 {len(sections)} 个情节要点标题。")
    
    # 将小节标题列表保存到JSON文件中，以供调试或记录
    output_dir = os.path.join(output_prefix, "final_process")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"sections_{id}.json")

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump({"sections": sections}, json_file, ensure_ascii=False, indent=4)

    # 返回以换行符分隔的小节标题字符串，供下一个函数使用
    return "\n".join(sections)


def story_generate(premise: str, event_str: str, id: str, output_prefix: str) -> str:
    """
    **小说模式**: 遍历每个情节要点，生成详细的故事情节，并最终合并成一个完整的Markdown文件。
    此函数是故事创作的核心。
    """
    if not event_str.strip():
        print("ERROR: 未提供任何情节要点 (event_str为空)，无法生成故事内容。")
        return "stop"

    print("INFO: [小说创作模式] 故事内容生成启动。")

    # --- 代理定义 ---
    llm_config_writer = {"config_list": config_list, "temperature": 0.75}
    llm_config_critic = {"config_list": config_list, "temperature": 0.4}

    section_writer = ConversableAgent(
        name="SectionWriter",
        system_message=f"""
你是一位技艺精湛的小说家，文笔优美，擅长刻画人物和营造氛围。
你的【首要职责】是围绕【故事主题 (Premise)】进行故事创作。

**写作手法指引 (请务必借鉴):**
1.  **展示，而非说教 (Show, Don't Tell)**: 不要直接说“他很伤心”，而是通过描写他“紧握的双拳和颤抖的嘴唇”来展示他的情绪。
2.  **感官细节**: 充分运用视觉、听觉、嗅觉、触觉和味觉的描写，让读者身临其境。
3.  **内心独白与情感**: 深入角色的内心世界，描写他们的思想、情感和动机。
4.  **黄金句式**: 运用长短句结合、比喻、拟人等修辞手法，增强语言的节奏感和表现力。
5.  **对话**: 对话要符合人物性格，推动情节发展，而不是简单的信息交换。

**本次任务:**
- 为我指定的【当前情节要点】创作一段生动、具体的场景。
- 你的内容【必须】严格服务于核心主题：`{premise}`。
- 只输出当前场景的故事内容，不要包含情节标题本身或任何题外话。
""",
        llm_config=llm_config_writer,
        human_input_mode="NEVER",
    )
    # 为作者代理添加消息压缩能力，以防对话过长
    redact_handling.add_to_agent(section_writer)

    section_critic = ConversableAgent(
        name="SectionCritic",
        system_message=f"""
你是一位眼光毒辣的文学评论家。你的唯一职责是审查故事场景的文学质量和叙事效果。
**【核心主题 (Premise)】**: `{premise}`

**审查指令**:
- 如果场景【文笔流畅、情感真挚、成功推进了情节、并运用了出色的写作技巧】，请只回复 `APPROVE`。
- 如果场景【存在陈词滥调、人物塑造扁平、描写空洞、对话生硬、或偏离主题】，请立即回复 `REWRITE:` 并附上具体的、可操作的修改意见（例如：“这里的环境描写可以更具体，加入一些声音和气味的细节”或“这段对话无法体现主角的焦急心态，建议修改”）。
""",
        llm_config=llm_config_critic,
        human_input_mode="NEVER",
    )

    user_proxy = ConversableAgent("user_proxy", human_input_mode="NEVER", llm_config=False, code_execution_config=False)

    # --- 循环为每个小节生成内容 ---
    print("\n--- [小说创作模式] 开始逐一生成和审核每个故事场景 ---")
    final_textbook_content = []
    # 从 event_str (由event_extract函数生成) 中获取小节列表
    section_list = [line.strip() for line in event_str.split('\n') if line.strip()]

    for i, section_title in enumerate(section_list):
        print(f"\n正在处理情节 {i+1}/{len(section_list)}: {section_title}...")
        
        # 为了美观，我们用分割线来区分不同场景
        if i > 0:
            final_textbook_content.append("\n* * *\n\n")

        match = re.match(r"([\d\.]+)\s*(.*)", section_title)
        if match:
            section_number = match.group(1)
            clean_title = match.group(2)
        else:
            clean_title = section_title
            
        writing_prompt = f"""
请严格遵循你的小说家角色设定，为以下情节要点创作故事场景。

**核心主题 (Premise):** `{premise}`
**当前情节要点:** `{clean_title}`
"""
        
        # 为一个小节内容进行最多3轮的“写作-审核”循环
        for attempt in range(3):
            try:
                # 写作
                user_proxy.initiate_chat(section_writer, message=writing_prompt, max_turns=1, clear_history=True)
                generated_content = section_writer.last_message().get("content", "")
                
                # 审核
                critic_prompt = f"""
请审查以下故事场景的文学质量。
**核心主题:** `{premise}`
**情节要点:** `{clean_title}`
**待审内容**:
`{generated_content}`
"""
                user_proxy.initiate_chat(section_critic, message=critic_prompt, max_turns=1, clear_history=True)
                critic_feedback = section_critic.last_message().get("content", "")
                
                print(f"第 {attempt+1} 次尝试审核反馈: {critic_feedback[:100]}...")
                
                if "APPROVE" in critic_feedback.upper():
                    print("结果: 审核通过。")
                    final_textbook_content.append(generated_content.strip() + "\n")
                    break # 成功，跳出重试循环
                else:
                    print("结果: 审核不通过，准备重写。")
                    # 将审核意见加入下一次的写作提示中
                    writing_prompt = f"""
你的上一稿未通过审核，请根据以下文学评论家的意见重新创作。
**评论家意见:** {critic_feedback}
---
请严格遵循你的小说家角色设定，为以下情节要点创作故事场景。

**核心主题 (Premise):** `{premise}`
**当前情节要点:** `{clean_title}`
"""
                    if attempt == 2: # 最后一次尝试失败
                        print(f"WARNING: 情节 '{section_title}' 经过多次尝试后仍未通过审核，将采用最后一版内容。")
                        final_textbook_content.append(generated_content.strip() + "\n")

            except Exception as e:
                print(f"ERROR: 处理情节 '{section_title}' 时发生错误: {e}")
                # 发生异常时，跳过此情节以保证流程继续
                break
    
    # --- 保存最终的教材文件 ---
    print("\n--- [小说创作模式] 所有情节处理完毕，正在保存为 Markdown 文件 ---")
    
    output_dir = os.path.join(output_prefix, "final_textbook")
    os.makedirs(output_dir, exist_ok=True)
    # 将文件名后缀改为 .md
    output_path = os.path.join(output_dir, f"textbook_{id}.md")

    full_textbook = "".join(final_textbook_content)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(full_textbook)
        print(f"INFO: 完整小说已成功保存至: {output_path}")

    return "finish"
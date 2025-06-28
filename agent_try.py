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
PRIMARY_API_KEY = "ca" # <--- 请在这里填入您的主 API Key
PRIMARY_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
PRIMARY_MODEL = "glm-4-flash-250414"

# --- 备用 LLM 配置 (请替换成您的备用模型信息) ---
BACKUP_API_KEY = "sk-"  # <--- 【请修改】您的备用 API Key
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
1.  **目标长度**: 将内容压缩至其原始长度的15%左右。
2.  **保留核心**: 必须保留最关键的信息、关键决策、主要冲突和最终的解决方案或结论。
3.  **省略细节**: 移除不影响核心逻辑的场景描述、次要对话和修饰性语言。
4.  **保持中立**: 以客观、中立的第三人称视角进行总结。
5.  **确保连贯**: 压缩后的文本必须逻辑清晰、语句通顺，能独立构成一段有意义的摘要。

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
    根据故事大纲生成一系列结构化的事件。
    【已更新】: AutoGen会自动处理LLM的回退，无需手动重试循环。
    """
    event_creator = ConversableAgent(
        name="EventCreator",
        system_message=f"""
你是一位富有创意的故事策划师。你的核心任务是根据下面提供的【故事大纲 (Premise)】，设计出一系列结构完整、富有戏剧性的事件。
**你的所有工作都必须严格服务于这个核心大纲：**
`{premise}`
请严格按照以下格式为每个事件生成梗概，并确保内容翔实且引人入胜：
**事件**: [为事件起一个简洁的标题]
**场景**: [描述事件发生的时间、地点和环境氛围]
**角色**: [列出参与此事件的核心角色]
**行动**: [描述角色在此场景中的具体行为和对话]
**冲突**: [明确指出此事件中的核心矛盾点]
**情节转折**: [设计一个出人意料的转折或发现]
""",
        llm_config={"config_list": config_list, "temperature": 1}, # 传入包含主备的config_list
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
    )

    event_auditor = ConversableAgent(
        name="EventAuditor",
        system_message="""
你是一位经验丰富的剧本医生。你的任务是评估上一个代理生成的事件梗概是否合格。
**评估标准 (按重要性排序):**
1.  **核心一致性**: 这是【最高优先级】。事件是否绝对忠实于故事大纲 (Premise) 的核心设定？
2.  **戏剧性**: 冲突是否足够强烈？情节是否引人入胜？
3.  **逻辑性**: 事件的起因、经过、结果是否符合逻辑？情节转折是否突兀？
4.  **完整性**: 是否严格遵循了要求的格式（事件、场景、角色、行动、冲突、情节转折）？
**指令:**
- 如果事件在以上四个方面都表现出色，请直接回复 `TERMINATE`。
- 如果事件【不符合核心一致性】，或存在其他明显缺陷，请回复 `重新生成`，并简要说明需要改进的核心问题。
""",
        llm_config={"config_list": config_list, "temperature": 0.5}, # 传入包含主备的config_list
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
    )

    try:
        print(f"INFO: Attempting to generate events...")
        # AutoGen会自动处理回退，因此我们只需要调用一次
        event_auditor.initiate_chat(event_creator, message="请根据我提供给你的故事大纲开始创作事件。", max_turns=10)
        print("INFO: Event generation process finished.")
    except Exception as e:
        # 这个异常现在只会在所有LLM都失败时触发
        print(f"ERROR: Event generation failed after trying all LLMs in config_list. Error: {e}")
        return "None"

    # 提取并保存结果
    messages = [entry['content'] for entries in event_auditor.chat_messages.values() for entry in entries]
    output_dir = os.path.join(output_prefix, "final_events")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"final_events_{id}.json")
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(messages, file, ensure_ascii=False, indent=4)
        
    final_event_list = ""
    chat_history = event_auditor.chat_messages.get(event_creator)
    if chat_history and len(chat_history) >= 2:
        # 通常最后一个有意义的消息是倒数第二个（最后一个是TERMINATE）
        content = chat_history[-2].get("content")
        if isinstance(content, str):
            final_event_list = content
        else:
            print(f"WARNING: Expected a string from message content, but got {type(content)}. Using empty string.")
    else:
        # 如果找不到，尝试从所有消息中寻找非TERMINATE的最后一个消息
        all_msgs = [msg.get("content", "") for msg in chat_history if "TERMINATE" not in msg.get("content", "")]
        if all_msgs:
            final_event_list = all_msgs[-1]
        else:
            print("WARNING: Could not find sufficient chat history to extract final event list.")
            
    return final_event_list


def event_extract(text_data: str, id: str, output_prefix: str) -> str:
    """
    从生成的文本中抽取出事件描述。
    """
    if not isinstance(text_data, str):
        print(f"ERROR: event_extract expects a string, but got {type(text_data)}.")
        return ""
        
    print("INFO: Extracting structured events from text...")
    event_pattern = r"(?i)(?:\*\*事件\*\*|Event)[\s:]*.*?(?=\n\n(?i)(?:\*\*事件\*\*|Event)|$)"
    events = re.findall(event_pattern, text_data, re.DOTALL)

    if not events:
        print("WARNING: No events found in the text.")
        return ""
    
    print(f"INFO: Extracted {len(events)} events.")
    event_data = {"events": [{"event": event.strip()} for event in events]}
    
    output_dir = os.path.join(output_prefix, "final_process")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"events_{id}.json")

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(event_data, json_file, ensure_ascii=False, indent=4)

    event_str = "\n\n".join(event["event"] for event in event_data["events"])
    return event_str


def story_generate(premise: str, event_str: str, id: str, output_prefix: str) -> str:
    """
    基于事件列表，通过多代理协作生成完整的故事。
    【已更新】: 所有Agent都使用支持回退的config_list。
    """
    print("INFO: Story generation process started using a sequential workflow.")

    # --- 代理定义 ---
    # 所有代理都使用包含主备的 config_list
    llm_config_main = {"config_list": config_list, "temperature": 0.5}
    llm_config_writer = {"config_list": config_list, "temperature": 0.4}
    llm_config_critic = {"config_list": config_list, "temperature": 0.3}


    sub_events_divider = ConversableAgent(
        name="SubEventsDivider",
        system_message="""你是一个专业的叙事结构设计师。你的任务是将给定的“事件”分解成多个逻辑递进、情节更细化的“子事件”。确保划分后的子事件共同构成一个完整的小型叙事弧（开端-发展-高潮），并严格遵循输出格式。""",
        llm_config=llm_config_main,
        human_input_mode="NEVER",
    )

    story_writer = ConversableAgent(
        name="StoryWriter",
        system_message="""你是一位才华横溢的小说家。你的【首要且不可违背的职责】是围绕一个【核心设定 (Premise)】来创作故事。
**写作要求:**
1. **服从核心**: 你的所有创作都【必须】严格服务于我将提供给你的【核心设定】。
2. **聚焦当前**: 在遵循核心设定的前提下，将创作内容聚焦于我指定的【当前子事件】。
3. **生动描写**: 运用感官细节、角色心理活动和精彩的对话来丰满故事情节。
4. **输出格式**: 直接输出故事文本，不要添加任何标题或标签。""",
        llm_config=llm_config_writer,
        human_input_mode="NEVER",
    )
    redact_handling.add_to_agent(story_writer)

    story_critic = ConversableAgent(
        name="StoryCritic",
        system_message=f"""你是一名极其严格的剧本监督。你的唯一职责是【逐字逐句地】审查故事，确保其严格按照预设的【核心设定 (Premise)】发展。
**【核心设定 (Premise)】 - 这是唯一的评判标准**:
`{premise}`
**指令**:
- 如果故事内容【完全符合】核心设定，请只回复 `APPROVE`。
- 如果内容【哪怕有丝毫偏离】，请立即回复 `REWRITE:` 并附上具体的、必须修改的意见。""",
        llm_config=llm_config_critic,
        human_input_mode="NEVER",
    )

    user_proxy = ConversableAgent("user_proxy", human_input_mode="NEVER", llm_config=False)

    # --- 步骤 1: 划分出子事件 ---
    print("\n--- Step 1: Dividing events into sub-events ---")
    sub_events_prompt = f"""
{sub_events_divider.system_message}
**严格格式**:
事件1: [原事件1]
事件1.1: [子事件1.1]
事件1.2: [子事件1.2]
事件1.3: [子事件1.3]
**需要处理的事件如下:**
{event_str}
"""
    
    try:
        user_proxy.initiate_chat(sub_events_divider, message=sub_events_prompt, max_turns=1)
        last_msg = sub_events_divider.last_message()
        sub_events_text = last_msg.get("content") if last_msg else None

        if not isinstance(sub_events_text, str):
            print(f"ERROR: Failed to get valid sub-events text. Got: {sub_events_text}")
            return "stop"
        
        print("INFO: Sub-events created successfully.")
    except Exception as e:
        print(f"ERROR: Failed to create sub-events after trying all LLMs: {e}")
        return "stop"
    
    # --- 步骤 2: 循环生成和批判每个子事件的故事 ---
    print("\n--- Step 2: Writing and critiquing story for each sub-event ---")
    final_story_parts = []

    sub_event_list = [line for line in sub_events_text.split('\n') if re.match(r'事件\d\.\d:', line)]
    if not sub_event_list:
        print("ERROR: Could not parse sub-events from the output. Using raw output as fallback.")
        sub_event_list = [sub_events_text]

    for i, sub_event in enumerate(sub_event_list):
        print(f"\nProcessing sub-event {i+1}/{len(sub_event_list)}: {sub_event[:80]}...")
        
        writing_prompt = f"""
{story_writer.system_message}
---
# **核心设定 (Premise) - 你本次创作必须严格遵守的最高指令**
`{premise}`
---
# **当前任务**
请严格按照上面的【核心设定】，为下面的【子事件】创作故事：
`{sub_event}`
"""
        
        try:
            # 写作
            user_proxy.initiate_chat(story_writer, message=writing_prompt, max_turns=1)
            generated_story_part = story_writer.last_message().get("content", "")
            
            # 批判
            critic_prompt = f"""
{story_critic.system_message}
**请审查以下故事片段是否偏离了核心设定或逻辑**:
`{generated_story_part}`
"""
            user_proxy.initiate_chat(story_critic, message=critic_prompt, max_turns=1)
            critic_feedback = story_critic.last_message().get("content", "")
            
            print(f"Critic Feedback: {critic_feedback}")
            
            if "APPROVE" in critic_feedback.upper():
                print("Outcome: Approved.")
            else:
                print("Outcome: Disapproved, but adding to story anyway (as requested).")
            
            final_story_parts.append(generated_story_part)
        except Exception as e:
            print(f"ERROR: Failed to process sub-event '{sub_event[:50]}...' after trying all LLMs. Error: {e}")
            # 你可以选择是跳过这个子事件还是停止整个流程
            continue

    # --- 步骤 3: 保存结果 ---
    print("\n--- Step 3: Saving the final story ---")
    
    story_dir = os.path.join(output_prefix, "final_story")
    os.makedirs(story_dir, exist_ok=True)
    story_path = os.path.join(story_dir, f"final_story_{id}.json")
    with open(story_path, "w", encoding="utf-8") as file:
        json.dump(final_story_parts, file, ensure_ascii=False, indent=4)
        print(f"INFO: Final story saved to {story_path}")

    return "finish"

import copy
import json
import os
import re
from typing import Dict, List, Tuple

import autogen
from autogen import AssistantAgent, ConversableAgent
from autogen.agentchat.contrib.capabilities import transform_messages
from openai import OpenAI


# ==============================================================================
# 1. 配置信息 (Configuration)
# ==============================================================================
# 使用腾讯混元大模型作为示例

url = "https://api.hunyuan.cloud.tencent.com/v1"
config_list = [
    {
        "model": "hunyuan-lite",
        "base_url": url,
        "api_key": "sk-",  #  <--- 请在这里填入您的 API Key
        "api_type": "openai",
        "price": [0.001, 0.001],
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
    """
    def __init__(self):
        self.condensed_cache = {}
        # 初始化 OpenAI 客户端用于调用压缩模型
        # 请将 'sk-' 替换为您自己的 API Key
        self.client = OpenAI(
            api_key="sk-",  #  <--- 请在这里填入您的 API Key
            base_url="https://api.hunyuan.cloud.tencent.com/v1",
        )
        # 优化后的压缩指令
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

    def apply_transform(self, messages: List[Dict]) -> List[Dict]:
        temp_messages = copy.deepcopy(messages)

        # 我们只压缩中间的对话历史，保留第一条和最后两条消息的完整性
        if len(temp_messages) > 3:
            for message in temp_messages[1:-2]:
                content = message.get("content", "")
                message_id = hash(content)

                # 仅当消息内容足够长时才进行压缩
                if len(content) > 150:
                    if message_id in self.condensed_cache:
                        print(f"INFO: Using cached condensed message for role: {message['role']}")
                        message["content"] = self.condensed_cache[message_id]
                    else:
                        print(f"INFO: Compressing message for role: {message['role']}...")
                        try:
                            full_prompt = self.prompt_template + content
                            completion = self.client.chat.completions.create(
                                model="hunyuan-lite",
                                messages=[{"role": "user", "content": full_prompt}],
                            ).choices[0].message.content
                            
                            message["content"] = completion
                            self.condensed_cache[message_id] = completion
                            print("INFO: Compression successful.")
                        except Exception as e:
                            print(f"ERROR: Could not compress message due to: {e}")
                            # 压缩失败则使用原内容
                            pass
                            
        return temp_messages

    def get_logs(self, pre_transform_messages: List[Dict], post_transform_messages: List[Dict]) -> Tuple[str, bool]:
        # 检查消息是否被修改过，用于日志记录
        for i in range(len(pre_transform_messages)):
            if pre_transform_messages[i].get("content") != post_transform_messages[i].get("content"):
                return "Message content was condensed.", True
        return "No message content was changed.", False

# 实例化消息转换处理器
redact_handling = transform_messages.TransformMessages(transforms=[MessageRedact()])


# ==============================================================================
# 3. 核心功能函数 (Core Generation Functions)
# ==============================================================================

def event_generate(premise: str, id: str, output_prefix: str) -> str:
    """
    根据故事大纲生成一系列结构化的事件。
    """
    # 角色1: 事件生成器
    event_creator = ConversableAgent(
        name="EventCreator",
        system_message="""
你是一位富有创意的故事策划师。你的任务是根据给定的故事大纲，设计出一系列结构完整、富有戏剧性的事件。

请严格按照以下格式为每个事件生成梗概，并确保内容翔实且引人入胜：

**事件**: [为事件起一个简洁的标题，例如：意外的相遇]
**场景**: [描述事件发生的时间、地点和环境氛围，例如：一个暴雨的夜晚，在城市边缘一家即将打烊的咖啡馆里]
**角色**: [列出参与此事件的核心角色]
**行动**: [描述角色在此场景中的具体行为和对话]
**冲突**: [明确指出此事件中的核心矛盾点，是角色之间的、角色与环境的，还是角色内心的挣扎]
**情节转折**: [设计一个出人意料的转折或发现，推动故事向前发展]

请确保每个事件都紧扣故事大纲，并为后续事件的发生埋下伏笔。
""",
        llm_config={"config_list": config_list, "temperature": 1.0},
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
    )

    # 角色2: 事件审核员
    event_auditor = ConversableAgent(
        name="EventAuditor",
        system_message="""
你是一位经验丰富的剧本医生。你的任务是评估上一个代理生成的事件梗概是否合格。

**评估标准:**
1.  **逻辑性**: 事件的起因、经过、结果是否符合逻辑？情节转折是否突兀？
2.  **戏剧性**: 冲突是否足够强烈？情节是否引人入胜？
3.  **一致性**: 事件是否与故事大纲的核心设定保持一致？
4.  **完整性**: 是否严格遵循了要求的格式（事件、场景、角色、行动、冲突、情节转折）？

**指令:**
- 如果事件在以上四个方面都表现出色，请直接回复 `TERMINATE`。
- 如果事件存在明显逻辑漏洞、缺乏戏剧性或不符合大纲，请回复 `重新生成`，并简要说明需要改进的核心问题，例如：“重新生成：冲突不够激烈，情节转折缺乏惊喜。”
""",
        llm_config={"config_list": config_list, "temperature": 0.4},
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
    )

    try_count = 0
    while try_count < 5:
        try:
            print(f"INFO: Attempting to generate events, try #{try_count + 1}...")
            initial_message = f"故事大纲: {premise}"
            event_auditor.initiate_chat(event_creator, message=initial_message, max_turns=10)
            print("INFO: Event generation successful.")
            break
        except Exception as e:
            try_count += 1
            print(f"WARNING: Event generation failed with error: {e}. Retrying...")
            if try_count >= 5:
                print(f"ERROR: Gave up after {try_count} attempts. Returning None.")
                return "None"

    # 提取并保存结果
    messages = [entry['content'] for entries in event_auditor.chat_messages.values() for entry in entries]
    output_dir = os.path.join(output_prefix, "final_events")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"final_events_{id}.json")
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(messages, file, ensure_ascii=False, indent=4)
        
    final_event_list = event_auditor.chat_messages[event_creator][-2]['content']
    return final_event_list


def event_extract(text_data: str, id: str, output_prefix: str) -> str:
    """
    从生成的文本中抽取出事件描述。
    """
    print("INFO: Extracting structured events from text...")
    # 正则表达式用于匹配以“事件:”或“Event”开头的段落
    event_pattern = r"(?i)(?:\*\*事件\*\*|Event)[\s:]*.*?(?=\n\n(?i)(?:\*\*事件\*\*|Event)|$)"
    events = re.findall(event_pattern, text_data, re.DOTALL)

    if events:
        print(f"INFO: Extracted {len(events)} events.")
    else:
        print("WARNING: No events found in the text.")
        return ""

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
    [已重构以解决 system message 顺序问题]
    """
    print("INFO: Story generation process started using a sequential workflow.")

    # --- 代理定义 (保持不变) ---
    # 我们将代理的指令暂时存储在system_message中，但在调用时会手动处理。
    sub_events_divider = ConversableAgent(
        name="SubEventsDivider",
        system_message="""你是一个专业的叙事结构设计师。你的任务是将给定的“事件”分解成多个逻辑递进、情节更细化的“子事件”。确保划分后的子事件共同构成一个完整的小型叙事弧（开端-发展-高潮），并严格遵循输出格式。""",
        llm_config={"config_list": config_list, "temperature": 0.5},
        human_input_mode="NEVER",
    )

    story_writer = ConversableAgent(
        name="StoryWriter",
        system_message="""你是一位才华横溢的小说家。你的任务是根据给定的子事件，创作出一段生动、细腻、引人入胜的故事。
**写作要求:**
1. **聚焦当前**: 你的创作内容必须仅仅围绕当前指定的子事件展开。
2. **生动描写**: 运用感官细节、角色心理活动和精彩的对话来丰满故事情节。
3. **输出格式**: 直接输出故事文本，不要添加任何标题或标签。""",
        llm_config={"config_list": config_list, "temperature": 1.0},
        human_input_mode="NEVER",
    )
    redact_handling.add_to_agent(story_writer) # 为写手启用上下文压缩

    story_critic = ConversableAgent(
        name="StoryCritic",
        system_message=f"""你是一名严格的剧本监督。你的唯一职责是确保故事严格按照预设的核心设定（Premise）和子事件发展。
**核心设定 (Premise)**: {premise}
**指令**:
- 如果故事内容符合要求，请回复 `APPROVE`。
- 如果内容发生偏离，请回复 `REWRITE:` 并附上具体的修改意见。""",
        llm_config={"config_list": config_list, "temperature": 0.2},
        human_input_mode="NEVER",
    )

    # --- 步骤 1: 划分出子事件 ---
    print("\n--- Step 3.1: Dividing events into sub-events ---")
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
    # 创建一个临时的“用户代理”来发起对话
    user_proxy = ConversableAgent("user_proxy", human_input_mode="NEVER", llm_config=False)
    
    try:
        user_proxy.initiate_chat(sub_events_divider, message=sub_events_prompt, max_turns=1)
        sub_events_text = sub_events_divider.last_message()["content"]
        print("INFO: Sub-events created successfully.")
        # 在这里可以添加解析sub_events_text的代码，将其转换为列表
    except Exception as e:
        print(f"ERROR: Failed to create sub-events: {e}")
        return "stop"
    
    # (为了简化，我们假设模型完美输出了子事件，并直接将其用于下一步)
    # 在生产环境中，您需要用正则表达式或更可靠的方法来解析这里的 'sub_events_text'
    
    # --- 步骤 2: 循环生成和批判每个子事件的故事 ---
    # 这是最关键的部分，我们模拟一个简单的写作-批判循环
    print("\n--- Step 3.2: Writing and critiquing story for each sub-event ---")
    final_story_parts = []
    
    # 假设的子事件列表 (需要从sub_events_text中解析)
    # 这里用一个简单的例子代替真实的解析逻辑
    # 您需要实现一个函数来从 'sub_events_text' 提取出这样的列表
    sub_event_list = [line for line in sub_events_text.split('\n') if re.match(r'事件\d\.\d:', line)]
    if not sub_event_list:
        print("ERROR: Could not parse sub-events from the output. Using raw output.")
        sub_event_list = [sub_events_text] # Fallback

    for i, sub_event in enumerate(sub_event_list):
        print(f"\nProcessing sub-event {i+1}/{len(sub_event_list)}: {sub_event[:50]}...")
        
        # 为了避免system message问题，我们将指令嵌入到user message中
        writing_prompt = f"""
{story_writer.system_message}

**当前任务**: 请根据以下子事件进行创作：
`{sub_event}`
"""
        
        # 写作
        user_proxy.initiate_chat(story_writer, message=writing_prompt, max_turns=1)
        generated_story_part = story_writer.last_message()["content"]
        
        # 批判
        critic_prompt = f"""
{story_critic.system_message}

**请审查以下故事片段是否偏离了核心设定或逻辑**:
`{generated_story_part}`
"""
        user_proxy.initiate_chat(story_critic, message=critic_prompt, max_turns=1)
        critic_feedback = story_critic.last_message()["content"]
        
        print(f"Critic Feedback: {critic_feedback}")
        
        if "APPROVE" in critic_feedback.upper():
            print("Outcome: Approved.")
            final_story_parts.append(generated_story_part)
        else:
            print("Outcome: Disapproved, but adding to story anyway (without REJECTED tag).")
            # 关键改动：无论是否通过，都直接添加原始的故事片段
            final_story_parts.append(generated_story_part)

    # --- 步骤 3: 保存结果 ---
    print("\n--- Step 3.3: Saving the final story ---")
    
    # 保存最终故事
    story_dir = os.path.join(output_prefix, "final_story")
    os.makedirs(story_dir, exist_ok=True)
    story_path = os.path.join(story_dir, f"final_story_{id}.json")
    with open(story_path, "w", encoding="utf-8") as file:
        json.dump(final_story_parts, file, ensure_ascii=False, indent=4)
        print(f"INFO: Final story saved to {story_path}")

    return "finish"
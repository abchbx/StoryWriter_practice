from urllib.parse import unquote
from typing import Annotated, Literal
import copy
import pprint
import re
import os
import json
import autogen
from autogen import AssistantAgent
from autogen import ConversableAgent
from typing import Dict, List, Tuple
from autogen.agentchat.contrib.capabilities import transform_messages, transforms
from autogen.agentchat.contrib.capabilities.text_compressors import LLMLingua
from autogen.agentchat.contrib.capabilities.transforms import TextMessageCompressor
from openai import OpenAI

# 建议：将API密钥存储在环境变量中，而不是直接写在代码里，这样更安全。
# 在您的系统中设置环境变量 'TENCENT_API_KEY'，然后代码就可以安全地读取它。
# 例如，在Linux/macOS中: export TENCENT_API_KEY='your_secret_key'
# 在Windows中: set TENCENT_API_KEY='your_secret_key'
TENCENT_API_KEY = os.environ.get("TENCENT_API_KEY", "sk-...") # 如果未设置环境变量，则使用旧的密钥作为备用


url = "https://api.hunyuan.cloud.tencent.com/v1"

config_list = [
    {
        "model":"hunyuan-lite",
        "base_url": url,
        'api_key': TENCENT_API_KEY,
        'api_type':"openai",
        'price' : [0.001,0.001]
    },
]


condensed_cache = {}


class MessageRedact:
    def __init__(self):
        self._openai_key_pattern = r"sk-([a-zA-Z0-9]{48})"
        self._replacement_string = "REDACTED"
        self.condensed_cache = {}
        # 将OpenAI客户端的初始化移到构造函数中，提高效率
        try:
            self.client = OpenAI(
                api_key=TENCENT_API_KEY,
                base_url="https://api.hunyuan.cloud.tencent.com/v1",
            )
        except Exception as e:
            print(f"Error initializing OpenAI client in MessageRedact: {e}")
            self.client = None

    def apply_transform(self, messages: List[Dict]) -> List[Dict]:
        if not self.client:
            return messages

        temp_messages = copy.deepcopy(messages)
        
        if len(temp_messages) > 2:
            for message in temp_messages[1:-2]:  
                if message.get("content") and isinstance(message["content"], str) and len(message["content"]) > 150:
                    message_id = hash(message["content"])
                    
                    if message_id in self.condensed_cache:
                        print(f"Using cached condensed message for: {message['role']}")
                        message["content"] = self.condensed_cache[message_id]
                    else:
                        prompt = "请将以下对话内容浓缩至其原始长度的15%，保留核心信息、关键决策和情节转折。确保浓缩后的版本能清晰地传达对话的核心进展。"
                        print(f"Condensing message for: {message['role']}")
                        try:
                            completion = self.client.chat.completions.create(
                                model="hunyuan-lite",
                                messages=[{"role": "user", "content": f"{prompt}\n\n{message['content']}"}],
                            ).choices[0].message.content
                            
                            message["content"] = completion
                            self.condensed_cache[message_id] = completion
                        except Exception as e:
                            print(f"Could not condense message due to an error: {e}")

        return temp_messages

    def get_logs(self, pre_transform_messages: List[Dict], post_transform_messages: List[Dict]) -> Tuple[str, bool]:
        for i in range(len(pre_transform_messages)):
            if pre_transform_messages[i]["content"] != post_transform_messages[i]["content"]:
                return "Message condensed", True
        return "No condensation", False

redact_handling = transform_messages.TransformMessages(transforms=[MessageRedact()])


episode_edit = ConversableAgent(name="episode_edit_model",
                      system_message="你是一个顶级的剧本编辑，负责协调和管理整个故事创作流程。",
                      llm_config={
                          "config_list": config_list,
                          "temperature": 1,
                      },
                      is_termination_msg=lambda msg: msg.get("content") is not None and "Epilogue" in msg["content"],
                      )

def event_generate(premise, task_id, output_prefix):
    event = ConversableAgent(name="event_model",
                        system_message="""你是一位专业的编剧。请根据故事大纲，创造性地生成一系列连贯且有趣的事件梗概。请严格按照“事件 1: [内容]”、“事件 2: [内容]”的格式输出，每个事件另起一段。""",
                        llm_config={
                            "config_list": config_list,
                            "temperature": 1,
                        },
                        human_input_mode="NEVER",
                        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
                        )
    
    event_rerank = ConversableAgent(name="event_rerank_model",
                        system_message="""你是一位严谨的事件评估师。请评估所生成的事件是否连贯、合理且有趣。
- 如果事件质量高、逻辑清晰，请只回复 “TERMINATE”。
- 如果事件存在问题（例如，逻辑不通、与大纲矛盾），请不要只说“重新生成”。请提出具体的修改建议，并以“重新生成：[你的修改建议]”开头，以指导优化。""",
                        llm_config={
                            "config_list": config_list,
                            "temperature": 1,
                        },
                        human_input_mode="NEVER", 
                        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
                        )
    
    premise = premise     
    try_count = 0
    while try_count < 5:
        try:
            print(f"尝试生成事件，第 {try_count + 1} 次")
            event_rerank.initiate_chat(event, message=f"请根据以下故事大纲生成5个关键事件：\n\n{premise}", max_turns=5)
            break
        except Exception as e:
            try_count += 1
            if try_count >= 5:
                print(f"尝试 {try_count} 次后放弃生成事件。错误: {e}")
                return None
    
    final_events_message = ""
    chat_history = event_rerank.chat_messages.get(event, [])
    if len(chat_history) > 1:
        final_events_message = chat_history[-2]['content']
    
    output_dir = os.path.join(output_prefix, "final_events")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"final_events_{task_id}.txt")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(final_events_message)
    return final_events_message


def event_extract(text_data, task_id, output_prefix):
    if not text_data:
        print("没有事件文本可供提取。")
        return ""
    
    event_pattern = re.compile(r'(?:事件|Event)\s*\d+\s*[:：](.*?)(?=(?:\n\n(?:事件|Event)\s*\d+\s*[:：])|\Z)', re.DOTALL | re.IGNORECASE)
    
    events = event_pattern.findall(text_data)
    
    if not events:
        print("未找到匹配格式的事件。将尝试按行分割。")
        events = [line for line in text_data.split('\n') if line.strip()]

    print("提取到的事件:")
    for idx, event in enumerate(events, 1):
        print(f"事件 {idx}: {event.strip()}")

    event_data = {"events": [{"event": event.strip()} for event in events]}

    output_dir = os.path.join(output_prefix, "final_process")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"events_{task_id}.json")

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(event_data, json_file, ensure_ascii=False, indent=4)

    event_str = "\n".join(f"事件 {i+1}: {e['event']}" for i, e in enumerate(event_data["events"]))

    print("处理后的事件字符串:\n", event_str)
    return event_str

def story_generate(premise, event_str, task_id, output_prefix):
    story_writer = ConversableAgent(name="story_writer_model",
                                    system_message="""你是一位富有创意的小说家。你的任务是根据给定的子事件，创作出引人入胜的故事情节。请严格遵循收到的指令，一次只写一个子事件的故事。在完成最后一个子事件的写作后，回复“TERMINATE”。在写作时，请确保：
1.  **紧扣主题**：内容与核心设定（Premise）和当前子事件紧密相关。
2.  **生动描写**：运用丰富的细节和感官描写，让场景和人物活起来。
3.  **推动情节**：确保每个子事件都对整个故事的进展有所贡献。
4.  **避免重复**：不要在当前或之前的回复中重复已经写过的情节。""",
                                    llm_config={ "config_list": config_list, "temperature": 1 }, # 保持作者的创造性
                                    human_input_mode="NEVER",
                                    )
    redact_handling.add_to_agent(story_writer)
    
    sub_events = ConversableAgent(name="sub_events_spliter",
                                    system_message="你是一个专业的事件划分器。你的任务是将每个主事件精确地划分为几个逻辑连贯、循序渐进的子事件。请确保划分后的子事件能够清晰地展示情节的逐步发展。",
                                    llm_config={ "config_list": config_list, "temperature": 0.5 },
                                    human_input_mode="NEVER",
                                    )
    
    disperse_events = ConversableAgent(name="narrative_architect",
                                    system_message="""你是一位经验丰富的叙事结构师。你的任务是将一系列子事件组织成结构合理的章节，以达到最佳的叙事效果。请考虑叙事节奏和悬念设置，并确保逻辑连贯。""",
                                    llm_config={ "config_list": config_list, "temperature": 0.5 },
                                    human_input_mode="NEVER",
                                    max_consecutive_auto_reply=1, 
                                    )

    # ##################################################################
    # ########                方案一：核心修改区域                  ########
    # ##################################################################
    story_critic = ConversableAgent(name="story_critic_editor",
                                    system_message=f"""你是专业的剧本医生和故事编辑。你的核心职责是确保故事严格按照核心设定（Premise）和事件大纲发展。
故事的核心设定（Premise）是：{premise}

当你收到故事作者的稿件后，请进行评估：

1.  **评估内容**: 判断故事是否偏离主题、情节是否连贯、有无重复内容。
2.  **提供反馈**:
    - **如果内容合格**: 请直接提供下一个要写作的子事件，格式为：“下一个子事件是:{{NEXT_SUB_EVENT}}”。如果所有子事件都已完成，请回复“TERMINATE”。
    - **如果内容不合格**: 请提供具体的、可执行的修改意见，并严格按照格式回复。你的修改意见每次应尝试提供新的角度或更清晰的说明。
      格式：“RE-WRITE:[这里是你的具体修改意见。]\n下一个子事件是:{{Next_sub_event}}”

3.  **【重要】避免无限循环的退出机制**:
    - 如果你已经**连续两次**针对同一个子事件提出了修改意见，但作者依然无法写出合格内容，请不要再要求重写。
    - 在这种情况下，请强制结束当前流程，并回复 “**TERMINATE_FORCE: [这里简要说明卡住的原因，例如：作者无法在事件2.1中体现角色的转变]**”。
""",
                                    llm_config={ "config_list": config_list, "temperature": 0.2 },
                                    human_input_mode="NEVER",
                                    # 这个终止条件现在可以捕获 "TERMINATE" 和 "TERMINATE_FORCE"
                                    is_termination_msg=lambda msg: msg.get("content") is not None and ("TERMINATE" in msg["content"]),
                                    )

    try_count = 0
    while try_count < 5:
        try:
            print(f"尝试生成故事，第 {try_count + 1} 次")

            chat_results = story_critic.initiate_chats([
                {
                    "recipient": sub_events, 
                    "message": f"请将以下事件列表中的每个事件划分为3个连贯的子事件。这是故事的整体背景，请在划分时参考：\n故事大纲：{premise}\n\n事件列表：\n{event_str}",
                    "max_turns": 1,
                },
                {
                    "recipient": disperse_events, 
                    "message": "请根据你作为叙事结构师的专业知识，将这些子事件组织成章节。请按此格式回复:\n章节1:\n- 子事件:...\n- 子事件:...\n\n章节2:\n- 子事件:...\n...",
                    "max_turns": 2,
                    'summary_method': 'last_msg',
                },
                {
                    "recipient": story_writer, 
                    "message": f"你将与一位故事评论家合作。请根据他提供的核心设定和子事件来创作故事。请等待评论家给你第一个子事件，然后开始你的创作。\n核心设定(Premise): {premise}",
                    "max_turns": 50,
                },
            ])
            break
        except Exception as e:
            try_count += 1
            if try_count >= 5:
                print(f"尝试 {try_count} 次后放弃生成故事。错误: {e}")
                return "stop"  
    
    story_chat_history = story_critic.chat_messages.get(story_writer, [])
    final_story_content = [entry['content'] for entry in story_chat_history]
    
    output_filename = f"story_{task_id}.json"
    
    story_dir = os.path.join(output_prefix, "final_story")
    os.makedirs(story_dir, exist_ok=True)
    file_story_path = os.path.join(story_dir, output_filename)

    with open(file_story_path, "w", encoding="utf-8") as file:
        json.dump(final_story_content, file, ensure_ascii=False, indent=4)
        
    print(f"故事已成功生成并保存至: {file_story_path}")
    return "finish"
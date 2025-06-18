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

import re
import json
# transform_messages.MessageTransform protocol where /miniconda3/envs/autogen/lib/python3.1/site-packages/autogen/agentchat/contrib/capabilities/transforms.py
# class MessageTransform(Protocol):
#     """Defines a contract for message transformation.

#     Classes implementing this protocol should provide an `apply_transform` method
#     that takes a list of messages and returns the transformed list.
#     """

#     def apply_transform(self, messages: List[Dict]) -> List[Dict]:
#         """Applies a transformation to a list of messages.

#         Args:
#             messages: A list of dictionaries representing messages.

#         Returns:
#             A new list of dictionaries containing the transformed messages.
#         """
#         ...

#     def get_logs(self, pre_transform_messages: List[Dict], post_transform_messages: List[Dict]) -> Tuple[str, bool]:
#         """Creates the string including the logs of the transformation

#         Alongside the string, it returns a boolean indicating whether the transformation had an effect or not.

#         Args:
#             pre_transform_messages: A list of dictionaries representing messages before the transformation.
#             post_transform_messages: A list of dictionaries representig messages after the transformation.

#         Returns:
#             A tuple with a string with the logs and a flag indicating whether the transformation had an effect or not.
#         """
#         ...


url = "https://svip.xty.app/v1"
# Setup API key. Add your own API key to config file or environment variable
config_list = [
    {
        "model":"gpt-4o-mini-2024-07-18",
        "base_url": url,
        'api_key': "sk-vKuzSeyqwtWSlnFJP8WYKKiCErrMYybVODN2FtBsBBhsJj8k",
        'api_type':"openai"
    },
]

# llm_lingua = LLMLingua()
# text_compressor = TextMessageCompressor(text_compressor=llm_lingua)
condensed_cache = {}


class MessageRedact:
    def __init__(self):
        self._openai_key_pattern = r"sk-([a-zA-Z0-9]{48})"
        self._replacement_string = "REDACTED"
        self.if_prod_first=False
        self.condensed_cache = {} 

    def apply_transform(self, messages: List[Dict]) -> List[Dict]:
        temp_messages = copy.deepcopy(messages)
        
        if len(temp_messages)>2:
            client = OpenAI(
                        api_key="sk-vKuzSeyqwtWSlnFJP8WYKKiCErrMYybVODN2FtBsBBhsJj8k",
                        base_url="https://svip.xty.app/v1",
                    )
            for message in temp_messages[1:-2]:  
                # message_id = hash(message["content"])
                message_id = hash(message["content"])
                prompt="Please condense the following story to 10% percent of its original length, retaining the key plot points, character conflicts, and the twist or resolution at the end. Omit lengthy descriptions of the setting and minor dialogues, ensuring the condensed version still clearly conveys the core progression of the story."
                print(message["role"],len(message["content"]))
                
                if len(message["content"])>150:
                    if message_id in self.condensed_cache:
                        print(f"Using cached condensed message for: {message['role']}")
                        message["content"] = self.condensed_cache[message_id]
                    else:
                        message["content"]=completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18",
                            messages=[{"role": "user", "content": prompt+"\n"+message["content"]}],
                            ).choices[0].message.content
                        self.condensed_cache[message_id] = completion
                # if isinstance(message["content"], str):
                #     if not re.search(r"\bEvent\b", message["content"], re.IGNORECASE):
                #         message["content"]="" 
                # elif isinstance(message["content"], list):
                #     
                #     message["content"] = [
                #         item for item in message["content"]
                #         if item["type"] == "text" and re.search(r"\bEvent\b", item["text"], re.IGNORECASE)
                #     ]
        print("temp_messages:",temp_messages)
        # file_path = os.path.join("./writer_input")
        # print(file_path)
        # with open(file_path+"/input.json", "w") as file:
        #     json.dump(temp_messages[0]["content"], file, ensure_ascii=False, indent=4)
        return temp_messages
    # def get_logs(self, pre_transform_messages: List[Dict], post_transform_messages: List[Dict]) -> Tuple[str, bool]:
    #     keys_redacted = self._count_redacted(post_transform_messages) - self._count_redacted(pre_transform_messages)
    #     if keys_redacted > 0:
    #         return f"Redacted {keys_redacted} OpenAI API keys.", True
    #     return "", False
    def get_logs(self, pre_transform_messages: List[Dict], post_transform_messages: List[Dict]) -> Tuple[str, bool]:
        for i in range(len(pre_transform_messages)):
            if pre_transform_messages[i]["content"]!=post_transform_messages[i]["content"]:
                return "temporary",True

        return "temporary",False

    # def _count_redacted(self, messages: List[Dict]) -> int:
    #     # counts occurrences of "REDACTED" in message content
    #     count = 0
    #     for message in messages:
    #         if isinstance(message["content"], str):
    #             if "REDACTED" in message["content"]:
    #                 count += 1
    #         elif isinstance(message["content"], list):
    #             for item in message["content"]:
    #                 if isinstance(item, dict) and "text" in item:
    #                     if "REDACTED" in item["text"]:
    #                         count += 1
    #     return count
redact_handling = transform_messages.TransformMessages(transforms=[MessageRedact()])


# event = ConversableAgent(name="event model",
#                      system_message="You need to play a role that Generate an outline for one event at a time based on the premise.",
#                      llm_config={
#                          "config_list": config_list,
#                          "temperature": 1,
#                      },
#                     #  max_consecutive_auto_reply=1,
#                      is_termination_msg=lambda msg: msg.get("content") is not None and "THE END" in msg["content"],
#                      )
# redact_handling.add_to_agent(event)
# event_rerank = ConversableAgent(name="event_rerank model",
#                      system_message="You are a story outlines critics, and you accept the event writer and rate the event based on the premise,if the event is not good, you should ask the event model to regenerate a new one. Give THE END when the event come to the end of the premise",
#                      llm_config={
#                          "config_list": config_list,
#                          "temperature": 1,
#                      },
#                     #  max_consecutive_auto_reply=1,
#                      is_termination_msg=lambda msg: msg.get("content") is not None and "THE END" in msg["content"],
#                      )
episode_edit = ConversableAgent(name="episode_edit_model",
                     system_message="",
                     llm_config={
                         "config_list": config_list,
                         "temperature": 1,
                     },
                    #  max_consecutive_auto_reply=1, 
                     is_termination_msg=lambda msg: msg.get("content") is not None and "Epilogue" in msg["content"],
                     )
# read the events generated by the event model



# big = AssistantAgent(name="event model",
#                      max_consecutive_auto_reply=100,
#                      system_message="Act as a literary-judger. You must ask the writer to create a 10k-long story. If it is not long enough, you should ask him to expand the story",
#                      llm_config={
#                          "config_list": config_list,
#                          "temperature": 1,
#                      })

# save the chat_history of the event generate   
def event_generate(premise,id,output_prefix):
    event = ConversableAgent(name="event_model",
                     system_message="""Generate an event outline based on the premise. Here is an example: Event n: {EVENT1} Setting:{SETTING}\nCharacter:{CHARACTER}\nAction:{ACTION}\nConflict:{CONFLICT}\nPlot Twist:{PLOT_TWIST}\n""",
                     llm_config={
                         "config_list": config_list,
                         "temperature": 1,
                     },
                     human_input_mode="NEVER",
                    #  max_consecutive_auto_reply=1,
                     is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
                     )
    event_rerank = ConversableAgent(name="event_rerank_model",
                     system_message="Check if the events structrue is coherent and make sense.If not, Reply re-generate.If the events are good and coherent enough reply TERMINATE when the events are good and coherent enough",
                     llm_config={
                         "config_list": config_list,
                         "temperature": 1,
                     },
                     human_input_mode="NEVER",
                    #  max_consecutive_auto_reply=1, 
                     is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
                     )
    premise = premise             
    try_count = 0
    while try_count < 5:
        try:

            print(f"try to generate events for the {try_count} time")

            event_rerank.initiate_chat(event, message="Premise:"+premise,detail="aa",max_turns=100)
            break
        except Exception as e:
            try_count += 1
            if try_count >= 5:
                print(f"give up after {try_count} time")
                return "None" #
    
    message = [entry['content'] for entries in event_rerank.chat_messages.values() for entry in entries]
    output_path = output_prefix+"/final_events/final_events_{}.json".format(str(id))
    with open(output_path, "w") as file:
        json.dump(message, file, ensure_ascii=False, indent=4)
    return message

def event_extract(text_data,id,output_prefix):
    print(text_data)
    event_pattern = r'(Event \d+:.*?)(?=\n\n|$)'  

    events = []
    for text in text_data:
        matches = re.findall(event_pattern, text, re.DOTALL)
        events.extend(matches)

    if events:
        print("Extracted Events:\n")
        for idx, event in enumerate(events, 1):
            print(f"Event {idx}:\n{event}\n")
    else:
        print("No events found.")


    event_data = {"events": [{"event": event.strip()} for event in events]}

    output_path = output_prefix+"/final_process/events_{}.json".format(str(id))
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(event_data, json_file, ensure_ascii=False, indent=4)
    # return event_data
    
    # exit()
# event_history = []
# with open('./events.json', 'r', encoding='utf-8') as file:
#      event_history= json.load(file)
    event_str = "\n".join(event["event"] for event in event_data["events"])

    print(event_str)
    return event_str

def story_generate(event_str,id,output_prefix):
    # print("???????????????",event_str)
    episode_critc = ConversableAgent(name="episode_critc_model",
                     system_message="Focus on the relationships between events, and don’t let the story deviate from the outlines.",
                     llm_config={
                         "config_list": config_list,
                         "temperature": 0.4,
                     },
                     human_input_mode="NEVER",
                    #  max_consecutive_auto_reply=1,  
                     is_termination_msg=lambda msg: msg.get("content") is not None and ("Epilogue" in msg["content"] or "TERMINATE" in msg["content"]),
                     )

    story_writer = ConversableAgent(name="story_writer_model",
                        system_message="""Generate a story for one sub-event at a time in sequence in this format: Story1.1{Story1.1}. Reply "TERMINATE" only when it comes to the last sub-event.""",
                        llm_config={
                            "config_list": config_list,
                            "temperature": 1,
                        },
                        human_input_mode="NEVER",
                        #  max_consecutive_auto_reply=1,  
                        # is_termination_msg=lambda msg: msg.get("content") is not None and "THE END" in msg["content"],
                        )
    redact_handling.add_to_agent(story_writer)
    sub_events = ConversableAgent(name="sub_events",
                        system_message="You are an events divider.Your task is to divide each provided event into several sub events",
                        llm_config={
                            "config_list": config_list,
                            "temperature": 0.5,# todo
                        },
                        human_input_mode="NEVER",
                        #  max_consecutive_auto_reply=1,  
                        # is_termination_msg=lambda msg: msg.get("content") is not None and "THE END" in msg["content"],
                        )
    disperse_events = ConversableAgent(name="disperse_events",
                        system_message="I have a series of events that have been broken down into smaller sub-events. Your task is to distribute these sub-events across different chapters. Some chapters might focus entirely on one event, while others might weave together multiple events. The key is to ensure the narrative flow remains coherent and respects the logical relationships between the events. The distribution should also build suspense and maintain reader engagement throughout the story.",
                        llm_config={
                            "config_list": config_list,
                            "temperature": 0.5,# todo
                        },
                        human_input_mode="NEVER",
                         max_consecutive_auto_reply=1, 
                        # is_termination_msg=lambda msg: msg.get("content") is not None and "THE END" in msg["content"],
                        )
    chapter_writer = ConversableAgent(name="chapter_writer",
                        system_message="Your task is to generate the chapters for the story according to the given text.",
                        llm_config={
                            "config_list": config_list,
                            "temperature": 1,# todo
                        },
                        human_input_mode="NEVER",
                        #  max_consecutive_auto_reply=1,  
                        is_termination_msg=lambda msg: msg.get("content") is not None and "THE END" in msg["content"],
                        )
    story_critic = ConversableAgent(name="story_critic",
                        system_message="""You need to ensure that the story generation agent does not deviate from the outline. If it deviates, please rewrite or revise this sub-event in the following format: "**RE-WRITE**:{RE-WRITE}\n Next sub-event is:{Next_sub_event}".If there is no deviation, please reply Next sub-event:{NEXT_SUB_EVENT}""",
                        llm_config={
                            "config_list": config_list,
                            "temperature": 0.2,# todo
                        },
                        human_input_mode="NEVER",
                        #  max_consecutive_auto_reply=1,  
                        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
                        )
    
    try_count = 0
    while try_count < 5:
        try:

            print(f"generating story for {try_count} time")

            chat_results = episode_critc.initiate_chats([
                {
                    "recipient":sub_events, 
                    "message":"please review the events and divide them into 3 sub-events for each event in form of Event1: Event 1.1:{Event 1.1}\nEvent 1.2{Event 1.2}\nEvent 1.3{Event 1.3}\nEvent2: Event 2.1{Event 2.1}\n Event 2.2{Event 2.2}\n"+event_str,
                    "max_turns":1,},
                {
                    "recipient":disperse_events, 
                    "message":"reply in this form: Chapter1:{Chapter1}\n Sub-event:{Sub-event}\nSub-event:{Sub-event}\nSub-event:{Sub-event}\n",
                    "max_turns":2,
                    'summary_method': 'last_msg',},
                {
                    "sender":story_critic,
                    "recipient":story_writer, 
                    "message":"generate one sub-story at a time. Do not reply other words",
                    "max_turns":50,},
            ])
            break
        except Exception as e:
            try_count += 1
            if try_count >= 5:
                print(f"give up after {try_count} attempts")
                return "stop"  
    
        
    # print(chat_results[0].summary)
    message=[entry['content'] for entries in episode_critc.chat_messages.values() for entry in entries]
    message_story=[entry['content'] for entries in story_critic.chat_messages.values() for entry in entries]
    stt="story"+id
    stt+=".json"
    file_path = os.path.join(output_prefix+"/final_outlines", stt)
    print(file_path)
    with open(file_path, "w") as file:
        json.dump(message, file, ensure_ascii=False, indent=4)
    
    file_story_path=os.path.join(output_prefix+"/final_story", stt)
    with open(file_story_path, "w") as file:
        json.dump(message_story, file, ensure_ascii=False, indent=4)
    return "finish"
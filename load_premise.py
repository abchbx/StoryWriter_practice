import json
import os, sys
## ablation experiment:change the path of this
sys.path.append("/HDD_DATA/HDD_HOME/xiaht/ablation/no_all_re.py")
import sys
import no_all_re as agent_try
# import agent_try
# ./complete.jsonl todo

with open("/HDD_DATA/HDD_HOME/xiaht/autogen/autogen/moderate.jsonl", "r", encoding="utf-8") as file:
    data = [json.loads(line) for line in file]

num=0
for record in data:
    # os.remove("/HDD_DATA/HDD_HOME/xiaht/autogen/autogen/cache/cache.db")

    id=record["id"]
    # if id=="8e997abf-0a45-4be9-8197-1781759b0a0c" or id == "3245a711-f3de-4338-a421-22e1fc93b58f" or id ==  "ca060c96-65c5-49cd-8e00-113eef88ab76":
    #     continue
    num+=1
    premise=record["premise"]
    if os.path.exists("./output/"+"final_events/final_events_"+id+"json"):
        with open("./output/"+"final_events/final_events_"+id+"json", "r", encoding="utf-8") as file:
            message=file
    else:
        message=agent_try.event_generate(premise,id,"./output/"+sys.argv[1])
        if message=="None":
            with open("./output/{}".format(sys.argv[1])+'/failed_attempts.txt', 'w') as file:
                file.write(f"event_id: {id} error\n")
            continue

    # if os.path.exists("./output/"+"final_process/events_"+id+"json"):
    #     with open("./output/"+"final_process/events_"+id+"json", "r", encoding="utf-8") as file:
    #         message=file
    # else:
    events=agent_try.event_extract(message,id,"./output/"+sys.argv[1])

    if os.path.exists("./output/"+"final_story/story"+id+"json"):
        continue
    story=agent_try.story_generate(events,id,"./output/"+sys.argv[1])
    if story=="stop":
        with open("./output/{}".format(sys.argv[1])+'/failed_attempts.txt', 'w') as file:
            file.write(f"story_id: {id} error\n")
        continue
    print("{}th story".format(num))
# import os

# def get_files(directory,mm):
#     files = []
#     for dirpath, dirnames, filenames in os.walk(directory):
#         for file in filenames:
#             if mm==1:
#                 # print(os.path.join(dirpath, file)[40:])
#                 files.append(os.path.join(dirpath, file)[52:])
#             else:
#                 # print(os.path.join(dirpath, file)[52:])
#                 files.append(os.path.join(dirpath, file)[53:])
#     return files

# a_files = set(get_files('/HDD_DATA/HDD_HOME/xiaht/autogen/autogen/story',1))
# a_files.append(get_files())
# b_files = set(get_files('/HDD_DATA/HDD_HOME/xiaht/autogen/autogen/story2',2))
# print(a_files)
# difference = b_files - a_files
# print(difference)
# if not difference:
#     print("file b in file a")
# else:
#     print("file b not in file a:")
#     for file in difference:
#         print(file)
# diff=a_files-b_files
# print(len(diff))
# # print(b_files)
# exit()
# folder_path = "./final_events"
# for filename in os.listdir(folder_path):
#     # print(filename[17:])
    
#     if (filename[17:] not in diff):
#         continue
#     filename="processed_before_b3390991-28dd-4fb1-9b02-58b9da1f2c7d.json"
#     os.remove("./cache/cache.db")
#     file_path = os.path.join(folder_path, filename)
#     with open(file_path, 'r', encoding='utf-8') as file:
#         data = json.load(file)
#     str=""
#     for item in data:
#         str+=item+"\n\n\n"
    # print(str)
    # print(filename)
    # exit()
    # print(filename[17:-5])
    # exit()
    # story=agent_try.story_generate(str,filename[17:-5])

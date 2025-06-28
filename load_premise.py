import json
import os, sys
## ablation experiment:change the path of this
# sys.path.append("/HDD_DATA/HDD_HOME/xiaht/ablation/no_all_re.py") # This line might need to be adjusted based on your project structure
import sys
# import no_all_re as agent_try # This line might need to be adjusted based on your project structure
import agent_try

# Make sure the path to the JSONL file is correct
with open("/workspace/StoryWriter/LongStory/premise.jsonl", "r", encoding="utf-8") as file:
    data = [json.loads(line) for line in file]

num=0
for record in data:
    id=record["id"]
    premise=record["premise"]
    
    # Define output directories based on the script argument
    output_base_dir = os.path.join("./output", sys.argv[1])
    final_events_path = os.path.join(output_base_dir, "final_events", f"final_events_{id}.json")
    final_story_path = os.path.join(output_base_dir, "final_story", f"story{id}.json")
    failed_attempts_path = os.path.join(output_base_dir, 'failed_attempts.txt')

    num+=1
    
    # Ensure the output directories exist
    os.makedirs(os.path.dirname(final_events_path), exist_ok=True)
    os.makedirs(os.path.dirname(final_story_path), exist_ok=True)


    if os.path.exists(final_events_path):
        print(f"DEBUG: Loading file: {final_events_path}") # <--- 添加这行来调试
        with open(final_events_path, "r", encoding="utf-8") as file:
            message=json.load(file)
    else:
        message=agent_try.event_generate(premise,id,output_base_dir)
        if message=="None":
            with open(failed_attempts_path, 'a') as file:
                file.write(f"event_id: {id} error\n")
            continue

    events=agent_try.event_extract(message,id,output_base_dir)

    if os.path.exists(final_story_path):
        continue
    
    # MODIFICATION START: Passed 'premise' to the story_generate function
    story=agent_try.story_generate(premise, events, id, output_base_dir)
    # MODIFICATION END

    if story=="stop":
        with open(failed_attempts_path, 'a') as file:
            file.write(f"story_id: {id} error\n")
        continue
        
    print(f"{num}th story processed.")
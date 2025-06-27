import os

total_files_count = 0

for i in range(6):
    folder_name = str(i)
    final_story_path = os.path.join("./output/"+folder_name, 'final_story')

    if os.path.exists(final_story_path) and os.path.isdir(final_story_path):
        for root, dirs, files in os.walk(final_story_path):
            total_files_count += len(files)

print(f"final_story total: {total_files_count}")

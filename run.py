import subprocess

processes = []
for i in range(5):
    process = subprocess.Popen(["python", "./load_premise.py", str(i)])
    processes.append(process)
    print(f"{i}.py running\n\n")

# 可选：等待所有子进程完成
for process in processes:
    process.wait()
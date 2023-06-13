import subprocess
import matplotlib.pyplot as plt
import re

cmd = "./main -m models/wz-30B-uncensored.ggmlv3.q4_0.bin --no-penalize-nl --color 2000 --temp 0.85 --repeat_penalty 1.1 -n -1 -p \"### SYSTEM:\\n Your name is Bob and you are an architect in Palo Alto. ### Instruction: Rate this memory from 1-10 and include a single sentence explanation. ### Response: \""

# Defining the range of threads to loop over
min_threads = 4
max_threads = 16
step = 2

# Defining the number of runs for each thread cmd evaluation
n_runs = 5

# Initializing the lists to store the results
threads_list = []
token_time_list = []

for threads in range(min_threads, max_threads + 1, step):

    print(f"Running with {threads} threads...")

    token_times = []

    for run in range(n_runs):

        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
        )
        output = result.stdout.decode()

        # Extracting the token time 
        pattern = r"\d+\.\d+ ms per token"
        match = re.search(pattern, output)
        
        if match:
            token_time = float(match.group().split()[0])
            token_times.append(token_time)
            print(f"\t {threads} threads | run {run+1}/{n_runs} | current token time {round(token_time, 2)} ms")
        else:
            print(f"\t {threads} threads | run {run+1}/{n_runs} | Unable to extract token time")

    if token_times:
        # Get the average token time for the current number of threads
        avg_token_time = sum(token_times) / len(token_times)
        token_time_list.append(avg_token_time)
        threads_list.append(threads)
    else:
        print(f"No valid token times found for {threads} threads.")

# Plot the result
plt.plot(threads_list, token_time_list)
plt.xlabel("Number of threads")
plt.ylabel("Token time (ms)")
plt.title("Token time vs Number of threads")
plt.show()  
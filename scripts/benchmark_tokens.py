import re
import sys
import matplotlib.pyplot as plt


def plot_simulation_output(output_file):
    with open(output_file, "r") as f:
        output = f.read()

    load_times = []
    prompt_eval_times = []
    eval_times = []

    # extract timing data for each model
    for i in range(1, 5):
        load_time_match = re.search(
            rf"Model {i}\nllama_print_timings:\s+load time =\s+(\d+\.\d+) ms", output
        )
        print(load_time_match)
        prompt_eval_time_match = re.search(
            r"llama_print_timings:\s+prompt eval time =\s+(\d+\.\d+) ms",
            output,
        )
        print(prompt_eval_time_match)
        eval_time_match = re.search(
            r"llama_print_timings:\s+eval time =\s+(\d+\.\d+) ms", output
        )
        print(eval_time_match)

        # check if all matches were found
        if not all(
            [
                load_time_match,
                prompt_eval_time_match,
                eval_time_match,
            ]
        ):
            print(f"Error: Failed to find timing data for Model {i}.")
            return

        load_time = float(load_time_match.group(1))
        print(load_time)
        prompt_eval_time = float(prompt_eval_time_match.group(1))
        print(prompt_eval_time)
        eval_time = float(eval_time_match.group(1))
        print(eval_time)

        load_times.append(load_time)
        prompt_eval_times.append(prompt_eval_time)
        eval_times.append(eval_time)

    # create grouped bar chart
    labels = ["LLaMA 13B", ""]
    metrics = [
        "Load Time",
        "Prompt Eval Time",
        "Eval Time",
    ]

    values = [load_times, prompt_eval_times, eval_times]
    fig, ax = plt.subplots()

    width = 0.2  # width of the bars
    x_pos = [i for i in range(len(labels))]

    for i, metric in enumerate(metrics):
        offset = (i - 1) * width / 2
        ax.bar(
            [j + offset for j in x_pos],
            values[i],
            width,
            label=metric,
            log=True,
        )

    ax.set_ylabel("Time (ms)")
    ax.set_title("Inference Statistics")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.legend()

    # fig, ax = plt.subplots()
    # width = 0.15  # width of the bars

    # for i, metric in enumerate(metrics):
    #     x_pos = [j + i * width for j in range(len(labels))]
    #     ax.bar(x_pos, values[i], width, label=metric)

    # ax.set_ylabel("Time (ms)")
    # ax.set_title("Inference Statistics")
    # ax.set_xticks([j + width * 2 for j in range(len(labels))])
    # ax.set_xticklabels(labels)
    # ax.legend()

    plt.show()
    plt.savefig("inference.png", dpi=300)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please provide an output file as an argument.")
    else:
        output_file = sys.argv[1]
        plot_simulation_output(output_file)

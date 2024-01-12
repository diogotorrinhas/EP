import matplotlib.pyplot as plt

iterations = []
threshold_values = []

with open('output.txt', 'r') as file:
    for line in file:
        iteration, all_outputs_within_threshold = map(int, line.split())
        iterations.append(iteration)
        threshold_values.append(all_outputs_within_threshold)

plt.figure(figsize=(10, 6))
plt.plot(iterations, threshold_values, marker='o', linestyle='-', color='b', label='all_outputs_within_threshold')
plt.title('Iterations vs. Valid output relative to desirable output')
plt.xlabel('Iteration')
plt.ylabel('Number outputs within the threshold for each iteration')
plt.grid(axis='y', linestyle='--', alpha=0.7) 
plt.legend()
plt.tight_layout()
plt.show()




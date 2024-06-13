import matplotlib.pyplot as plt


def plot_sorted_data(data):
    # Define a function to extract epoch and batch from the string
    sorted_data = sorted(data, key=lambda x: (x[1].split('_')[3], x[1].split('_')[5]))


    # Extract values and labels
    values = [item[0] for item in sorted_data]
    labels = [" ".join(item[1].split('_')[2:6]) for item in sorted_data]

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(labels, values, marker='o')
    plt.xlabel('Checkpoints')
    plt.ylabel('Score')
    plt.title('FFTranslator BLEU Scores')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(True)
    plt.show()


# Example data
data = [
    (0.26225445465249975, 'model_new1_epoch_4_batch_0'),
    (0.2695720859985482, 'model_new1_epoch_5_batch_0'),
    (0.2241620967126834, 'model_new1_epoch_2_batch_1000'),
    (0.23918840863556606, 'model_new1_epoch_3_batch_0'),
    (0.16289361991098703, 'model_new1_epoch_1_batch_1000'),
    (0.26274263383453833, 'model_new1_epoch_4_batch_1000'),
    (0.19239743725231107, 'model_new1_epoch_2_batch_0'),
    (0.25602682504108676, 'model_new1_epoch_3_batch_1000'),
    (0.0014471929687869544, 'model_new1_epoch_1_batch_0')
]


# Plot the data
if __name__ == "__main__":
    plot_sorted_data(data)

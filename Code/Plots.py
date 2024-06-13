import matplotlib.pyplot as plt
import numpy as np

def plot_sorted_data(list_data: list[list[tuple[float, str]]]):
    # Define a function to extract epoch and batch from the string
    for data in list_data:
        sorted_data = sorted(data, key=lambda x: (x[1].split('_')[3], x[1].split('_')[5]))


        # Extract values and labels
        values = [item[0] for item in sorted_data]
        labels = [" ".join(item[1].split('_')[2:6]) for item in sorted_data]

        # Create the plot
        color = np.random.rand(3, )

        # Plot the data with the random color
        plt.plot(labels, values, marker='o', color=color, label=f'Dataset {list_data.index(data) + 1}')
    plt.xlabel('Checkpoints')
    plt.ylabel('Score')
    plt.title('FFTranslator BLEU Scores')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(True)
    plt.show()


# Example data
data3 = [(0.26225445465249975, 'model_new1_epoch_4_batch_0'), (0.2695720859985482, 'model_new1_epoch_5_batch_0'), (0.2241620967126834, 'model_new1_epoch_2_batch_1000'), (0.23918840863556606, 'model_new1_epoch_3_batch_0'), (0.16289361991098703, 'model_new1_epoch_1_batch_1000'), (0.26274263383453833, 'model_new1_epoch_4_batch_1000'), (0.19239743725231107, 'model_new1_epoch_2_batch_0'), (0.25602682504108676, 'model_new1_epoch_3_batch_1000'), (0.0014471929687869544, 'model_new1_epoch_1_batch_0')]
data2 = [(0.19331260532114583, 'model_new2_epoch_2_batch_0'), (0.24107846356469423, 'model_new2_epoch_3_batch_0'), (0.2666794571560046, 'model_new2_epoch_5_batch_1000'), (0.23045242182804684, 'model_new2_epoch_2_batch_1000'), (0.26856342464360733, 'model_new2_epoch_5_batch_0'), (0.26401294709733086, 'model_new2_epoch_4_batch_1000'), (0.2591844049731434, 'model_new2_epoch_4_batch_0'), (0.2646807270564696, 'model_new2_epoch_6_batch_1000'), (0.2650639787418506, 'model_new2_epoch_6_batch_0'), (0.2565008446086964, 'model_new2_epoch_3_batch_1000'), (0.265537121262804, 'model_new2_epoch_7_batch_0')]
data1 = [(0.2579527764309045, 'model_new3_epoch_3_batch_1000'), (0.2674273519935779, 'model_new3_epoch_6_batch_1000'), (0.24018211573207304, 'model_new3_epoch_3_batch_0'), (0.1976114867265978, 'model_new3_epoch_2_batch_0'), (0.26594114515167, 'model_new3_epoch_4_batch_1000'), (0.2289571589746973, 'model_new3_epoch_2_batch_1000'), (0.26604378286514746, 'model_new3_epoch_6_batch_0'), (0.268090364239886, 'model_new3_epoch_5_batch_1000'), (0.2644419135281811, 'model_new3_epoch_4_batch_0'), (0.27205998360116307, 'model_new3_epoch_5_batch_0')]
# Plot the data
if __name__ == "__main__":
    plot_sorted_data([data3,data2,data1])

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


def plot_epoch_batch_performance(data_sets):
    plt.figure(figsize=(14, 10))
    colors = ['b', 'g', 'r', 'c']  # Farben für verschiedene Datensätze
    markers = ['o', 's', '^', 'D']  # Marker für verschiedene Datensätze

    for i, data in enumerate(data_sets):
        epochs = []
        batches = []
        performances = []
        for item in data:
            performance, decoder, _ = item
            epoch = int(decoder.split('_')[2])
            batch = int(decoder.split('_')[4])
            epochs.append(epoch)
            batches.append(batch)
            performances.append(performance)

        # Erstelle einen kombinierten x-Wert aus Epoch und Batch
        x_values = [e + b / 100 for e, b in zip(epochs, batches)]

        # Sortiere die Datenpunkte nach x_values (wichtig für Datensatz 1)
        sorted_data = sorted(zip(x_values, performances, epochs, batches), key=lambda x: x[0], reverse=(i == 0))
        x_values, performances, epochs, batches = zip(*sorted_data)

        plt.scatter(x_values, performances, color=colors[i], marker=markers[i], s=100,
                    label=f'Random Seed {i + 1}')

        # Verbinde die Punkte mit Linien
        plt.plot(x_values, performances, color=colors[i], linestyle='-', linewidth=1, alpha=0.5)

        # Füge Beschriftungen für Epoch und Batch hinzu
        for x, y, e, b in zip(x_values, performances, epochs, batches):
            plt.annotate(f'E{e}B{b}', (x, y), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=8)

    plt.xlabel('Epoch.Batch', fontsize=12)
    plt.ylabel('Performance', fontsize=12)
    plt.title('Performance über Epochen und Batches', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


# Die Datensätze (unverändert)
data_sets = [
    [(0.36108204470992, 'decoder_epoch_10_batch_6', 'encoder_epoch_10_batch_6'),
     (0.3685984039566479, 'decoder_epoch_8_batch_6', 'encoder_epoch_8_batch_6'),
     (0.35941475757483055, 'decoder_epoch_9_batch_6', 'encoder_epoch_9_batch_6'),
     (0.34828202063157915, 'decoder_epoch_11_batch_6', 'encoder_epoch_11_batch_6')],

    [(0.3718710536757511, 'decoder_epoch_6_batch_6', 'encoder_epoch_6_batch_6'),
     (0.3572673903593455, 'decoder_epoch_7_batch_6', 'encoder_epoch_7_batch_6'),
     (0.3577354098717659, 'decoder_epoch_8_batch_6', 'encoder_epoch_8_batch_6')],

    [(0.36510628464239014, 'decoder_epoch_3_batch_6', 'encoder_epoch_3_batch_6'),
     (0.36728521196063973, 'decoder_epoch_4_batch_6', 'encoder_epoch_4_batch_6'),
     (0.3657297638743422, 'decoder_epoch_5_batch_6', 'encoder_epoch_5_batch_6'),
     (0.36841043042526256, 'decoder_epoch_6_batch_6', 'encoder_epoch_6_batch_6')],

    [(0.3647545739762098, 'decoder_epoch_5_batch_6', 'encoder_epoch_5_batch_6'),
     (0.3730983841346014, 'decoder_epoch_5_batch_36', 'encoder_epoch_5_batch_36'),
     (0.36388196041296894, 'decoder_epoch_5_batch_66', 'encoder_epoch_5_batch_66'),
     (0.3548500721691888, 'decoder_epoch_6_batch_6', 'encoder_epoch_6_batch_6'),
     (0.3654212396135533, 'decoder_epoch_6_batch_36', 'encoder_epoch_6_batch_36'),
     (0.36956818224831445, 'decoder_epoch_6_batch_66', 'encoder_epoch_6_batch_66'),
     (0.3646471355569043, 'decoder_epoch_7_batch_6', 'encoder_epoch_7_batch_6'),
     (0.3685623397345794, 'decoder_epoch_7_batch_36', 'encoder_epoch_7_batch_36'),
     (0.3600306615511063, 'decoder_epoch_7_batch_66', 'encoder_epoch_7_batch_66')]
]

plot_epoch_batch_performance(data_sets)

def plot_attention_comparison(normal_attention, masked_attention):
    plt.figure(figsize=(14, 10))

    def process_data(data, color, marker, label):
        epochs = []
        batches = []
        performances = []
        for item in data:
            performance, decoder, _ = item
            epoch = int(decoder.split('_')[2])
            batch = int(decoder.split('_')[4])
            epochs.append(epoch)
            batches.append(batch)
            performances.append(performance)

        x_values = [e + b / 100 for e, b in zip(epochs, batches)]

        plt.scatter(x_values, performances, color=color, marker=marker, s=100, label=label)
        plt.plot(x_values, performances, color=color, linestyle='-', linewidth=1, alpha=0.5)

        for x, y, e, b in zip(x_values, performances, epochs, batches):
            plt.annotate(f'E{e}B{b}', (x, y), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=8, color=color)

    process_data(normal_attention, 'blue', 'o', 'Normale Attention')
    process_data(masked_attention, 'red', 's', 'Masked Attention')

    plt.xlabel('Epoch.Batch', fontsize=12)
    plt.ylabel('Performance', fontsize=12)
    plt.title('Vergleich: Normale Attention vs. Masked Attention', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


# Datensätze
normal_attention = [
    (0.3647545739762098, 'decoder_epoch_5_batch_6', 'encoder_epoch_5_batch_6'),
    (0.3730983841346014, 'decoder_epoch_5_batch_36', 'encoder_epoch_5_batch_36'),
    (0.36388196041296894, 'decoder_epoch_5_batch_66', 'encoder_epoch_5_batch_66'),
    (0.3548500721691888, 'decoder_epoch_6_batch_6', 'encoder_epoch_6_batch_6'),
    (0.3654212396135533, 'decoder_epoch_6_batch_36', 'encoder_epoch_6_batch_36'),
    (0.36956818224831445, 'decoder_epoch_6_batch_66', 'encoder_epoch_6_batch_66'),
    (0.3646471355569043, 'decoder_epoch_7_batch_6', 'encoder_epoch_7_batch_6'),
    (0.3685623397345794, 'decoder_epoch_7_batch_36', 'encoder_epoch_7_batch_36'),
    (0.3600306615511063, 'decoder_epoch_7_batch_66', 'encoder_epoch_7_batch_66')
]

masked_attention = [
    (0.3339468343278821, 'decoder_epoch_3_batch_6', 'encoder_epoch_3_batch_6'),
    (0.35680812289799835, 'decoder_epoch_3_batch_36', 'encoder_epoch_3_batch_36'),
    (0.3608441020609596, 'decoder_epoch_3_batch_66', 'encoder_epoch_3_batch_66'),
    (0.3630521509716972, 'decoder_epoch_4_batch_6', 'encoder_epoch_4_batch_6'),
    (0.3723649149174263, 'decoder_epoch_4_batch_36', 'encoder_epoch_4_batch_36'),
    (0.3780294690748949, 'decoder_epoch_4_batch_66', 'encoder_epoch_4_batch_66'),
    (0.37071188791526916, 'decoder_epoch_5_batch_6', 'encoder_epoch_5_batch_6'),
    (0.37703093448586855, 'decoder_epoch_5_batch_36', 'encoder_epoch_5_batch_36'),
    (0.3688320503224798, 'decoder_epoch_5_batch_66', 'encoder_epoch_5_batch_66'),
    (0.3633446963261525, 'decoder_epoch_6_batch_6', 'encoder_epoch_6_batch_6'),
    (0.3748793157821535, 'decoder_epoch_6_batch_36', 'encoder_epoch_6_batch_36'),
    (0.35909413156348396, 'decoder_epoch_6_batch_66', 'encoder_epoch_6_batch_66')
]

plot_attention_comparison(normal_attention, masked_attention)

def plot_dropout_comparison(without_dropout, with_dropout):
    plt.figure(figsize=(14, 10))

    def process_data(data, color, marker, label):
        epochs = []
        batches = []
        performances = []
        for item in data:
            performance, decoder, _ = item
            epoch = int(decoder.split('_')[2])
            batch = int(decoder.split('_')[4])
            epochs.append(epoch)
            batches.append(batch)
            performances.append(performance)

        x_values = [e + b / 100 for e, b in zip(epochs, batches)]

        plt.scatter(x_values, performances, color=color, marker=marker, s=100, label=label)
        plt.plot(x_values, performances, color=color, linestyle='-', linewidth=1, alpha=0.5)

        for x, y, e, b in zip(x_values, performances, epochs, batches):
            plt.annotate(f'E{e}B{b}', (x, y), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=8, color=color)

    process_data(without_dropout, 'blue', 'o', 'Ohne Dropout')
    process_data(with_dropout, 'red', 's', 'Mit Dropout')

    plt.xlabel('Epoch.Batch', fontsize=12)
    plt.ylabel('Performance', fontsize=12)
    plt.title('Vergleich: Ohne Dropout vs. Mit Dropout', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


# Datensätze
without_dropout = [
    (0.3339468343278821, 'decoder_epoch_3_batch_6', 'encoder_epoch_3_batch_6'),
    (0.35680812289799835, 'decoder_epoch_3_batch_36', 'encoder_epoch_3_batch_36'),
    (0.3608441020609596, 'decoder_epoch_3_batch_66', 'encoder_epoch_3_batch_66'),
    (0.3630521509716972, 'decoder_epoch_4_batch_6', 'encoder_epoch_4_batch_6'),
    (0.3723649149174263, 'decoder_epoch_4_batch_36', 'encoder_epoch_4_batch_36'),
    (0.3780294690748949, 'decoder_epoch_4_batch_66', 'encoder_epoch_4_batch_66'),
    (0.37071188791526916, 'decoder_epoch_5_batch_6', 'encoder_epoch_5_batch_6'),
    (0.37703093448586855, 'decoder_epoch_5_batch_36', 'encoder_epoch_5_batch_36'),
    (0.3688320503224798, 'decoder_epoch_5_batch_66', 'encoder_epoch_5_batch_66'),
    (0.3633446963261525, 'decoder_epoch_6_batch_6', 'encoder_epoch_6_batch_6'),
    (0.3748793157821535, 'decoder_epoch_6_batch_36', 'encoder_epoch_6_batch_36'),
    (0.35909413156348396, 'decoder_epoch_6_batch_66', 'encoder_epoch_6_batch_66')
]

with_dropout = [
    (0.3640082327125792, 'decoder_epoch_4_batch_6', 'encoder_epoch_4_batch_6'),
    (0.37278117426464946, 'decoder_epoch_4_batch_36', 'encoder_epoch_4_batch_36'),
    (0.3698925365854511, 'decoder_epoch_4_batch_66', 'encoder_epoch_4_batch_66'),
    (0.37837230672271893, 'decoder_epoch_5_batch_6', 'encoder_epoch_5_batch_6'),
    (0.36603977134730137, 'decoder_epoch_5_batch_36', 'encoder_epoch_5_batch_36'),
    (0.3566606006406616, 'decoder_epoch_5_batch_66', 'encoder_epoch_5_batch_66'),
    (0.3575313840154873, 'decoder_epoch_6_batch_6', 'encoder_epoch_6_batch_6'),
    (0.3694922439301947, 'decoder_epoch_6_batch_36', 'encoder_epoch_6_batch_36'),
    (0.35582696723379, 'decoder_epoch_6_batch_66', 'encoder_epoch_6_batch_66')
]

plot_dropout_comparison(without_dropout, with_dropout)

import matplotlib.pyplot as plt

def plot_attention_comparison(normal_attention, dot_attention):
    plt.figure(figsize=(14, 10))

    def process_data(data, color, marker, label):
        epochs = []
        batches = []
        performances = []
        for item in data:
            performance, decoder, _ = item
            epoch = int(decoder.split('_')[2])
            batch = int(decoder.split('_')[4])
            epochs.append(epoch)
            batches.append(batch)
            performances.append(performance)

        x_values = [e + b / 100 for e, b in zip(epochs, batches)]

        plt.scatter(x_values, performances, color=color, marker=marker, s=100, label=label)
        plt.plot(x_values, performances, color=color, linestyle='-', linewidth=1, alpha=0.5)

        for x, y, e, b in zip(x_values, performances, epochs, batches):
            plt.annotate(f'E{e}B{b}', (x, y), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=8, color=color)

    process_data(normal_attention, 'blue', 'o', 'Normale Attention')
    process_data(dot_attention, 'red', 's', 'Dot-Attention')

    plt.xlabel('Epoch.Batch', fontsize=12)
    plt.ylabel('Performance', fontsize=12)
    plt.title('Vergleich: Normale Attention vs. Dot-Attention', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

# Datensätze
normal_attention = [
    (0.3640082327125792, 'decoder_epoch_4_batch_6', 'encoder_epoch_4_batch_6'),
    (0.37278117426464946, 'decoder_epoch_4_batch_36', 'encoder_epoch_4_batch_36'),
    (0.3698925365854511, 'decoder_epoch_4_batch_66', 'encoder_epoch_4_batch_66'),
    (0.37837230672271893, 'decoder_epoch_5_batch_6', 'encoder_epoch_5_batch_6'),
    (0.36603977134730137, 'decoder_epoch_5_batch_36', 'encoder_epoch_5_batch_36'),
    (0.3566606006406616, 'decoder_epoch_5_batch_66', 'encoder_epoch_5_batch_66'),
    (0.3575313840154873, 'decoder_epoch_6_batch_6', 'encoder_epoch_6_batch_6'),
    (0.3694922439301947, 'decoder_epoch_6_batch_36', 'encoder_epoch_6_batch_36'),
    (0.35582696723379, 'decoder_epoch_6_batch_66', 'encoder_epoch_6_batch_66')
]

dot_attention = [
    (0.36232038287146334, 'decoder_epoch_5_batch_6', 'encoder_epoch_5_batch_6'),
    (0.3649318934245246, 'decoder_epoch_5_batch_56', 'encoder_epoch_5_batch_56'),
    (0.3592337924811395, 'decoder_epoch_6_batch_6', 'encoder_epoch_6_batch_6'),
    (0.35492988907341455, 'decoder_epoch_6_batch_56', 'encoder_epoch_6_batch_56'),
    (0.36063442837030607, 'decoder_epoch_7_batch_6', 'encoder_epoch_7_batch_6'),
    (0.37142743734103506, 'decoder_epoch_7_batch_56', 'encoder_epoch_7_batch_56'),
    (0.3568595819565794, 'decoder_epoch_8_batch_6', 'encoder_epoch_8_batch_6'),
    (0.34876340704794956, 'decoder_epoch_8_batch_56', 'encoder_epoch_8_batch_56'),
    (0.36358432849495653, 'decoder_epoch_9_batch_6', 'encoder_epoch_9_batch_6'),
    (0.3612404887812549, 'decoder_epoch_9_batch_56', 'encoder_epoch_9_batch_56')
]

plot_attention_comparison(normal_attention, dot_attention)

import matplotlib.pyplot as plt

def plot_lstm_hidden_layer_comparison(lstm_256, lstm_1024):
    plt.figure(figsize=(14, 10))

    def process_data(data, color, marker, label):
        epochs = []
        batches = []
        performances = []
        for item in data:
            performance, decoder, _ = item
            epoch = int(decoder.split('_')[2])
            batch = int(decoder.split('_')[4])
            epochs.append(epoch)
            batches.append(batch)
            performances.append(performance)

        x_values = [e + b / 100 for e, b in zip(epochs, batches)]

        plt.scatter(x_values, performances, color=color, marker=marker, s=100, label=label)
        plt.plot(x_values, performances, color=color, linestyle='-', linewidth=1, alpha=0.5)

        for x, y, e, b in zip(x_values, performances, epochs, batches):
            plt.annotate(f'E{e}B{b}', (x, y), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=8, color=color)

    process_data(lstm_256, 'blue', 'o', 'LSTM Hidden Layer 256')
    process_data(lstm_1024, 'red', 's', 'LSTM Hidden Layer 1024')

    plt.xlabel('Epoch.Batch', fontsize=12)
    plt.ylabel('Performance', fontsize=12)
    plt.title('Vergleich: LSTM Hidden Layer 256 vs. 1024', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

# Datensätze
lstm_256 = [
    (0.3640082327125792, 'decoder_epoch_4_batch_6', 'encoder_epoch_4_batch_6'),
    (0.37278117426464946, 'decoder_epoch_4_batch_36', 'encoder_epoch_4_batch_36'),
    (0.3698925365854511, 'decoder_epoch_4_batch_66', 'encoder_epoch_4_batch_66'),
    (0.37837230672271893, 'decoder_epoch_5_batch_6', 'encoder_epoch_5_batch_6'),
    (0.36603977134730137, 'decoder_epoch_5_batch_36', 'encoder_epoch_5_batch_36'),
    (0.3566606006406616, 'decoder_epoch_5_batch_66', 'encoder_epoch_5_batch_66'),
    (0.3575313840154873, 'decoder_epoch_6_batch_6', 'encoder_epoch_6_batch_6'),
    (0.3694922439301947, 'decoder_epoch_6_batch_36', 'encoder_epoch_6_batch_36'),
    (0.35582696723379, 'decoder_epoch_6_batch_66', 'encoder_epoch_6_batch_66')
]

lstm_1024 = [
    (0.36901178245906924, 'decoder_epoch_4_batch_6', 'encoder_epoch_4_batch_6'),
    (0.35531278178684966, 'decoder_epoch_4_batch_36', 'encoder_epoch_4_batch_36'),
    (0.3574613067155715, 'decoder_epoch_4_batch_66', 'encoder_epoch_4_batch_66'),
    (0.3705677147727105, 'decoder_epoch_5_batch_6', 'encoder_epoch_5_batch_6'),
    (0.3629585260910479, 'decoder_epoch_5_batch_36', 'encoder_epoch_5_batch_36'),
    (0.3686326744017133, 'decoder_epoch_5_batch_66', 'encoder_epoch_5_batch_66'),
    (0.35576336814137544, 'decoder_epoch_6_batch_6', 'encoder_epoch_6_batch_6'),
    (0.36078967751331736, 'decoder_epoch_6_batch_36', 'encoder_epoch_6_batch_36'),
    (0.36356167027357894, 'decoder_epoch_6_batch_66', 'encoder_epoch_6_batch_66')
]

plot_lstm_hidden_layer_comparison(lstm_256, lstm_1024)


"""
# Example data
data3 = [(0.26225445465249975, 'model_new1_epoch_4_batch_0'), (0.2695720859985482, 'model_new1_epoch_5_batch_0'), (0.2241620967126834, 'model_new1_epoch_2_batch_1000'), (0.23918840863556606, 'model_new1_epoch_3_batch_0'), (0.16289361991098703, 'model_new1_epoch_1_batch_1000'), (0.26274263383453833, 'model_new1_epoch_4_batch_1000'), (0.19239743725231107, 'model_new1_epoch_2_batch_0'), (0.25602682504108676, 'model_new1_epoch_3_batch_1000'), (0.0014471929687869544, 'model_new1_epoch_1_batch_0')]
data2 = [(0.19331260532114583, 'model_new2_epoch_2_batch_0'), (0.24107846356469423, 'model_new2_epoch_3_batch_0'), (0.2666794571560046, 'model_new2_epoch_5_batch_1000'), (0.23045242182804684, 'model_new2_epoch_2_batch_1000'), (0.26856342464360733, 'model_new2_epoch_5_batch_0'), (0.26401294709733086, 'model_new2_epoch_4_batch_1000'), (0.2591844049731434, 'model_new2_epoch_4_batch_0'), (0.2646807270564696, 'model_new2_epoch_6_batch_1000'), (0.2650639787418506, 'model_new2_epoch_6_batch_0'), (0.2565008446086964, 'model_new2_epoch_3_batch_1000'), (0.265537121262804, 'model_new2_epoch_7_batch_0')]
data1 = [(0.2579527764309045, 'model_new3_epoch_3_batch_1000'), (0.2674273519935779, 'model_new3_epoch_6_batch_1000'), (0.24018211573207304, 'model_new3_epoch_3_batch_0'), (0.1976114867265978, 'model_new3_epoch_2_batch_0'), (0.26594114515167, 'model_new3_epoch_4_batch_1000'), (0.2289571589746973, 'model_new3_epoch_2_batch_1000'), (0.26604378286514746, 'model_new3_epoch_6_batch_0'), (0.268090364239886, 'model_new3_epoch_5_batch_1000'), (0.2644419135281811, 'model_new3_epoch_4_batch_0'), (0.27205998360116307, 'model_new3_epoch_5_batch_0')]
# Plot the data
if __name__ == "__main__":
    plot_sorted_data([data3,data2,data1])
"""
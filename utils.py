import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def get_vocab_list(vocab_path):
    with open(vocab_path, 'r') as vocab_file:
        lines = vocab_file.readlines()
        lines = [line.strip() for line in lines]
    return np.array(lines)

def calc_batch_accuracy(out, batch_targets, vocab_path='vocab.txt', verbose=False):
    vocab = get_vocab_list(vocab_path)
    acc_sum = 0
    batch_size = out.shape[0]
    for seq in range(batch_size):
        pred_indices = torch.argmax(out[seq], axis=1).cpu().numpy()
        predictions = vocab[pred_indices]
        original_sentence = vocab[batch_targets.cpu().numpy()[seq]]
        acc = accuracy_score(original_sentence, predictions) # don't calc accuracy on <sos> and <eos>
        acc_sum += acc
        if verbose:
            print(f'Seq #{seq}')
            print(f'Original sentence: {" ".join(original_sentence)}')
            print(f'Predicted sentence: {" ".join(predictions)}')
            print(f'Accuracy: {acc}')
    batch_acc = acc_sum/batch_size
    return batch_acc

def plot_metric(values):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(values, '-', label='accuracy')
    ax.tick_params(axis='x', colors='w')
    ax.tick_params(axis='y', colors='w')
    plt.legend()
    plt.show()
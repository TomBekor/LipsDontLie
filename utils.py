import numpy as np
from sqlalchemy import true
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
        eos_idx = np.where(original_sentence == '<eos>')[0][0] # first <eos> sentence
        original_sentence = original_sentence[:eos_idx+1]
        predictions = predictions[:eos_idx+1]
        if len(predictions) <= 1:
            acc = 0
        else:
            acc = accuracy_score(original_sentence[:-1], predictions[:-1]) # don't calc accuracy on <sos> and <eos>
        acc_sum += acc
        if verbose:
            print(f'Seq #{seq}')
            print(f'Original sentence: {" ".join(original_sentence)}')
            print(f'Predicted sentence: {" ".join(predictions)}')
            print(f'Accuracy: {acc}')
    batch_acc = acc_sum/batch_size
    return batch_acc

def predicted_sentence(out, vocab_path='vocab.txt'):
    vocab = get_vocab_list(vocab_path)
    pred_indices = torch.argmax(out[0], axis=1).cpu().numpy()
    predictions = vocab[pred_indices]
    predictions = list(predictions)
    predictions = predictions[:predictions.index('<eos>')+1]
    return predictions
    

def plot_metric(values, label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(values, '-', label=label)
    ax.tick_params(axis='x', colors='w')
    ax.tick_params(axis='y', colors='w')
    plt.legend()
    plt.show()

def float_array_form_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        array = lines[0][1:-1]
        array = array.split(',')
        array = [float(v.strip()) for v in array]
    return array

def write_metric(values, output_path):
    with open(output_path, 'w+') as output_file:
        output_file.write(str(values))
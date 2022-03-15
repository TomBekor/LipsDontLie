import torch
import numpy as np
from dataloader import DataLoader, Tokenizer
from model import LandmarksNN, Transformer
from utils import get_vocab_list
import config as cfg
from sklearn.metrics import accuracy_score
from tqdm import tqdm

torch.manual_seed(0)

'''
Loads pretrained models and prints accuracy on the test-set.
'''

landmarksNN_model_path = cfg.LANDMARKSNN_LOAD
transformer_model_path = cfg.TRANSFORMER_LOAD

print()
print(f'Pretrained models:')
print(landmarksNN_model_path)
print(transformer_model_path)
print()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the vocabulary
vocab_file = open(cfg.VOCABULARY_FILE, 'r')
idx2word = vocab_file.read().splitlines()
vocab_size = len(idx2word)
word2idx = {word:idx for idx, word in enumerate(idx2word)}
vocab_file.close()
vocab = get_vocab_list(cfg.VOCABULARY_FILE)
# Intialize DataLoader and Tokenizer
tokenizer = Tokenizer(word2idx)

test_loader = DataLoader(landmarks_path=cfg.LANDMARKS_DIR, annotations_path=cfg.ANNOTATIONS_DIR,
 vocab=vocab, mode='test', batch_size=100, shuffle=True)

landmarks_model = LandmarksNN()
landmarks_model.load_state_dict(torch.load(landmarksNN_model_path))
landmarks_model.to(device)

transformer = Transformer(vocab_size)
transformer.load_state_dict(torch.load(transformer_model_path))
transformer.to(device)

def pytorch_model_num_of_params(torch_model):
    return sum(p.numel() for p in torch_model.parameters() if p.requires_grad)

lnmarks_params = pytorch_model_num_of_params(landmarks_model)
transformer_params = pytorch_model_num_of_params(transformer)
print(f'landmarks_model number of parameters: {lnmarks_params:,}')
print(f'transformer_model number of parameters: {transformer_params:,}')
print(f'model total number of parameters: {lnmarks_params + transformer_params:,}')
print()

landmarks_model.eval()
transformer.eval()
with torch.no_grad():
    test_accs = []
    for samples, targets in tqdm(test_loader):
        # Start with <sos> token
        preds = np.array([['<sos>']]*len(samples))
        # Forward
        for pred_idx in range(7):
            tokenizer_res = tokenizer.tokenize(samples, preds)
            batch_inputs, batch_targets,\
            batch_in_pad_masks, batch_tgt_pad_masks = tokenizer_res

            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_in_pad_masks = batch_in_pad_masks.to(device)
            batch_tgt_pad_masks = batch_tgt_pad_masks.to(device)

            landmarks_out = landmarks_model.forward(batch_inputs)
            landmarks_out = landmarks_out.to(device)

            out = transformer.forward(landmarks_out, batch_targets, batch_in_pad_masks, batch_tgt_pad_masks)
            preds_tokens = torch.argmax(out[:,pred_idx,:], dim=1).cpu()
            preds_tokens = np.array(preds_tokens)

            preds_words = np.array(list(map(lambda x: idx2word[x], preds_tokens)))
            preds = np.concatenate([preds, preds_words.reshape(-1,1)], axis=1)

        # Accuracy evaluation
        preds = preds[:,1:-1]
        acc = accuracy_score(np.array(targets).flatten(), preds.flatten())
        test_accs.append(acc)
    
    print()
    print(f'Test accuracy: {round(100*np.mean(test_accs),2)}%')
    print()
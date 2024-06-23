import sys

sys.path.append('space-model')

import math
import json
from collections import Counter
import random
import os

import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from tqdm import tqdm

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

from datasets import load_dataset, Dataset, DatasetDict

from pynvml import *
from numba import cuda

from space_model.model import *
from space_model.loss import *

from logger import get_logger, log_continue
from utils import free_resources_deco


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def on_gpu(f):
    def wrapper(*args):
        if torch.cuda.is_available():
            return f(*args)
        else:
            log.warn('cuda unavailable')

    return wrapper


@on_gpu
def print_gpu_utilization(dev_id):
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(dev_id)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")
    except Exception as e:
        print(e)


@on_gpu
def free_gpu_cache(dev_id=0):
    print("Initial GPU Usage")
    print_gpu_utilization(dev_id)

    torch.cuda.empty_cache()

    print("GPU Usage after emptying the cache")
    print_gpu_utilization(dev_id)


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def prepare_dataset(model_name, max_seq_len, device, seed):
    """
    Loads dataset from csv files, encodes labels, tokenizes text and returns tokenized dataset
    !!! This function is dataset specific !!!
    !!! It is supposed to be rewritten for every dataset provided to the model !!!
    :param model_name:
    :param max_seq_len:
    :param device:
    :param seed:
    :return:
    """
    # Load the IMDb dataset
    dataset = load_dataset("imdb")

    # Split the training set into training (80%) and validation (20%) sets
    train_testvalid = dataset['train'].train_test_split(test_size=0.2, seed=42)

    # Assign datasets
    train_dataset = train_testvalid['train']
    val_dataset = train_testvalid['test']
    test_dataset = dataset['test']

    dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset,
        'val': val_dataset
    })

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=max_seq_len,
                            return_tensors='pt'),
        batched=True)
    tokenized_dataset.set_format('torch', device=device)

    return tokenized_dataset


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@log_continue
def plot_results(log, history, plot_name, do_val=True):
    # log is passed here to comply with log_continue decorator

    fig, ax = plt.subplots(figsize=(8, 8))

    x = list(range(0, len(history['train_losses'])))

    # loss

    ax.plot(x, history['train_losses'], label='train_loss')

    if do_val:
        ax.plot(x, history['val_losses'], label='val_loss')

    plt.title('Train / Validation Loss')
    plt.legend(loc='upper right')

    # check if directory exists
    if not os.path.exists(f'plots/{plot_name}'):
        os.makedirs(f'plots/{plot_name}', exist_ok=True)

    fig.savefig(f'plots/{plot_name}/loss.png')

    # accuracy

    if 'train_acc' in history:
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot(x, history['train_acc'], label='train_acc')

        if do_val:
            ax.plot(x, history['val_acc'], label='val_acc')

        plt.title('Train / Validation Accuracy')
        plt.legend(loc='upper right')

        fig.savefig(f'plots/{plot_name}/acc.png')

    # f1-score

    if 'train_f1' in history:
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot(x, history['train_f1'], label='train_f1')

        if do_val:
            ax.plot(x, history['val_f1'], label='val_f1')

        plt.title('Train / Validation F1')
        plt.legend(loc='upper right')

        fig.savefig(f'plots/{plot_name}/f1.png')

    # cs accuracy

    if 'cs_train_acc' in history and history['cs_train_acc']:
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot(x, history['cs_train_acc'], label='cs_train_acc')

        if do_val:
            ax.plot(x, history['cs_val_acc'], label='cs_val_acc')

        plt.title('Train / Validation CS Accuracy')
        plt.legend(loc='upper right')

        fig.savefig(f'plots/{plot_name}/cs_acc.png')

    # cs f1-score

    if 'cs_train_f1' in history and history['cs_train_f1']:
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot(x, history['cs_train_f1'], label='cs_train_f1')

        if do_val:
            ax.plot(x, history['cs_val_f1'], label='cs_val_f1')

        plt.title('Train / Validation CS F1')
        plt.legend(loc='upper right')

        fig.savefig(f'plots/{plot_name}/cs_f1.png')

    # precision

    if 'train_precision' in history:
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot(x, history['train_precision'], label='train_precision')

        if do_val:
            ax.plot(x, history['val_precision'], label='val_precision')

        plt.title('Train / Validation Precision')
        plt.legend(loc='upper right')

        fig.savefig(f'plots/{plot_name}/precision.png')

    # recall

    if 'train_recall' in history:
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot(x, history['train_recall'], label='train_recall')

        if do_val:
            ax.plot(x, history['val_recall'], label='val_recall')

        plt.title('Train / Validation Recall')
        plt.legend(loc='upper right')

        fig.savefig(f'plots/{plot_name}/recall.png')


def eval(f):
    def wrapper(model, *args, **kwargs):
        model.eval()
        return f(model, *args, **kwargs)

    return wrapper


def train(f):
    def wrapper(model, *args, **kwargs):
        model.train()
        return f(model, *args, **kwargs)

    return wrapper


def concept_space_to_preds(concept_spaces):
    tensor_concept_spaces = torch.cat([cs.unsqueeze(0) for cs in concept_spaces], dim=0)
    concept_space_dist = tensor_concept_spaces.permute(1, 0, 2, 3).mean(dim=(2, 3))  # (B, n)
    return torch.argmax(concept_space_dist, dim=1).detach().cpu().tolist()

def get_preds_from_logits(outputs):
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).cpu()  # (B)
    return outputs.loss, pred, logits

@train
def train_epoch(model, train_dataloader, optimizer, scheduler, config):
    train_loss = 0.0
    train_preds = []
    cs_train_preds = []
    train_labels = []

    for step, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
        ids = batch['input_ids'].to(model.device, dtype=torch.long)
        mask = batch['attention_mask'].to(model.device, dtype=torch.long)
        targets = batch['label'].to(model.device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels=targets)  # (B, Seq_Len, 2)

        loss, pred, logits = config['preds_from_logits_func'](outputs)

        train_preds += pred.detach().tolist()
        train_labels += targets.cpu().tolist()

        ### Distance Based Classification
        # out.concept_spaces (n, B, seq_len, n_latent)
        if hasattr(outputs, 'concept_spaces'):
            preds = concept_space_to_preds(outputs.concept_spaces)

            # multi-label classification
            if len(targets.shape) > 1:
                # turn preds into one-hot
                preds = F.one_hot(torch.tensor(preds), len(outputs.concept_spaces))
            cs_train_preds += preds
        ### END

        if (step + 1) % config['gradient_accumulation_steps'] == 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        batch_loss = loss.item()
        if config['gradient_accumulation_steps'] > 1:
            batch_loss = batch_loss / config['gradient_accumulation_steps']

        train_loss += batch_loss
    return train_loss, train_preds, train_labels, cs_train_preds


@eval
def eval_epoch(model, val_dataloader, config):
    val_loss = 0.0
    val_preds = []
    cs_val_preds = []
    val_labels = []

    with torch.no_grad():

        for step, batch in enumerate(tqdm(val_dataloader, total=len(val_dataloader))):
            ids = batch['input_ids'].to(model.device, dtype=torch.long)
            mask = batch['attention_mask'].to(model.device, dtype=torch.long)
            targets = batch['label'].to(model.device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)

            loss, pred, logits = config['preds_from_logits_func'](outputs)  # [loss, pred, logits]

            val_preds += pred.detach().tolist()
            val_labels += targets.cpu().tolist()

            ### Distance Based Classification
            # out.concept_spaces (n, B, seq_len, n_latent)
            if hasattr(outputs, 'concept_spaces'):
                preds = concept_space_to_preds(outputs.concept_spaces)

                # multi-label classification
                if len(targets.shape) > 1:
                    # turn preds into one-hot
                    preds = F.one_hot(torch.tensor(preds), len(outputs.concept_spaces))
                cs_val_preds += preds
            ### END

            val_loss += loss.item()
    return val_loss, val_preds, val_labels, cs_val_preds

@free_resources_deco
def training(model, train_data, val_data, log, config):
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    num_train_steps = int(len(train_data) / config['batch_size'] * config['num_epochs'])
    steps_per_epoch = len(train_data) / config['batch_size']

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['num_warmup_steps'],
        num_training_steps=num_train_steps,
    )

    log.debug(f'Train steps: {num_train_steps}', terminal=config['log_terminal'])
    log.debug(f'Steps per epoch: {steps_per_epoch}', terminal=config['log_terminal'])

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=config['batch_size'])

    history = {
        'train_losses': [],
        'val_losses': [],
        'train_acc': [],
        'val_acc': [],
        'cs_train_acc': [],
        'cs_val_acc': [],
        'train_f1': [],
        'val_f1': [],
        'cs_train_f1': [],
        'cs_val_f1': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': [],
    }

    for epoch_num in range(config['num_epochs']):
        log.info(f'Epoch: {epoch_num + 1}', terminal=config['log_terminal'])

        # train stage
        train_loss, train_preds, train_labels, cs_train_preds = train_epoch(model, train_dataloader, optimizer,
                                                                            scheduler, config)

        # eval stage
        val_loss, val_preds, val_labels, cs_val_preds = eval_epoch(model, val_dataloader, config)

        # metrics
        if len(cs_train_preds) != 0:
            cs_train_acc = accuracy_score(train_labels, cs_train_preds)
            cs_val_acc = accuracy_score(val_labels, cs_val_preds)

            cs_train_f1 = f1_score(train_labels, cs_train_preds, average='macro')
            cs_val_f1 = f1_score(val_labels, cs_val_preds, average='macro')

            history['cs_train_acc'].append(cs_train_acc)
            history['cs_val_acc'].append(cs_val_acc)

            history['cs_train_f1'].append(cs_train_acc)
            history['cs_val_f1'].append(cs_val_acc)

        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        train_precision = precision_score(train_labels, train_preds, average='macro')
        val_precision = precision_score(val_labels, val_preds, average='macro')
        train_recall = recall_score(train_labels, train_preds, average='macro')
        val_recall = recall_score(val_labels, val_preds, average='macro')

        history['train_losses'].append(train_loss / len(train_dataloader))
        history['val_losses'].append(val_loss / len(val_dataloader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)

        log.info(terminal=config['log_terminal'])
        log.info(f'Train loss: {train_loss / len(train_dataloader)} | Val loss: {val_loss / len(val_dataloader)}',
                 terminal=config['log_terminal'])
        log.info(f'Train acc: {train_acc} | Val acc: {val_acc}', terminal=config['log_terminal'])

        if len(cs_train_preds) != 0:
            log.info(f'CS Train acc: {cs_train_acc} | CS Val acc: {cs_val_acc}', terminal=config['log_terminal'])

        log.info(f'Train f1: {train_f1} | Val f1: {val_f1}', terminal=config['log_terminal'])

        if len(cs_train_preds) != 0:
            log.info(f'CS Train f1: {cs_train_f1} | CS Val f1: {cs_val_f1}', terminal=config['log_terminal'])

        log.info(f'Train precision: {train_precision} | Val precision: {val_precision}', terminal=config['log_terminal'])
        log.info(f'Train recall: {train_recall} | Val recall: {val_recall}', terminal=config['log_terminal'])
    return history


@log_continue
def train_base(log, tokenized_dataset, val_dataloader, config, device):
    base_model = AutoModelForSequenceClassification.from_pretrained(config["model_name"],
                                                                    num_labels=config['num_labels']).to(
        device)

    for param in base_model.bert.parameters():
        param.requires_grad = False

    log.info(f'Number of parameters: {count_parameters(base_model)}')

    total_iterations = 0
    full_history = dict()

    # TODO: fix this, we are using this only to make sure experiments are consistent, since we do restarts for scheduler
    # in future just configure scheduler to restart and remove iterations, keeping only epochs
    iterations = config['iterations']

    for i in range(iterations):
        log.info('*' * 30 + f' Iteration: {i + 1} ' + '*' * 30)

        total_iterations += config['num_epochs']
        # total_iterations += 1

        history = training(base_model, tokenized_dataset['train'], tokenized_dataset['val'], log, config)

        full_history = {k: full_history.get(k, []) + v for k, v in history.items()}

        val_loss, val_preds, val_labels, cs_val_preds = eval_epoch(base_model, val_dataloader, config)

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_precision = precision_score(val_labels, val_preds, average='macro')
        val_recall = recall_score(val_labels, val_preds, average='macro')

        log.info(f'Val loss: {val_loss / len(val_dataloader)}')
        log.info(f'Val acc: {val_acc}')
        log.info(f'Val f1: {val_f1}')
        log.info(f'Val precision: {val_precision}')
        log.info(f'Val recall: {val_recall}')

    if not os.path.exists(f'models/{config["experiment_name"]}'):
        os.makedirs(f'models/{config["experiment_name"]}', exist_ok=True)

    full_model_path = f'models/{config["experiment_name"]}/{config["dataset_name"]}_{config["model_name"]}_{config["num_epochs"] * config["iterations"]}.bin'

    torch.save(base_model.state_dict(), full_model_path)

    return base_model, full_history


@log_continue
def train_space(log, tokenized_dataset, val_dataloader, config, device):
    base_model = AutoModel.from_pretrained(config['model_name']).to(device)

    space_model = SpaceModelForSequenceClassification(
        base_model,
        n_embed=768, n_latent=config['n_latent'],
        n_concept_spaces=config['num_labels'],
        l1=config['l1'],
        l2=config['l2'],
        ce_w=config['cross_entropy_weight'],
        fine_tune=True
    ).to(device)

    log.info(f'Number of space model parameters: {count_parameters(space_model)}')

    # estimation losses

    ids = tokenized_dataset['test'][0]['input_ids'].unsqueeze(0).to(device)
    mask = tokenized_dataset['test'][0]['attention_mask'].unsqueeze(0).to(device)
    targets = tokenized_dataset['test'][0]['label'].unsqueeze(0).to(device)

    base_embed = space_model.base_model(ids, mask).last_hidden_state

    concept_spaces = space_model.space_model(base_embed).concept_spaces

    log.debug(f'Inter-space loss: {space_model.l1 * inter_space_loss(concept_spaces, targets) * config["batch_size"]}',
              terminal=config['log_terminal'])
    log.debug(f'Intra-space loss: {space_model.l2 * intra_space_loss(concept_spaces) * config["batch_size"]}',
              terminal=config['log_terminal'])

    total_iterations = 0
    full_history = dict()

    best_results = {'loss': 0, 'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'cs_acc': 0, 'cs_f1': 0}

    iterations = config['iterations']

    for i in range(iterations):
        log.info('*' * 30 + f' Iteration: {i + 1} ' + '*' * 30)

        total_iterations += config['num_epochs']
        # total_iterations += 1

        space_history = training(space_model, tokenized_dataset['train'], tokenized_dataset['val'], log, config)

        full_history = {k: full_history.get(k, []) + v for k, v in space_history.items()}

        val_loss, val_preds, val_labels, cs_val_preds = eval_epoch(space_model, val_dataloader, config)

        cs_val_acc = accuracy_score(val_labels, cs_val_preds)
        cs_val_f1 = f1_score(val_labels, cs_val_preds, average='macro')

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_precision = precision_score(val_labels, val_preds, average='macro')
        val_recall = recall_score(val_labels, val_preds, average='macro')

        log.info(f'Val loss: {val_loss / len(val_dataloader)}')
        log.info(f'Val acc: {val_acc}')
        log.info(f'CS Val acc: {cs_val_acc}')
        log.info(f'Val f1: {val_f1}')
        log.info(f'CS Val f1: {cs_val_f1}')
        log.info(f'Val precision: {val_precision}')
        log.info(f'Val recall: {val_recall}')

        # track best metrics based on cs f1
        if cs_val_f1 > best_results['cs_f1']:
            best_results['loss'] = val_loss / len(val_dataloader)
            best_results['acc'] = val_acc
            best_results['f1'] = val_f1
            best_results['precision'] = val_precision
            best_results['recall'] = val_recall
            best_results['cs_acc'] = cs_val_acc
            best_results['cs_f1'] = cs_val_f1

    log.info('Best results:')
    for k, v in best_results.items():
        log.info(f'{k}: {v}')

    if not os.path.exists(f'models/{config["experiment_name"]}'):
        os.makedirs(f'models/{config["experiment_name"]}', exist_ok=True)

    full_model_name = f'models/{config["experiment_name"]}/{config["dataset_name"]}_space-{config["model_name"].replace("/", "_")}-({config["n_latent"]})_{config["num_epochs"] * config["iterations"]}.bin'

    log.info(f'Saving space model: {full_model_name}')

    torch.save(space_model.state_dict(), full_model_name)

    return space_model, full_history


@log_continue
def eval_results(log, results_path, model, val_dataloader, config):
    # base_model.load_state_dict(torch.load(f'models/{config["dataset_name"]}_{config["model_name"]}_{config["num_epochs"] * config["iterations"]}.bin'))
    # base_model.to(device)

    val_loss, val_preds, val_labels, cs_val_preds = eval_epoch(model, val_dataloader, config)

    if len(cs_val_preds) != 0:
        cs_val_acc = accuracy_score(val_labels, cs_val_preds)
        cs_val_f1 = f1_score(val_labels, cs_val_preds, average='macro')

    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    val_precision = precision_score(val_labels, val_preds, average='macro')
    val_recall = recall_score(val_labels, val_preds, average='macro')

    log.info(f'Val loss: {val_loss / len(val_dataloader)}')
    log.info(f'Val acc: {val_acc}')

    if len(cs_val_preds) != 0:
        log.info(f'CS Val acc: {cs_val_acc}')
    log.info(f'Val f1: {val_f1}')

    if len(cs_val_preds) != 0:
        log.info(f'CS Val f1: {cs_val_f1}')

    log.info(f'Val precision: {val_precision}')
    log.info(f'Val recall: {val_recall}')

    if not os.path.exists(f'results/{config["experiment_name"]}'):
        os.makedirs(f'results/{config["experiment_name"]}', exist_ok=True)

    with open(f'results/{config["experiment_name"]}/{results_path}_eval.txt', 'w') as f:
        f.writelines(
            [
                f'Val loss: {val_loss / len(val_dataloader)}\n',
                f'Val acc: {val_acc}\n',
                f'CS Val acc: {cs_val_acc}\n' if len(cs_val_preds) != 0 else 'CS Val acc: N/A\n',
                f'Val f1: {val_f1}\n',
                f'CS Val acc: {cs_val_f1}\n' if len(cs_val_preds) != 0 else 'CS Val f1: N/A\n',
                f'Val precision: {val_precision}\n',
                f'Val recall: {val_recall}\n'
            ]
        )


def run(config):
    base_name = f'{config["dataset_name"]}_{config["model_name"].replace("/", "_")}_{"space" if config["train_space"] else ""}_({config["n_latent"]})_{config["num_epochs"] * config["iterations"]}_{config["device_id"]}'

    log = get_logger(f'logs/{config["experiment_name"]}/', f'{base_name}.log')

    log.info('Starting...', terminal=config['log_terminal'])

    log.info(f'Config: {config}')

    # initialize device
    device_id = config['device_id']
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    seed_everything(seed=config['seed'])

    log.debug('Loading dataset...', terminal=config['log_terminal'])
    tokenized_dataset = config['get_data_func'](config['model_name'], config['max_seq_len'], device, config['seed'])

    val_dataset = tokenized_dataset['test']
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'])

    if config['train_base']:
        log.debug('Training base model...', terminal=config['log_terminal'])

        # this may return None if some error occured, and log_continue fired
        res = train_base(log, tokenized_dataset, val_dataloader, config, device)
        if res is not None:
            base_model, base_history = res
            free_gpu_cache(device_id)

            plot_results(log, base_history, base_name)

            log.debug('Evaluating base model on test set:')
            eval_results(log, base_name, base_model, val_dataloader, config)
        else:
            log.critical('Base model training failed...')

    if config['train_space']:
        log.debug('Training space model...', terminal=config['log_terminal'])

        space_name = f'{config["dataset_name"]}_space-{config["model_name"].replace("/", "_")}-({config["n_latent"]})_{config["num_epochs"] * config["iterations"]}_{config["device_id"]}'
        # this may return None if some error occured, and log_continue fired
        res = train_space(log, tokenized_dataset, val_dataloader, config, device)
        if res is not None:
            space_model, space_history = res
            free_gpu_cache(device_id)

            plot_results(log, space_history, space_name)

            log.debug('Evaluating space model on test set:')
            eval_results(log, space_name, space_model, val_dataloader, config)
        else:
            log.critical('Space model training failed...')


if __name__ == '__main__':
    MODEL_NAME = 'bert-base-cased'
    DATASET_NAME = 'fake-news-net'

    SEED = 42
    NUM_EPOCHS = 5
    BATCH_SIZE = 256
    MAX_SEQ_LEN = 512
    LEARNING_RATE = 2e-5
    MAX_GRAD_NORM = 1000
    N_LATENT = 3

    run({
        'experiment_name': 'default',
        'log_terminal': False,

        'device_id': 0,
        'train_base': True,
        'train_space': True,

        'seed': SEED,
        'dataset_name': DATASET_NAME,
        'model_name': MODEL_NAME,

        'num_labels': 2,
        'num_epochs': NUM_EPOCHS,
        'iterations': 1,

        'max_seq_len': MAX_SEQ_LEN,
        'batch_size': BATCH_SIZE,
        'lr': LEARNING_RATE,
        'fp16': False,
        'max_grad_norm': MAX_GRAD_NORM,
        'weight_decay': 0.01,
        'num_warmup_steps': 0,
        'gradient_accumulation_steps': 1,

        'n_latent': N_LATENT,
        'cross_entropy_weight': 1.0,
        'l1': 0.1,
        'l2': 1e-5,

        # funcs:
        'preds_from_logits_func': get_preds_from_logits,
        'get_data_func': prepare_dataset,
    })

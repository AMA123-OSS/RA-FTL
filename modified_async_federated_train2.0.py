import copy
import heapq
import itertools
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

warnings.filterwarnings('ignore')

# ----------------------
# 基础设置
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

folder_name = 'MMMM_FTL_async_container_var1'
folder_path = os.path.join("./", folder_name)
os.makedirs(folder_path, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----------------------
# 数据集
# ----------------------
class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.transform = transform
        try:
            self.dataframe = pd.read_csv(csv_file, header=None)
        except FileNotFoundError:
            print(f"Warning: {csv_file} not found, empty dataset will be used")
            self.dataframe = pd.DataFrame(columns=[0, 1])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        image = Image.open(img_name).convert('L')
        label = int(self.dataframe.iloc[idx, 1])
        label = torch.tensor(label, dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label

# ----------------------
# 模型
# ----------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(20 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def freeze_layers(self, freeze_option):
        if freeze_option == 'none':
            for param in self.parameters():
                param.requires_grad = True
        elif freeze_option == 'first':
            for name, param in self.named_parameters():
                if 'features.0' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        elif freeze_option == 'all_features':
            for name, param in self.named_parameters():
                if 'features' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

# ----------------------
# 工具函数
# ----------------------
def train_model(model, train_data, criterion, optimizer, num_epochs, compute_power):
    loader = DataLoader(train_data, batch_size=1, shuffle=True)
    if len(loader) == 0:
        return 0.0, 0.0
    model.train()
    total_loss = 0.0
    for _ in range(num_epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    training_time = len(loader) * num_epochs / compute_power
    return total_loss / len(loader), training_time

def get_client_csv(client_idx, com_folder, spec_folder):
    if client_idx < 4:
        return os.path.join(com_folder, f'client{client_idx}.csv')
    return os.path.join(spec_folder, f'client{client_idx - 4}.csv')

def get_sample_counts(com_folder, spec_folder):
    counts = []
    for i in range(4):
        file = os.path.join(com_folder, f'client{i}.csv')
        try:
            df = pd.read_csv(file, header=None)
            counts.append(len(df))
        except Exception:
            counts.append(1500)
    for i in range(8):
        file = os.path.join(spec_folder, f'client{i}.csv')
        try:
            df = pd.read_csv(file, header=None)
            counts.append(len(df))
        except Exception:
            counts.append(900)
    return counts

def average_weights(state_dicts, sample_counts):
    total = float(sum(sample_counts))
    avg_state = {}
    for key in state_dicts[0].keys():
        weighted_tensor = None
        for state, count in zip(state_dicts, sample_counts):
            part = state[key].detach().clone() * (count / total)
            weighted_tensor = part if weighted_tensor is None else weighted_tensor + part
        avg_state[key] = weighted_tensor
    return avg_state

def evaluate_model(model, test_data, criterion):
    loader = DataLoader(test_data, batch_size=1, shuffle=False)
    if len(loader) == 0:
        return 0.0, 0.0, None, 0.0, 0.0
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
    accuracy = 100 * correct / total if total > 0 else 0.0
    test_loss = total_loss / len(loader)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except Exception:
        auc = None
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return test_loss, accuracy, auc, precision, recall

def clone_state_dict(state_dict):
    return {k: v.detach().clone() for k, v in state_dict.items()}

def flatten_state_dict(state_dict):
    return torch.cat([state_dict[k].detach().float().reshape(-1).cpu() for k in sorted(state_dict.keys())])

def state_dict_similarity(state_a, state_b):
    vec_a = flatten_state_dict(state_a)
    vec_b = flatten_state_dict(state_b)
    norm_a = torch.norm(vec_a)
    norm_b = torch.norm(vec_b)
    if norm_a.item() == 0 or norm_b.item() == 0:
        return 1.0 if torch.allclose(vec_a, vec_b) else 0.0
    return torch.dot(vec_a, vec_b).item() / (norm_a.item() * norm_b.item())

def schedule_next_event(client_idx, start_time, client_models, client_versions, compute_powers, criterion, com_folder, spec_folder, event_counter):
    train_file = get_client_csv(client_idx, com_folder, spec_folder)
    train_data = ImageDataset(train_file, transform)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, client_models[client_idx].parameters()), lr=0.00015)
    _, local_time = train_model(client_models[client_idx], train_data, criterion, optimizer, num_epochs=1, compute_power=compute_powers[client_idx])
    ready_time = start_time + local_time
    upload_state = clone_state_dict(client_models[client_idx].classifier.state_dict())
    return ready_time, next(event_counter), client_idx, client_versions[client_idx], upload_state

# ----------------------
# 异步联邦训练
# ----------------------
def federated_train(num_rounds, freeze_option='none', agg_trigger_size=8, similarity_threshold=0.95,
                    convergence_window=5, convergence_variance_threshold=3.0):
    com_folder = "./COM_DirichletDistribution1"
    spec_folder = "./SPEC_DirichletDistribution0.5"

    compute_powers = {i: 5000 for i in range(0, 4)}
    compute_powers.update({i: 4000 for i in range(4, 8)})
    compute_powers.update({i: 3000 for i in range(8, 12)})

    num_clients = 12
    client_models = [CNN().to(device) for _ in range(num_clients)]
    global_model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    sample_counts = get_sample_counts(com_folder, spec_folder)

    pretrain_clients = [1, 2, 4, 5]
    other_clients = [i for i in range(num_clients) if i not in pretrain_clients]

    container_capacity = len(other_clients)
    if agg_trigger_size > container_capacity:
        raise ValueError("agg_trigger_size 不能大于主训练客户端数量")

    total_pretrain = 0.0
    global_feat = clone_state_dict(global_model.features.state_dict())
    T_pretrain = 100

    # --------------------
    # 预训练
    # --------------------
    for round_idx in range(T_pretrain):
        round_times = []
        for idx in pretrain_clients:
            model = client_models[idx]
            model.features.load_state_dict(global_feat)
            optimizer = optim.SGD(model.parameters(), lr=0.00015)
            train_file = get_client_csv(idx, com_folder, spec_folder)
            train_data = ImageDataset(train_file, transform)
            _, t = train_model(model, train_data, criterion, optimizer, 1, compute_powers[idx])
            round_times.append(t)
        max_t = max(round_times) if round_times else 0.0
        total_pretrain += max_t
        feat_dicts = [clone_state_dict(client_models[i].features.state_dict()) for i in pretrain_clients]
        pretrain_counts = [sample_counts[i] for i in pretrain_clients]
        global_feat = average_weights(feat_dicts, pretrain_counts)

    for idx in range(num_clients):
        client_models[idx].features.load_state_dict(global_feat)
        client_models[idx].freeze_layers(freeze_option)

    global_cls = clone_state_dict(global_model.classifier.state_dict())
    for idx in other_clients:
        client_models[idx].classifier.load_state_dict(global_cls)

    test_file = os.path.join(spec_folder, 'testAll.csv')
    test_data = ImageDataset(test_file, transform)

    accuracy_list, auc_list, precision_list, recall_list, testing_losses = [], [], [], [], []

    event_counter = itertools.count()
    event_heap = []
    container = []
    client_versions = {idx: 0 for idx in other_clients}
    client_participation = {idx: 0 for idx in other_clients}
    client_dropped = {idx: 0 for idx in other_clients}

    # 初始化事件
    for idx in other_clients:
        event = schedule_next_event(idx, 0.0, client_models, client_versions, compute_powers, criterion, com_folder, spec_folder, event_counter)
        heapq.heappush(event_heap, event)

    aggregation_round = 0
    converged = False
    convergence_round = None
    convergence_time = None
    federated_elapsed_time = 0.0

    while event_heap and aggregation_round < num_rounds and not converged:
        current_time, _, client_idx, version_tag, upload_state = heapq.heappop(event_heap)
        if version_tag != client_versions[client_idx]:
            continue
        federated_elapsed_time = max(federated_elapsed_time, current_time)

        # 重复上传检查
        duplicate_states = [item['state'] for item in container if item['client_id'] == client_idx]
        drop_update = False
        for old_state in duplicate_states:
            sim = state_dict_similarity(old_state, upload_state)
            if sim >= similarity_threshold:
                drop_update = True
                client_dropped[client_idx] += 1
                print(f"客户端 {client_idx} 上传被丢弃，原因：相似度 {sim:.4f} >= {similarity_threshold}")
                break

        if not drop_update and len(container) < container_capacity:
            container.append({'client_id': client_idx, 'state': upload_state, 'finish_time': current_time})
            client_participation[client_idx] += 1
            print(f"客户端 {client_idx} 上传进入容器 | 当前容器长度: {len(container)}/{agg_trigger_size}")

        # 下一轮训练
        next_event = schedule_next_event(client_idx, current_time, client_models, client_versions, compute_powers, criterion, com_folder, spec_folder, event_counter)
        heapq.heappush(event_heap, next_event)

        # 聚合
        if len(container) >= agg_trigger_size:
            participant_items = container[:agg_trigger_size]
            container = container[agg_trigger_size:]
            participant_states = [item['state'] for item in participant_items]
            participant_counts = [sample_counts[item['client_id']] for item in participant_items]
            global_cls = average_weights(participant_states, participant_counts)
            aggregation_round += 1

            participant_clients = sorted(set(item['client_id'] for item in participant_items))
            for pid in participant_clients:
                client_models[pid].classifier.load_state_dict(global_cls)
                client_versions[pid] += 1
                refreshed_event = schedule_next_event(pid, current_time, client_models, client_versions, compute_powers, criterion, com_folder, spec_folder, event_counter)
                heapq.heappush(event_heap, refreshed_event)

            global_model.features.load_state_dict(global_feat)
            global_model.classifier.load_state_dict(global_cls)
            test_loss, accuracy, auc, precision, recall = evaluate_model(global_model, test_data, criterion)

            accuracy_list.append(accuracy)
            auc_list.append(auc if auc is not None else 0.0)
            precision_list.append(precision)
            recall_list.append(recall)
            testing_losses.append(test_loss)

            print(f"\n聚合轮次 {aggregation_round} | 当前时间: {current_time:.2f}s")
            print(f"测试准确率: {accuracy:.2f}% | 精确率: {precision:.4f} | 召回率: {recall:.4f} | 测试损失: {test_loss:.4f}")
            print(f"参与聚合客户端: {participant_clients}")
            print(f"客户端参与次数: {client_participation}")
            print(f"客户端被丢弃次数: {client_dropped}")

            # 收敛判定
            if len(accuracy_list) >= convergence_window:
                recent_var = float(np.var(accuracy_list[-convergence_window:]))
                print(f"最近 {convergence_window} 次准确率方差: {recent_var:.4f}")
                if recent_var < convergence_variance_threshold:
                    converged = True
                    convergence_round = aggregation_round
                    convergence_time = total_pretrain + current_time
                    print(f"达到收敛条件：最近 {convergence_window} 次准确率方差 = {recent_var:.4f} < {convergence_variance_threshold}")

    total_time = total_pretrain + federated_elapsed_time

    # T@70/T@75/T@80
    thresholds = [0.7, 0.75, 0.8]
    T_threshold = {}
    for t in thresholds:
        T_threshold[f"T@{int(t*100)}"] = next((total_time * (i+1)/len(accuracy_list) for i, acc in enumerate(accuracy_list) if acc/100 >= t), None)

    # 最终报告
    report = {
        "Final Accuracy": accuracy_list[-1] if accuracy_list else None,
        "Convergence Time": convergence_time if convergence_time else total_time,
        **T_threshold,
        "Client Participation": client_participation,
        "Client Dropped Updates": client_dropped
    }

    pd.DataFrame([report]).to_csv(os.path.join(folder_path, 'final_report.csv'), index=False)
    print("\n训练完成，最终报告已生成：")
    print(report)

    return accuracy_list, auc_list, precision_list, recall_list, testing_losses, report

# ----------------------
# 运行训练
# ----------------------
freeze_option = 'all_features'

accuracy_list, auc_list, precision_list, recall_list, testing_losses, final_report = federated_train(
    num_rounds=100,
    freeze_option=freeze_option,
    agg_trigger_size=8,
    similarity_threshold=0.90,
    convergence_window=5,
    convergence_variance_threshold=2,
)
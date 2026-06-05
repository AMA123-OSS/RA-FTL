"""
单数据集算力异构场景
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import random
import collections
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# 检查GPU可用性
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 创建用于存储输出数据的新文件夹
folder_name = 'HMMM_FTL_100_freeze_all_1'
print(folder_name)
folder_path = os.path.join("./", folder_name)
os.makedirs(folder_path, exist_ok=True)

# 指定包含测试集和训练集的文件夹路径
data_folder_path = "./DirichletDistribution100"


# 加载和处理数据集
class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        try:
            self.dataframe = pd.read_csv(csv_file, header=None)
        except FileNotFoundError:
            # 如果文件不存在，尝试使用备用文件
            backup_file = csv_file.replace('client8', 'client0').replace('client9', 'client1').replace('client10',
                                                                                                       'client2').replace(
                'client11', 'client3')
            print(f"Warning: {csv_file} not found, using {backup_file} instead")
            self.dataframe = pd.read_csv(backup_file, header=None)
        self.transform = transform

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


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# 修改CNN模型，明确分割为特征提取器和分类器
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 特征提取器 (w_i^1)
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 20, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 分类器 (w_i^2)
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
        """
        根据选项冻结特定层
        freeze_option:
            'none' - 不冻结任何层
            'first' - 只冻结第一层卷积
            'all_features' - 冻结所有特征提取层
        """
        if freeze_option == 'none':
            # 解冻所有层
            for param in self.parameters():
                param.requires_grad = True

        elif freeze_option == 'first':
            # 只冻结第一层卷积
            for name, param in self.named_parameters():
                if 'features.0' in name:  # 第一层卷积
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        elif freeze_option == 'all_features':
            # 冻结所有特征提取层
            for name, param in self.named_parameters():
                if 'features' in name:  # 所有特征提取层
                    param.requires_grad = False
                else:
                    param.requires_grad = True


# 训练模型函数（返回训练时间）
def train_model(model, train_data, criterion, optimizer, num_epochs, compute_power):
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    model.train()

    # 计算训练时间：迭代次数 / 算力
    num_iterations = len(train_loader) * num_epochs
    training_time = num_iterations / compute_power

    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    return average_loss, training_time


# 获取样本数量
def get_sample_counts(data_folder_path, i_range):
    sample_counts = []
    for i in i_range:
        csv_file = os.path.join(data_folder_path, f'client{i}.csv')
        try:
            df = pd.read_csv(csv_file, header=None)
            sample_counts.append(len(df))
        except FileNotFoundError:
            # 如果文件不存在，使用对应档次的默认样本数
            print("不存在客户端数据，直接指定数值")
            if i < 4:  # 高档
                sample_counts.append(1500)  # 估计值
            elif i < 8:  # 中档
                sample_counts.append(1200)  # 估计值
            else:  # 低档
                sample_counts.append(900)  # 估计值
    return sample_counts


# 权重平均函数
def average_weights(state_dicts, sample_counts):
    average_dict = {}
    total_samples = sum(sample_counts)
    for key in state_dicts[0].keys():
        key_params = []
        for state_dict, count in zip(state_dicts, sample_counts):
            key_params.append(state_dict[key] * count)
        weighted_sum = sum(key_params)
        average_dict[key] = weighted_sum / total_samples
    return average_dict


# 模型评估函数
def evaluate_model(model, test_data, criterion):
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for images, labels in test_loader:
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
            all_probs.extend(probs.detach().cpu().numpy())
    test_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = None
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    return test_loss, accuracy, auc, precision, recall


# 联邦训练函数
def federated_train(num_rounds, freeze_option='none'):
    # 定义算力配置
    compute_powers = {
        0: 5090, 1: 5090, 2: 5090, 3: 5090,  # 高档: 5090
        4: 4060, 5: 4060, 6: 4060, 7: 4060,  # 中档: 4060
        8: 3050, 9: 3050, 10: 3050, 11: 3050  # 低档: 3050
    }

    num_clients = 12
    global_model = CNN().to(device)
    client_models = [CNN().to(device) for _ in range(num_clients)]
    criterion = nn.CrossEntropyLoss()

    # 获取样本数量
    sample_counts = get_sample_counts(data_folder_path, range(num_clients))

    # 指定预训练客户端（1-4，对应索引0-3）
    pretrain_clients = [0, 1, 4, 5]
    other_clients = [i for i in range(num_clients) if i not in pretrain_clients]

    print(f"预训练客户端: {pretrain_clients}")
    print(f"其他客户端: {other_clients}")
    print(f"样本数量: {sample_counts}")

    # 时间统计
    total_pretrain_time = 0
    total_federated_time = 0

    # 预训练阶段：训练特征提取器
    print("预训练阶段: 训练特征提取器")
    global_feature_extractor = global_model.features.state_dict()
    T_pretrain = 100  # 预训练轮数

    for round in range(T_pretrain):
        round_times = []

        # 训练每个预训练客户端
        for client_index in pretrain_clients:
            model = client_models[client_index]
            model.features.load_state_dict(global_feature_extractor)
            optimizer = optim.SGD(model.parameters(), lr=0.00015)

            train_data = ImageDataset(
                csv_file=os.path.join(data_folder_path, f'client{client_index}.csv'),
                transform=transform
            )

            _, client_time = train_model(
                model, train_data, criterion, optimizer,
                num_epochs=1, compute_power=compute_powers[client_index]
            )
            round_times.append(client_time)

        # 记录本轮最长训练时间
        max_round_time = max(round_times)
        total_pretrain_time += max_round_time

        # 聚合特征提取器参数
        feature_dicts = [client_models[i].features.state_dict() for i in pretrain_clients]
        pretrain_sample_counts = [sample_counts[i] for i in pretrain_clients]
        global_feature_extractor = average_weights(feature_dicts, pretrain_sample_counts)

        if (round + 1) % 1 == 0:
            print(f"预训练轮次 {round + 1}/{T_pretrain}, 本轮最长耗时: {max_round_time:.4f}s")

    print(f"预训练总时间: {total_pretrain_time:.4f}s")

    # 将预训练的特征提取器共享给所有客户端并根据选项冻结层
    for client_index in range(num_clients):
        client_models[client_index].features.load_state_dict(global_feature_extractor)
        client_models[client_index].freeze_layers(freeze_option)

    # 主要训练阶段：训练分类器
    print("主要训练阶段: 训练分类器")
    global_classifier = global_model.classifier.state_dict()

    accuracy_list = []
    auc_list = []
    precision_list = []
    recall_list = []
    testing_losses = []

    converged = False
    convergence_window = 5
    convergence_threshold = 0.05  # 0.05% 准确率差异
    round_num = 0

    while not converged and round_num < num_rounds:
        round_times = []

        # 训练每个其他客户端
        for client_index in other_clients:
            model = client_models[client_index]
            model.classifier.load_state_dict(global_classifier)

            # 只优化可训练的参数
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = optim.SGD(trainable_params, lr=0.00015)

            train_data = ImageDataset(
                csv_file=os.path.join(data_folder_path, f'client{client_index}.csv'),
                transform=transform
            )

            _, client_time = train_model(
                model, train_data, criterion, optimizer,
                num_epochs=1, compute_power=compute_powers[client_index]
            )
            round_times.append(client_time)

        # 记录本轮最长训练时间
        max_round_time = max(round_times)
        total_federated_time += max_round_time

        # 聚合分类器参数
        classifier_dicts = [client_models[i].classifier.state_dict() for i in other_clients]
        other_sample_counts = [sample_counts[i] for i in other_clients]
        global_classifier = average_weights(classifier_dicts, other_sample_counts)

        # 更新全局模型
        global_model.features.load_state_dict(global_feature_extractor)
        global_model.classifier.load_state_dict(global_classifier)

        # 评估全局模型
        test_data = ImageDataset(
            csv_file=os.path.join(data_folder_path, 'testAll.csv'),
            transform=transform
        )
        test_loss, accuracy, auc, precision, recall = evaluate_model(global_model, test_data, criterion)

        testing_losses.append(test_loss)
        accuracy_list.append(accuracy)
        auc_list.append(auc if auc is not None else 0)
        precision_list.append(precision)
        recall_list.append(recall)

        # 检查收敛条件
        if len(accuracy_list) >= convergence_window:
            recent_accuracies = accuracy_list[-convergence_window:]
            max_recent = max(recent_accuracies)
            min_recent = min(recent_accuracies)
            if max_recent - min_recent <= convergence_threshold:
                converged = True
                print(f"在第 {round_num + 1} 轮收敛")

        round_num += 1

        if round_num % 1 == 0 or converged:
            # 同样先处理 AUC 显示
            auc_str = f"{auc:.4f}" if auc is not None else "N/A"

            status = f"轮次 {round_num}"
            if converged:
                status += " (已收敛)"

            print(f"{status} | 测试损失: {test_loss:.4f} | 准确率: {accuracy:.2f}% | "
                  f"精确率: {precision:.4f} | 召回率: {recall:.4f} | "
                  f"AUC: {auc_str} | 本轮耗时: {max_round_time:.4f}s")

    # 计算总训练时间
    total_training_time = total_pretrain_time + total_federated_time
    print(f"\n总训练时间: {total_training_time:.4f}s")
    print(f"- 预训练时间: {total_pretrain_time:.4f}s ({T_pretrain}轮)")
    print(f"- 联邦训练时间: {total_federated_time:.4f}s ({round_num}轮)")

    # 保存时间信息
    time_info = {
        'total_time': total_training_time,
        'pretrain_time': total_pretrain_time,
        'federated_time': total_federated_time,
        'convergence_round': round_num
    }

    return accuracy_list, auc_list, precision_list, recall_list, testing_losses, time_info


# 选择冻结选项
freeze_option = 'all_features'  # 'none', 'first', 或 'all_features'

# 训练并评估联邦学习模型
accuracy_list, auc_list, precision_list, recall_list, testing_losses, time_info = federated_train(
    num_rounds=100,
    freeze_option=freeze_option
)

# 保存结果
accuracy_df = pd.DataFrame(accuracy_list, columns=["Accuracy"])
accuracy_df.to_csv(os.path.join(folder_path, 'accuracy.csv'), index=False)

auc_df = pd.DataFrame(auc_list, columns=["AUC"])
auc_df.to_csv(os.path.join(folder_path, 'auc.csv'), index=False)

precision_df = pd.DataFrame(precision_list, columns=["Precision"])
precision_df.to_csv(os.path.join(folder_path, 'precision.csv'), index=False)

recall_df = pd.DataFrame(recall_list, columns=["Recall"])
recall_df.to_csv(os.path.join(folder_path, 'recall.csv'), index=False)

test_df = pd.DataFrame(testing_losses, columns=["Testing Loss"])
test_df.to_csv(os.path.join(folder_path, 'testing_loss.csv'), index=False)

# 保存时间信息
time_df = pd.DataFrame([time_info])
time_df.to_csv(os.path.join(folder_path, 'training_time.csv'), index=False)

# 绘制准确率折线图
plt.figure(figsize=(10, 6))
plt.plot(range(len(accuracy_list)), accuracy_list)
plt.xlim(0, len(accuracy_list))
plt.ylim(0, 100)
plt.xlabel('Rounds')
plt.ylabel('Test Accuracy')
plt.title(f'FTL Accuracy with {freeze_option} freezing\nTotal Time: {time_info["total_time"]:.2f}s')
plt.savefig(os.path.join(folder_path, 'out1.png'))
plt.close()

print("训练完成，结果已保存。")
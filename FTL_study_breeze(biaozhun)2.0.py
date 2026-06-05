"""
    添加了相似客户端竞争预训练模型的步骤
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import itertools
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
folder_name = '2_FTL_0.5_freeze_first_Pretrain_Optimal5'
folder_path = os.path.join(".", folder_name)
os.makedirs(folder_path, exist_ok=True)

data_folder_path = "/root/autodl-tmp/DirichletDistribution0.5"
# 数据集定义
class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
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

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 特征提取器
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 20, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(20*32*32, 128),
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

# 训练函数
def train_model(model, train_data, criterion, optimizer, num_epochs=1):
    loader = DataLoader(train_data, batch_size=1, shuffle=True)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)

# 权重平均函数
def average_weights(state_dicts, sample_counts):
    avg_dict = {}
    total = sum(sample_counts)
    for key in state_dicts[0].keys():
        avg_dict[key] = sum(state_dict[key]*count for state_dict, count in zip(state_dicts, sample_counts))/total
    return avg_dict

# 模型评估
def evaluate_model(model, test_data, criterion):
    loader = DataLoader(test_data, batch_size=1, shuffle=False)
    model.eval()
    correct, total, total_loss = 0, 0, 0
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
    acc = 100 * correct / total
    test_loss = total_loss / len(loader)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = 0
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return test_loss, acc, auc, precision, recall

# 遍历所有客户端组合进行预训练
def pretrain_select_best_combination(N_clients=8, P_pretrain=2, freeze_option='first'):
    client_models = [CNN().to(device) for _ in range(N_clients)]
    global_model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # 获取样本数量
    sample_counts = [len(pd.read_csv(os.path.join(data_folder_path, f'client{i}.csv'))) for i in range(N_clients)]
    
    # 加载测试集
    test_data = ImageDataset(csv_file=os.path.join(data_folder_path, 'small_sample_test.csv'), transform=transform)

    # 遍历所有组合
    all_combinations = list(itertools.combinations(range(N_clients), P_pretrain))
    best_acc = -1
    best_feature_state = None
    best_combination = None

    for comb in all_combinations:
        # 初始化特征提取器
        global_feat = global_model.features.state_dict()
        # 预训练阶段
        T_pretrain = 10  # 可以调整100
        for round in range(T_pretrain):
            for idx in comb:
                model = client_models[idx]
                model.features.load_state_dict(global_feat)
                optimizer = optim.SGD(model.parameters(), lr=0.00015)
                train_data = ImageDataset(csv_file=os.path.join(data_folder_path, f'client{idx}.csv'), transform=transform)
                train_model(model, train_data, criterion, optimizer, num_epochs=1)
            # 聚合特征
            feature_dicts = [client_models[i].features.state_dict() for i in comb]
            comb_sample_counts = [sample_counts[i] for i in comb]
            global_feat = average_weights(feature_dicts, comb_sample_counts)
        # 测试组合
        global_model.features.load_state_dict(global_feat)
        # 分类器随机初始化
        # 分类器随机初始化并移动到 device
        global_model.classifier = CNN().classifier.to(device)
        _, acc, _, _, _ = evaluate_model(global_model, test_data, criterion)
        if acc > best_acc:
            best_acc = acc
            best_feature_state = global_feat
            best_combination = comb

    print(f"最佳预训练客户端组合: {best_combination}, 测试集准确率: {best_acc:.2f}%")
    # 冻结特征
    for model in client_models:
        model.features.load_state_dict(best_feature_state)
        model.freeze_layers(freeze_option)
    
    return client_models, best_feature_state, best_combination

# 主训练阶段
def main_training(client_models, num_rounds=100):
    global_model = CNN().to(device)
    global_model.features.load_state_dict(client_models[0].features.state_dict())
    criterion = nn.CrossEntropyLoss()

    N_clients = len(client_models)
    # 全局分类器初始化
    global_classifier = global_model.classifier.state_dict()
    accuracy_list, auc_list, precision_list, recall_list, testing_losses = [], [], [], [], []

    other_clients = list(range(N_clients))
    test_data = ImageDataset(csv_file=os.path.join(data_folder_path, 'testAll.csv'), transform=transform)

    for round in range(num_rounds):
        # 每个客户端训练分类器
        for idx in other_clients:
            model = client_models[idx]
            model.classifier.load_state_dict(global_classifier)
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = optim.SGD(trainable_params, lr=0.00015)
            train_data = ImageDataset(csv_file=os.path.join(data_folder_path, f'client{idx}.csv'), transform=transform)
            train_model(model, train_data, criterion, optimizer, num_epochs=1)
        # 聚合分类器
        classifier_dicts = [client_models[i].classifier.state_dict() for i in other_clients]
        sample_counts = [len(pd.read_csv(os.path.join(data_folder_path, f'client{i}.csv'))) for i in other_clients]
        global_classifier = average_weights(classifier_dicts, sample_counts)
        # 更新全局模型
        global_model.features.load_state_dict(client_models[0].features.state_dict())
        global_model.classifier.load_state_dict(global_classifier)
        # 测试
        test_loss, acc, auc, prec, rec = evaluate_model(global_model, test_data, criterion)
        testing_losses.append(test_loss)
        accuracy_list.append(acc)
        auc_list.append(auc)
        precision_list.append(prec)
        recall_list.append(rec)
        print(f"Round {round+1}: Acc={acc:.2f}% | Loss={test_loss:.4f}")

    return accuracy_list, auc_list, precision_list, recall_list, testing_losses


client_models, best_feature_state, best_combination = pretrain_select_best_combination(N_clients=8, P_pretrain=2, freeze_option='first')

accuracy_list, auc_list, precision_list, recall_list, testing_losses = main_training(client_models, num_rounds=100)

pd.DataFrame(accuracy_list, columns=["Accuracy"]).to_csv(os.path.join(folder_path,'accuracy.csv'), index=False)
pd.DataFrame(auc_list, columns=["AUC"]).to_csv(os.path.join(folder_path,'auc.csv'), index=False)
pd.DataFrame(precision_list, columns=["Precision"]).to_csv(os.path.join(folder_path,'precision.csv'), index=False)
pd.DataFrame(recall_list, columns=["Recall"]).to_csv(os.path.join(folder_path,'recall.csv'), index=False)
pd.DataFrame(testing_losses, columns=["Testing Loss"]).to_csv(os.path.join(folder_path,'testing_loss.csv'), index=False)

plt.figure(figsize=(10,6))
plt.plot(range(len(accuracy_list)), accuracy_list)
plt.xlabel('Rounds')
plt.ylabel('Test Accuracy')
plt.title('FTL Accuracy with Frozen Pretrain Features')
plt.savefig(os.path.join(folder_path,'out1.png'))
plt.close()

print("训练完成，结果已保存。")
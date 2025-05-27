import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 自定义数据集
# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, ast_list, cfg_list, cg_list, label_list,contract_list):
        self.ast_list = ast_list
        self.cfg_list = cfg_list
        self.cg_list = cg_list
        self.label_list = label_list
        self.contract_list = contract_list

        # 计算每个输入类型的均值和标准差
        self.ast_mean = torch.mean(torch.stack(self.ast_list), dim=0)
        self.ast_std = torch.std(torch.stack(self.ast_list), dim=0)
        
        self.cfg_mean = torch.mean(torch.stack(self.cfg_list), dim=0)
        self.cfg_std = torch.std(torch.stack(self.cfg_list), dim=0)
        
        self.cg_mean = torch.mean(torch.stack(self.cg_list), dim=0)
        self.cg_std = torch.std(torch.stack(self.cg_list), dim=0)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        ast = (self.ast_list[idx] - self.ast_mean) / (self.ast_std + 1e-7)  # 标准化 AST
        cfg = (self.cfg_list[idx] - self.cfg_mean) / (self.cfg_std + 1e-7)  # 标准化 CFG
        cg = (self.cg_list[idx] - self.cg_mean) / (self.cg_std + 1e-7)      # 标准化 CG
        label = self.label_list[idx]
        contract = self.contract_list[idx]
        return (ast, cfg, cg, label,contract)

# 数据集划分
def split_data(ast_list, cfg_list, cg_list, label_list,contract_list, test_size=0.2):
    ast_train, ast_test, cfg_train, cfg_test, cg_train, cg_test, label_train, label_test,contract_train,contract_test = train_test_split(
        ast_list, cfg_list, cg_list, label_list,contract_list, test_size=test_size, random_state=42
    )
    return (ast_train, cfg_train, cg_train, label_train,contract_train), (ast_test, cfg_test, cg_test, label_test,contract_test)

# 注意力机制模型
class AttentionFusion(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(AttentionFusion, self).__init__()
        # 定义线性层来学习注意力权重
        self.ast_fc = nn.Linear(input_dim, input_dim)
        self.cfg_fc = nn.Linear(input_dim, input_dim)
        self.cg_fc = nn.Linear(input_dim, input_dim)
        
        # 最终的分类器
        self.classifier = nn.Linear(input_dim*2, num_classes)
        

    def forward(self, ast, cfg, cg):
        # 计算注意力权重
        ast_w = torch.sigmoid(self.ast_fc(ast))
        cfg_w = torch.sigmoid(self.cfg_fc(cfg))
        cg_w = torch.sigmoid(self.cg_fc(cg))
        #print(cfg_w)
        # 对输入进行加权求和
        #fusion = ast_w * ast + cfg_w * cfg + cg_w * cg
        fusion = torch.cat((ast_w * ast,cfg_w * cfg), dim=-1)  # 沿最后一个维度拼接

        # 分类输出
        output = self.classifier(fusion)
        
        return fusion,output,cfg_w
    

# 训练函数
def train(model, optimizer, criterion, dataloader,test_loader, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for ast, cfg, cg, labels,name in dataloader:
            ast, cfg, cg, labels = ast.to(device), cfg.to(device), cg.to(device), labels.to(device)
            
            optimizer.zero_grad()
            fusing,outputs,cfg_w = model(ast, cfg, cg)
            #print(f'Outputs shape: {outputs.shape}, Labels shape: {labels.shape}')

            loss = criterion(outputs, labels.view(-1))  # 去掉多余的维度
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        for ast, cfg, cg, labels, name in test_loader:
            ast, cfg, cg, labels = ast.to(device), cfg.to(device), cg.to(device), labels.to(device)
            
            optimizer.zero_grad()
            fusing,outputs,cfg_w = model(ast, cfg, cg)
            #print(f'Outputs shape: {outputs.shape}, Labels shape: {labels.shape}')

            loss = criterion(outputs, labels.view(-1))  # 去掉多余的维度
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        #print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

# 测试函数
# 测试函数
def evaluate(model, dataloader,train_loader,save_path):
    model.eval()
    all_labels = []
    all_predictions = []
    fusion_data = []
    cfg_w_data = []
    with torch.no_grad():
        for ast, cfg, cg, labels,name in dataloader:
            ast, cfg, cg, labels = ast.to(device), cfg.to(device), cg.to(device), labels.to(device)
            fusion,outputs,cfg_w = model(ast, cfg, cg)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            fusions = fusion.cpu().numpy()
            labels = labels.cpu().numpy()
            #print(labels)
            if "reentrance.sol" in name:
                print(name)
                for i in range(cfg_w.shape[0]):
                    d ={name[i]:cfg_w[i].tolist()}
                    cfg_w_data.append(d)
                with open("222.json", 'w') as f:
                    json.dump(cfg_w_data, f, indent=4)  # indent=4 使输出更可读
                break
                
            for i in range(fusions.shape[0]):  # fusions.shape[0] 是 batch size
                fusion_dict = {
                    "fusion": fusions[i].tolist(),  # 将 NumPy 数组转换为列表
                    "label": int(labels[i].item())
                }
                fusion_data.append(fusion_dict)  # 添加到列表
        for ast, cfg, cg, labels, name in train_loader:
            ast, cfg, cg, labels = ast.to(device), cfg.to(device), cg.to(device), labels.to(device)
            fusion,outputs,cfg_w = model(ast, cfg, cg)
            
            if "reentrance.sol" in name:
                print(name)
                for i in range(cfg_w.shape[0]):
                    #print(cfg_w)
                    d ={name[i]:cfg_w[i].tolist()}
                    cfg_w_data.append(d)
                with open("333.json", 'w') as f:
                    json.dump(cfg_w_data, f, indent=4)  # indent=4 使输出更可读
                break
                
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            fusions = fusion.cpu().numpy()
            labels = labels.cpu().numpy()
            for i in range(fusions.shape[0]):  # fusions.shape[0] 是 batch size
                fusion_dict = {
                    "fusion": fusions[i].tolist(),  # 将 NumPy 数组转换为列表
                    "label": int(labels[i].item())
                }
                fusion_data.append(fusion_dict)  # 添加到列表

    # 将 fusion_data 列表保存为 JSON 文件
    with open("111.json", 'w') as f:
        json.dump(fusion_data, f, indent=4)  # indent=4 使输出更可读

    # 计算指标
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    #accuracy = (all_predictions == all_labels).mean()  # 修复此行
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    #print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1 Score: {f1 * 100:.2f}%')

# 加载 JSON 数据
def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    # 假设 ast_list, cfg_list, cg_list, label_list 已经加载好
    bug_lists = ['reentrancy']

    

    for bug_type in bug_lists:
        ast_list = []
        cfg_list = []
        cg_list = []
        label_list = []
        contract_list = []
        path_ast = f'logs/graph_classification/source_code/metapath2vec/ast/{bug_type}/hiddens.json'
        path_cfg = f'logs/graph_classification/source_code/metapath2vec/cfg/{bug_type}/hiddens.json'
        path_cg = f'logs/graph_classification/source_code/metapath2vec/cg/{bug_type}/hiddens.json'

        ast_data = load_json(path_ast)
        cfg_data = load_json(path_cfg)
        cg_data = load_json(path_cg)

        for ast_object in ast_data:
            contract_name = ast_object['contract_name']
            print(contract_name)
            ast_vector = torch.tensor(ast_object['hiddens'], dtype=torch.float32)
            ast_list.append(ast_vector)
            contract_list.append(contract_name)

            label = torch.tensor(ast_object['targets'], dtype=torch.long).unsqueeze(0)  
            label_list.append(label)

            # 获取 cfg_vector 和 cg_vector
            cfg_vector = next((torch.tensor(cfg_object['hiddens'], dtype=torch.float32) for cfg_object in cfg_data if cfg_object['contract_name'] == contract_name), None)
            cg_vector = next((torch.tensor(cg_object['hiddens'], dtype=torch.float32) for cg_object in cg_data if cg_object['contract_name'] == contract_name), None)

            if cfg_vector is not None:
                cfg_list.append(cfg_vector)

            if cg_vector is not None:
                cg_list.append(cg_vector)

        # 数据集划分
        (ast_train, cfg_train, cg_train, label_train,contract_train), (ast_test, cfg_test, cg_test, label_test,contract_test) = split_data(
            ast_list, cfg_list, cg_list, label_list,contract_list
        )

        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建自定义数据集
        train_dataset = CustomDataset(ast_train, cfg_train, cg_train, label_train,contract_train)
        test_dataset = CustomDataset(ast_test, cfg_test, cg_test, label_test,contract_test)

        # 使用 DataLoader 创建批次
        batch_size = 4  # 设置你想要的批次大小
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 定义模型、损失函数和优化器
        input_dim = ast_train[0].shape[0]  # 假设输入的每个向量维度相同
        num_classes = len(torch.unique(torch.tensor(label_list)))  # 类别数
        model = AttentionFusion(input_dim, num_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # 训练模型
        train(model, optimizer, criterion, train_loader,test_loader, num_epochs=20)
        save_path = f'logs/graph_classification/source_code/metapath2vec/ast/{bug_type}/ast_cfg.json'
        # 测试模型
        evaluate(model, test_loader,train_loader,save_path)

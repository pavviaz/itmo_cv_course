import os
import random
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm
import fiftyone as fo
from pytorch_metric_learning import losses
import matplotlib.pyplot as plt
import yaml
import argparse
import numpy as np
import faiss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TripletFODataset(Dataset):
    def __init__(self, samples, transform=None, label_to_idx=None):
        """
        Параметры:
            samples (list): Список кортежей (filepath, label) – путь к изображению и его строковая метка.
            transform: Трансформации для изображения.
            label_to_idx (dict): Словарь для отображения строковой метки в числовой индекс.
                            Если None, он будет вычислен по списку samples.
        """
        self.transform = transform
        # Если не передан mapping, вычисляем его из всех меток
        if label_to_idx is None:
            labels = sorted({label for _, label in samples})
            self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx

        # Преобразуем метки в числовые индексы
        self.samples = [
            (filepath, self.label_to_idx[label]) for filepath, label in samples
        ]

        # Построим словарь: для каждого класса список индексов образцов данного класса
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(self.samples):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Возвращает кортеж:
        (anchor_img, positive_img, negative_img, anchor_label, negative_label)
        """
        filepath, anchor_label = self.samples[index]
        anchor_img = Image.open(filepath).convert("RGB")
        if self.transform:
            anchor_img = self.transform(anchor_img)

        # Выбираем позитив: другое изображение того же класса
        positive_index = index
        while positive_index == index:
            positive_index = random.choice(self.class_to_indices[anchor_label])
        positive_filepath, _ = self.samples[positive_index]
        positive_img = Image.open(positive_filepath).convert("RGB")
        if self.transform:
            positive_img = self.transform(positive_img)

        # Выбираем негатив: изображение из другого класса
        negative_label = anchor_label
        while negative_label == anchor_label:
            negative_label = random.choice(list(self.class_to_indices.keys()))
        negative_index = random.choice(self.class_to_indices[negative_label])
        negative_filepath, negative_label = self.samples[negative_index]
        negative_img = Image.open(negative_filepath).convert("RGB")
        if self.transform:
            negative_img = self.transform(negative_img)

        # Приводим метки к тензорам
        return (
            anchor_img,
            positive_img,
            negative_img,
            torch.tensor(anchor_label),
            torch.tensor(negative_label),
        )


class EmbeddingNet(nn.Module):
    def __init__(self, backbone_name="resnet18", embedding_dim=128, pretrained=True):
        """
        Модель-эмбеддер, использующая бэкбон из timm и дополнительный FC слой.
        Параметры:
            backbone_name (str): Имя модели-бэкбона (например, "resnet18").
            embedding_dim (int): Размерность выходного эмбеддинга.
            pretrained (bool): Использовать ли предобученные веса.
        """
        super(EmbeddingNet, self).__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        backbone_features = self.backbone.num_features
        self.fc = nn.Linear(backbone_features, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


def train_one_epoch(model, dataloader, optimizer, device, config):
    model.train()
    running_loss = 0.0
    loss_type = config["training"]["loss_type"]
    margin = config["training"]["margin"]
    sampling_strategy = config["training"]["sampling_strategy"]

    if loss_type == "triplet":
        criterion = nn.TripletMarginLoss(margin=margin, p=2)
    elif loss_type == "proxy_nca":
        criterion = losses.ProxyNCALoss(
            num_classes=len(dataloader.dataset.label_to_idx),
            embedding_size=config["model"]["embedding_dim"],
        )
        criterion.to(device)

    for batch_idx, batch in tqdm(enumerate(dataloader)):
        anchor, positive, negative, anchor_label, negative_label = batch
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        anchor_label = anchor_label.to(device)
        negative_label = negative_label.to(device)

        optimizer.zero_grad()
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)

        if loss_type == "triplet":
            if sampling_strategy == "semi_hard":
                candidate_embeddings = torch.cat([anchor_out, negative_out], dim=0)
                candidate_labels = torch.cat([anchor_label, negative_label], dim=0)
                batch_loss = 0.0
                batch_size = anchor_out.size(0)
                for i in range(batch_size):
                    d_ap = torch.norm(anchor_out[i] - positive_out[i], p=2)
                    mask = candidate_labels != anchor_label[i]
                    if mask.sum() == 0:
                        chosen_negative = negative_out[i]
                    else:
                        candidate_emb = candidate_embeddings[mask]
                        d_an = torch.norm(
                            anchor_out[i].unsqueeze(0) - candidate_emb, p=2, dim=1
                        )
                        semi_hard_mask = (d_an > d_ap) & (d_an < d_ap + margin)
                        if semi_hard_mask.sum() > 0:
                            candidate_d_an = d_an[semi_hard_mask]
                            chosen_idx = torch.argmin(candidate_d_an)
                            chosen_negative = candidate_emb[semi_hard_mask][chosen_idx]
                        else:
                            chosen_negative = negative_out[i]
                    d_an_final = torch.norm(anchor_out[i] - chosen_negative, p=2)
                    loss_i = torch.relu(d_ap - d_an_final + margin)
                    batch_loss += loss_i
                loss = batch_loss / batch_size
            elif sampling_strategy == "batch_hard":
                batch_embeddings = torch.cat(
                    [anchor_out, positive_out, negative_out], dim=0
                )
                batch_labels = torch.cat(
                    [anchor_label, anchor_label, negative_label], dim=0
                )
                batch_size = anchor_out.size(0)
                hard_positives = torch.zeros_like(anchor_out)
                hard_negatives = torch.zeros_like(anchor_out)

                for i in range(batch_size):
                    distances = torch.norm(
                        anchor_out[i].unsqueeze(0) - batch_embeddings, p=2, dim=1
                    )
                    pos_mask = batch_labels == anchor_label[i]
                    pos_distances = distances[pos_mask]
                    hard_positive_idx = torch.argmax(pos_distances)
                    hard_positives[i] = batch_embeddings[pos_mask][hard_positive_idx]
                    neg_mask = batch_labels != anchor_label[i]
                    neg_distances = distances[neg_mask]
                    hard_negative_idx = torch.argmin(neg_distances)
                    hard_negatives[i] = batch_embeddings[neg_mask][hard_negative_idx]
                loss = criterion(anchor_out, hard_positives, hard_negatives)
            else:  # random
                loss = criterion(anchor_out, positive_out, negative_out)
        elif loss_type == "proxy_nca":
            embeddings = torch.cat([anchor_out, positive_out, negative_out], dim=0)
            labels = torch.cat([anchor_label, anchor_label, negative_label], dim=0)
            loss = criterion(embeddings, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}: Loss = {loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            anchor, positive, negative, _, _ = batch
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            loss = criterion(anchor_out, positive_out, negative_out)
            running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    return avg_loss


def compute_embeddings(model, dataloader, device):
    model.eval()

    embeddings_list = []
    labels_list = []
    with torch.no_grad():
        for batch in dataloader:
            anchor, _, _, labels, _ = batch
            anchor = anchor.to(device)
            emb = model(anchor)
            embeddings_list.append(emb.cpu())
            labels_list.append(labels)

    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return embeddings, labels


def compute_average_embeddings(train_embeddings, train_labels, num_classes):
    average_embeddings = []
    for cls in range(num_classes):
        cls_emb = train_embeddings[train_labels == cls]
        if cls_emb.size(0) == 0:
            raise ValueError(f"No embeddings for class {cls}")
        avg_emb = cls_emb.mean(dim=0)
        average_embeddings.append(avg_emb)

    return torch.stack(average_embeddings)


def validate_original(model, dataloader, k, device):
    model.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            anchor, _, _, labels, _ = batch
            anchor = anchor.to(device)
            emb = model(anchor)
            embeddings_list.append(emb)
            labels_list.append(labels.to(device))

    embeddings_all = torch.cat(embeddings_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)
    distances = torch.cdist(embeddings_all, embeddings_all, p=2)
    sorted_indices = torch.argsort(distances, dim=1)

    hits = 0
    N = embeddings_all.size(0)
    for i in range(N):
        neighbors = sorted_indices[i, 1 : k + 1]
        if (labels_all[neighbors] == labels_all[i]).any():
            hits += 1

    recall_at_k = hits / N
    return recall_at_k


def validate_with_faiss(train_embeddings, train_labels, val_embeddings, val_labels, k):
    train_embeddings_np = train_embeddings.cpu().numpy()
    val_embeddings_np = val_embeddings.cpu().numpy()
    train_labels_np = train_labels.cpu().numpy()
    val_labels_np = val_labels.cpu().numpy()

    d = train_embeddings_np.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(train_embeddings_np)
    _, indices = index.search(val_embeddings_np, k)

    hits = 0
    N = val_embeddings_np.shape[0]
    for i in range(N):
        neighbors = indices[i, :k]
        neighbor_labels = train_labels_np[neighbors]
        if np.any(neighbor_labels == val_labels_np[i]):
            hits += 1
    recall_at_k = hits / N
    return recall_at_k


def validate_with_average_embeddings(average_embeddings, val_embeddings, val_labels):
    distances = torch.cdist(val_embeddings, average_embeddings)
    pred_classes = torch.argmin(distances, dim=1)
    hits = (pred_classes == val_labels).sum().item()

    N = val_labels.size(0)
    recall_at_k = hits / N

    return recall_at_k


def validate_recall_at_k(model, train_loader, val_loader, device, config):
    k = config["validation"]["k"]
    method = config["validation"]["method"]

    if method == "original":
        return validate_original(model, val_loader, k, device)
    else:
        train_embeddings, train_labels = compute_embeddings(model, train_loader, device)
        val_embeddings, val_labels = compute_embeddings(model, val_loader, device)
        if method == "full":
            return validate_with_faiss(
                train_embeddings, train_labels, val_embeddings, val_labels, k
            )
        elif method == "average":
            num_classes = len(train_loader.dataset.label_to_idx)
            average_embeddings = compute_average_embeddings(
                train_embeddings, train_labels, num_classes
            )
            return validate_with_average_embeddings(
                average_embeddings, val_embeddings, val_labels
            )
        else:
            raise ValueError(f"Unknown validation method: {method}")


def main(config_path):
    # Загрузка конфига
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["training"]["seed"]
    set_seed(seed)

    # Установка устройства
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    # Загрузка данных
    dataset = fo.Dataset.from_images_dir(config["data"]["caltech_dir"])
    print(f"Загружен Caltech256: {len(dataset)} образцов")

    val_df = pd.read_csv(config["data"]["val_csv"])
    val_filenames = set(val_df["filename"].tolist())

    train_samples = []
    val_samples = []
    for sample in tqdm(dataset.iter_samples(autosave=False)):
        filename = sample.filename
        label = int(filename.split("_")[0])
        if filename in val_filenames:
            val_samples.append((sample.filepath, label))
        else:
            train_samples.append((sample.filepath, label))

    print(f"Обучающих сэмплов: {len(train_samples)}")
    print(f"Валидационных сэмплов: {len(val_samples)}")

    all_labels = {label for _, label in (train_samples + val_samples)}
    labels_sorted = sorted(all_labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels_sorted)}

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = TripletFODataset(
        train_samples, transform=transform, label_to_idx=label_to_idx
    )
    val_dataset = TripletFODataset(
        val_samples, transform=transform, label_to_idx=label_to_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    # Инициализация модели
    model = EmbeddingNet(
        backbone_name=config["model"]["backbone"],
        embedding_dim=config["model"]["embedding_dim"],
        pretrained=config["model"]["pretrained"],
    )
    model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=float(config["training"]["learning_rate"])
    )
    criterion = nn.TripletMarginLoss(margin=float(config["training"]["margin"]), p=2)

    # Списки для хранения метрик
    train_losses = []
    val_losses = []
    recalls = []

    # Цикл обучения
    save_path = os.path.join("exps", config["exp_name"])
    for epoch in range(config["training"]["num_epochs"]):
        print(f"\nЭпоха {epoch + 1}/{config['training']['num_epochs']}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, config)
        val_loss = validate(model, val_loader, criterion, device)
        recall_at_k = validate_recall_at_k(
            model, train_loader, val_loader, device, config
        )

        # Сохранение метрик
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        recalls.append(recall_at_k)

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Recall@{config['validation']['k']}: {recall_at_k:.4f}"
        )

        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), f"{save_path}/model_epoch_{epoch + 1}.pth")

    # Построение и сохранение графиков
    epochs = range(1, config["training"]["num_epochs"] + 1)

    # График лоссов
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/loss_plot.png")
    plt.close()

    # График Recall@1
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, recalls, label="Recall@1", marker="o", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Recall@1")
    plt.title("Recall@1 over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/recall_plot.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train embedding model on Caltech256 dataset"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()

    main(args.config)

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree
from prompt import *
from util.utils import *
from time import time


def MF_evaluation(model, config: dict, data, device='cpu'):
    data = data.to(device)  
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    train_edge_index = data.edge_index
    test_edge_index = data.edge_label_index
    train_loader = torch.utils.data.DataLoader(
        range(train_edge_index.size(1)),
        shuffle=True,
        batch_size=config['batch_size'],
    )

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = torch.nn.MSELoss().to(device)
    start_time = time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for index in train_loader:
            edge_batch = train_edge_index[:, index].t()
            src, dst = edge_batch[:, 0], edge_batch[:, 1]
            dst = dst - config['num_users']
            optimizer.zero_grad()
            pred = model(src, dst)
            target = torch.ones_like(pred)
            loss = loss_func(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)
        train_loss = total_loss / train_edge_index.size(1)  # 存疑

        model.eval()
        with torch.no_grad():
            user_emb = model.user_emb.weight
            item_emb = model.item_emb.weight
            precision = recall = total_examples = ndcg = 0
            for start in range(0, config['num_users'], config['batch_size']):
                end = start + config['batch_size']
                # User-item interaction scores
                logits = user_emb[start:end] @ item_emb.t()

                # Exclude training edges
                mask = ((train_edge_index[0] >= start) & (train_edge_index[0] < end))
                logits[train_edge_index[0, mask] - start, train_edge_index[1, mask] -
                       config['num_users']] = float('-inf')

                # Computing ground truth for test set
                ground_truth = torch.zeros_like(logits, dtype=torch.bool)
                mask = ((test_edge_index[0] >= start) & (test_edge_index[0] < end))
                ground_truth[test_edge_index[0, mask] - start, test_edge_index[1, mask] -
                             config['num_users']] = True

                # Count number of positive items for each user in test set
                node_count = degree(test_edge_index[0, mask] - start,
                                    num_nodes=logits.size(0))

                # Get top-k predictions
                topk_index = logits.topk(config['k'], dim=-1).indices
                isin_mat = ground_truth.gather(1, topk_index)
                precision += float((isin_mat.sum(dim=-1) / config['k']).sum())
                recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
                for i in range(logits.shape[0]):
                    if node_count[i] > 0:
                        ndcg += ndcg_at_k(logits[i], ground_truth[i], config['k'])
                total_examples += int((node_count > 0).sum())

            precision = precision / total_examples
            recall = recall / total_examples
            ndcg = ndcg / total_examples

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
              f"HR@{config['k']}: {precision:.4f}, "
              f"Recall@{config['k']}: {recall:.4f}, NDCG@{config['k']}: {ndcg:.4f}")
    end_time = time()
    print(f"Total time: {end_time - start_time:.2f}s")
    return model


def prompt_MF_evaluation(teacher, student, config: dict, retain_data, forget_data, device='cpu'):
    retain_data = retain_data.to(device)  # Convert to homogeneous graph, dtype is Data
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    retain_loader = torch.utils.data.DataLoader(
        range(retain_data.edge_index.size(1)),
        shuffle=True,
        batch_size=config['batch_size'],
    )

    # Resize the forget data
    forget_sample_size = retain_data.edge_index.size(1)
    forget_indices = torch.randperm(forget_data.edge_index.size(1))[:forget_sample_size]
    forget_sampled = forget_data.edge_index[:, forget_indices]
    forget_data = Data(x=config['num_users'] + config['num_books'], edge_index=forget_sampled,
                       edge_label_index=forget_sampled)
    forget_data = forget_data.to(device)

    forget_loader = torch.utils.data.DataLoader(
        range(forget_data.size(1)),
        shuffle=True,
        batch_size=config['batch_size'],
    )

    if config['tuning_type'] == 'simplePrompt':
        prompt = SimplePrompt(config["embedding_dim"]).to(device)
    elif config['tuning_type'] == 'complexPrompt':
        prompt = ComplexPrompt(config["embedding_dim"], config["number_p"]).to(device)
    else:
        raise ValueError("Invalid tuning type")

    teacher.eval()  # Teacher should always be in eval mode
    student.train()  
    optimizer = torch.optim.Adam(prompt.parameters(), lr=learning_rate, weight_decay=weight_decay)

    contrastive_loss = ContrastiveLoss(margin=1.0).to(device)
    start_time = time()
    for epoch in range(epochs):
        prompt.train()  # Prompt needs to be in train mode as well
        total_loss = total_examples = 0
        # Training on forget data
        for (retain_index, forget_index) in zip(retain_loader, forget_loader):
            # Retain
            retain_edge_index = retain_data.edge_index[:, retain_index].t()
            retain_src, retain_dst = retain_edge_index[:, 0], retain_edge_index[:, 1]
            retain_dst = retain_dst - config['num_users']

            # Forget
            forget_edge_index = forget_data.edge_index[:, forget_index].t()
            forget_src, forget_dst = forget_edge_index[:, 0], forget_edge_index[:, 1]
            forget_dst = forget_dst - config['num_users']

            optimizer.zero_grad()
            # Get teacher outputs without gradients
            with torch.no_grad():
                teacher_retain_output = teacher(retain_src, retain_dst)
                teacher_forget_output = teacher(forget_src, forget_dst)
            
            # Get student outputs with gradients
            student_retain_output = student(retain_src, retain_dst, prompt=prompt)
            student_forget_output = student(forget_src, forget_dst, prompt=prompt)
            
            # Ensure outputs are treated as logits
            retain_loss = compute_retain_loss(student_retain_output.float(), teacher_retain_output.float())
            forget_loss = compute_forget_loss(student_forget_output.float(), teacher_forget_output.float())

            if config['Contrastive_loss'] is True:
                # Contrastive loss
                negative_labels = torch.zeros(student_forget_output.size(0), device=device)
                contrastive_loss_forget = contrastive_loss(student_forget_output, teacher_forget_output,
                                                           negative_labels)
                positive_labels = torch.ones(student_forget_output.size(0), device=device)
                contrastive_loss_retain = contrastive_loss(student_forget_output, teacher_forget_output,
                                                           positive_labels)
                loss = (config['alpha'] * retain_loss + config['beta'] * forget_loss +
                        config['gamma'] * (contrastive_loss_forget+contrastive_loss_retain))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * (forget_edge_index.size(1) + retain_edge_index.size(1))
                total_examples += forget_edge_index.size(1) + retain_edge_index.size(1)
            else:
                loss = (config['alpha'] * retain_loss + config['beta'] * forget_loss)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * (forget_edge_index.size(1) + retain_edge_index.size(1))
                total_examples += forget_edge_index.size(1) + retain_edge_index.size(1)
        train_loss = total_loss / total_examples

        teacher.eval()
        student.eval()
        with torch.no_grad():
            user_emb, item_emb = student.get_embedding(prompt=prompt)

            precision = recall = total_examples = ndcg = 0
            for start in range(0, config['num_users'], config['batch_size']):
                end = start + config['batch_size']
                # User-item interaction scores
                logits = user_emb[start:end] @ item_emb.t()

                # Exclude training edges
                mask = ((retain_data.edge_index[0] >= start) & (retain_data.edge_index[0] < end))
                logits[retain_data.edge_index[0, mask] - start, retain_data.edge_index[1, mask] -
                       config['num_users']] = float('-inf')

                # Computing ground truth for test set
                ground_truth = torch.zeros_like(logits, dtype=torch.bool)
                mask = ((retain_data.edge_label_index[0] >= start) & (retain_data.edge_label_index[0] < end))
                ground_truth[retain_data.edge_label_index[0, mask] - start, retain_data.edge_label_index[1, mask] -
                             config['num_users']] = True

                # Count number of positive items for each user in test set
                node_count = degree(retain_data.edge_label_index[0, mask] - start,
                                    num_nodes=logits.size(0))

                # Get top-k predictions
                topk_index = logits.topk(config['k'], dim=-1).indices
                isin_mat = ground_truth.gather(1, topk_index)
                precision += float((isin_mat.sum(dim=-1) / config['k']).sum())
                recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
                for i in range(logits.shape[0]):
                    if node_count[i] > 0:
                        ndcg += ndcg_at_k(logits[i], ground_truth[i], config['k'])
                total_examples += int((node_count > 0).sum())

            precision = precision / total_examples
            recall = recall / total_examples
            ndcg = ndcg / total_examples

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
              f"HR@{config['k']}: {precision:.4f}, "
              f"Recall@{config['k']}: {recall:.4f}, NDCG@{config['k']}: {ndcg:.4f}")
    end_time = time()
    print(f"Running time: {end_time - start_time:.2f}s")
    return student, prompt


def MF_unlearning_evaluation(student, prompt, forget_data, config, device='cpu'):
    student.eval()
    forget_data = forget_data.to(device)
    num_users = config['num_users']
    k = config['k']
    batch_size = config['batch_size']

    with torch.no_grad():
        user_emb, book_emb = student.get_embedding(prompt=prompt)
        precision = recall = total_examples = ndcg = 0
        for start in range(0, num_users, batch_size):
            end = start + batch_size
            logits = user_emb[start:end] @ book_emb.t()

            mask = ((forget_data.edge_index[0] >= start) &
                    (forget_data.edge_index[0] < end))
            forget_edges = forget_data.edge_index[:, mask]
            node_count = degree(forget_data.edge_index[0, mask] - start,
                                num_nodes=logits.size(0))
            topk_index = logits.topk(k, dim=-1).indices

            ground_truth = torch.zeros_like(logits, dtype=torch.bool)
            ground_truth[forget_edges[0] - start, forget_edges[1] - num_users] = True

            isin_mat = ground_truth.gather(1, topk_index)

            precision += float((isin_mat.sum(dim=-1) / k).sum())
            recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
            total_examples += int((node_count > 0).sum())
            for i in range(logits.shape[0]):
                    if node_count[i] > 0:
                        ndcg += ndcg_at_k(logits[i], ground_truth[i], k)

        precision = precision / total_examples
        recall = recall / total_examples
        ndcg = ndcg / total_examples
    print(f"HR@{k}: {precision:.4f}, Recall@{k}: {recall:.4f}, NDCG@{k}: {ndcg:.4f}")

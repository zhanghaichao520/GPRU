from torch_geometric.data import Data
from prompt import *
from util.utils import *
from torch_geometric.utils import degree
from model.LightGCN import *
from time import time


# Normal lightgcn evaluation method (without prompt or teacher student framework)
def lightgcn_evaluation(model, config: dict, data, device='cpu'):
    # Preprocess the data
    num_users = config['num_users']
    num_books = config['num_books']
    k = config['k']
    epochs = config['epochs']
    test_topks = []
    batch_size = config['batch_size']
    start_time = time()
    data = data.to(device)  # numedge 2380730
    mask = data.edge_index[0] < data.edge_index[1]  
    train_edge_label_index = data.edge_index[:, mask]
    train_loader = torch.utils.data.DataLoader(
        range(train_edge_label_index.size(1)),
        shuffle=True,
        batch_size=batch_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(epochs):
        model.train()
        total_loss = total_examples = 0
        for index in train_loader:
            # Sample positive and negative labels.
            pos_edge_label_index = train_edge_label_index[:, index]
            neg_edge_label_index = torch.stack([
                pos_edge_label_index[0],
                torch.randint(num_users, num_users + num_books,
                              (index.numel(),), device=device)
            ], dim=0)
            edge_label_index = torch.cat([
                pos_edge_label_index,
                neg_edge_label_index,
            ], dim=1)

            optimizer.zero_grad()
            pos_rank, neg_rank = model(data.edge_index, edge_label_index, prompt=None).chunk(2)

            loss = recommendation_loss(
                model,
                None,
                pos_rank,
                neg_rank,
                node_id=edge_label_index.unique(),
            )
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pos_rank.numel()
            total_examples += pos_rank.numel()
        loss = total_loss / total_examples

        # Test
        model.eval()
        with torch.no_grad():
            emb = model.get_embedding(data.edge_index, prompt=None)  # Here get the embedding of nodes
            user_emb, book_emb = emb[:num_users], emb[num_users:]

            precision = recall = total_examples = ndcg = 0
            for start in range(0, num_users, batch_size):
                end = start + batch_size
                # User ratings matrix
                logits = user_emb[start:end] @ book_emb.t()

                # Exclude training edges:
                mask = ((train_edge_label_index[0] >= start) &
                        (train_edge_label_index[0] < end))
                logits[train_edge_label_index[0, mask] - start,
                       train_edge_label_index[1, mask] - num_users] = float('-inf')

                # Computing precision and recall:
                ground_truth = torch.zeros_like(logits, dtype=torch.bool)
                mask = ((data.edge_label_index[0] >= start) &
                        (data.edge_label_index[0] < end))
                ground_truth[data.edge_label_index[0, mask] - start,
                             data.edge_label_index[1, mask] - num_users] = True
                node_count = degree(data.edge_label_index[0, mask] - start,
                                    num_nodes=logits.size(0))  # Number of positive labels

                topk_index = logits.topk(k, dim=-1).indices
                isin_mat = ground_truth.gather(1, topk_index)

                precision += float((isin_mat.sum(dim=-1) / k).sum())
                recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())

                for i in range(logits.shape[0]):
                    if node_count[i] > 0:
                        ndcg += ndcg_at_k(logits[i], ground_truth[i], k)

                total_examples += int((node_count > 0).sum())

            precision = precision / total_examples
            recall = recall / total_examples
            ndcg = ndcg / total_examples
            test_topks.append((precision, recall, ndcg))
            print(f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, HR@{k}: '
                  f'{precision:.4f}, Recall@{k}: {recall:.4f}, NDCG@{k}: {ndcg:.4f}')
    end_time = time()
    print(f'Total time: {end_time - start_time:.2f}s')
    return model


# Prompt lightgcn evaluation method
def prompt_lightgcn_evaluation(teacher, student, whole_data, retain_data, forget_data, config: dict, device='cpu'):
    # Preprocess the data
    num_users = config['num_users']
    num_books = config['num_books']
    k = config['k']
    epochs = config['epochs']
    retain_data = retain_data.to(device)
    whole_data = whole_data.to(device)

    # Resize the forget data
    forget_sample_size = retain_data.edge_index.size(1)
    repeat_times = (forget_sample_size + forget_data.edge_index.size(1) - 1) // forget_data.edge_index.size(1)
    forget_indices = torch.randperm(forget_data.edge_index.size(1)).repeat(repeat_times)[:forget_sample_size]
    forget_sampled = forget_data.edge_index[:, forget_indices]
    forget_data = Data(x=num_users + num_books, edge_index=forget_sampled, edge_label_index=[[]])
    forget_data = forget_data.to(device)
    test_topks = []
    batch_size = config['batch_size']

    # Prepare data loaders
    retain_loader = torch.utils.data.DataLoader(
        range(retain_data.edge_index.size(1)),
        shuffle=True,
        batch_size=batch_size,
    )
    forget_loader = torch.utils.data.DataLoader(
        range(forget_data.size(1)),
        shuffle=True,
        batch_size=batch_size,
    )

    # Initialize the prompt and optimizer
    # We only train the prompt parameters, not the student model parameters
    if config['tuning_type'] == 'simplePrompt':
        prompt = SimplePrompt(config["embedding_dim"]).to(device)
    elif config['tuning_type'] == 'complexPrompt':
        prompt = ComplexPrompt(config["embedding_dim"], config["number_p"]).to(device)
    else:
        raise AssertionError("Invalid tuning type, please choose between 'simplePrompt' and 'complexPrompt'.")
    
    # Set student and teacher models to eval mode since we don't use any optim tech like droupout
    student.eval()
    teacher.eval()
    # Only optimize prompt parameters
    optimizer = torch.optim.AdamW(prompt.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    contrastive_loss = ContrastiveLoss(margin=1.0).to(device)
    start_time = time()
    for epoch in range(epochs):
        # Keep student and teacher in eval mode, only prompt in train mode
        prompt.train()
        total_loss = total_examples = 0

        for (retain_index, forget_index) in zip(retain_loader, forget_loader):
            retain_edge_index = retain_data.edge_index[:, retain_index]
            forget_edge_index = forget_data.edge_index[:, forget_index]
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_retain_output = teacher(whole_data.edge_index, retain_edge_index)
            student_retain_output = student(whole_data.edge_index, retain_edge_index, prompt=prompt)
            retain_loss = compute_retain_loss(student_retain_output, teacher_retain_output)

            with torch.no_grad():
                teacher_forget_output = teacher(whole_data.edge_index, forget_edge_index)
            student_forget_output = student(whole_data.edge_index, forget_edge_index, prompt=prompt)
            forget_loss = compute_forget_loss(student_forget_output, teacher_forget_output)

            if config['Contrastive_loss'] is True:
                # Contrastive loss
                negative_labels = torch.zeros(student_forget_output.size(0), device=device)  # need to fix
                contrastive_loss_forget = contrastive_loss(student_forget_output, teacher_forget_output,
                                                           negative_labels)
                
                positive_labels = torch.ones(student_retain_output.size(0), device=device)  # need to fix
                contrastive_loss_retain = contrastive_loss(student_retain_output, teacher_retain_output,
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
        loss = total_loss / total_examples

        # Evaluation
        # student already in eval mode
        prompt.eval()
        with torch.no_grad():
            emb = student.get_embedding(whole_data.edge_index, prompt=prompt)
            user_emb, book_emb = emb[:num_users], emb[num_users:]
            precision = recall = total_examples = ndcg = 0
            for start in range(0, num_users, batch_size):
                end = start + batch_size
                logits = user_emb[start:end] @ book_emb.t()
                # Exclude training edges:
                mask = ((retain_data.edge_index[0] >= start) &
                        (retain_data.edge_index[0] < end))
                logits[retain_data.edge_index[0, mask] - start,
                       retain_data.edge_index[1, mask] - num_users] = float('-inf')
                # Computing precision and recall:
                ground_truth = torch.zeros_like(logits, dtype=torch.bool)
                mask = ((retain_data.edge_label_index[0] >= start) &
                        (retain_data.edge_label_index[0] < end))
                ground_truth[retain_data.edge_label_index[0, mask] - start,
                             retain_data.edge_label_index[1, mask] - num_users] = True
                node_count = degree(retain_data.edge_label_index[0, mask] - start,
                                    num_nodes=logits.size(0))
                topk_index = logits.topk(k, dim=-1).indices
                isin_mat = ground_truth.gather(1, topk_index)
                precision += float((isin_mat.sum(dim=-1) / k).sum())
                recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
                total_examples += int((node_count > 0).sum())
                for i in range(logits.shape[0]):
                    if node_count[i] > 0:
                        ndcg += ndcg_at_k(logits[i], ground_truth[i], k)
            precision = precision / total_examples
            recall = recall / total_examples
            test_topks.append((precision, recall))
            ndcg = ndcg / total_examples
            print(f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, HR@{k}: '
                  f'{precision:.4f}, Recall@{k}: {recall:.4f}, NDCG@{k}: {ndcg:.4f}')
    end_time = time()
    print(f'Total time: {end_time - start_time:.2f}s')
    return student, prompt


def prompt_lightgcn_unlearning_evaluation(student, prompt, whole_data, forget_data, config, device='cpu'):
    student.eval()
    prompt.eval()
    forget_data = forget_data.to(device)
    whole_data = whole_data.to(device)
    num_users = config['num_users']
    k = config['k']
    batch_size = config['batch_size']

    with torch.no_grad():
        emb = student.get_embedding(whole_data.edge_index, prompt=prompt)
        user_emb, book_emb = emb[:num_users], emb[num_users:]
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
            # Exclude training edges
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
        print(f'HR@{k}: {precision:.4f}, Recall@{k}: {recall:.4f}, NDCG@{k}: {ndcg:.4f}')

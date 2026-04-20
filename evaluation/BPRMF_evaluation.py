from torch_geometric.data import Data
from prompt import *
from util.utils import *
from torch_geometric.utils import degree
from model.LightGCN import *
from time import time


def BPRMF_eva(model, config: dict, data, device='cpu'):
    # Preprocess the data
    num_users = config['num_users']
    num_books = config['num_books']
    k = config['k']
    epochs = config['epochs']
    data = data.to(device)  # Convert to homogeneous graph, dtype is Data
    epoch_tracks = []
    test_topks = []
    batch_size = config['batch_size']

    mask = data.edge_index[0] < data.edge_index[1]
    train_edge_label_index = data.edge_index[:, mask]
    train_loader = torch.utils.data.DataLoader(
        range(train_edge_label_index.size(1)),
        shuffle=True,
        batch_size=batch_size,
    )

    prompt = None
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    start_time = time()
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
            pos_rank, neg_rank = model(data.edge_index, edge_label_index, prompt=prompt).chunk(2)

            loss = recommendation_loss(
                model,
                prompt,
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
            epoch_tracks.append(epoch)
            emb = model.get_embedding(data.edge_index, prompt=prompt)
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
                                    num_nodes=logits.size(0))

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
            print(f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Precision@{k}: '
                  f'{precision:.4f}, Recall@{k}: {recall:.4f}, NDCG@{k}: {ndcg:.4f}')
    end_time = time()
    print(f'Total time: {end_time - start_time}')
    return model, epoch_tracks, test_topks


def prompt_BPRMF_unlearning_eva(teacher, student, retain_data, forget_data, config: dict, device='cpu'):
    # Preprocess the data
    num_users = config['num_users']
    num_books = config['num_books']
    k = config['k']
    epochs = config['epochs']
    retain_data = retain_data.to(device)

    # Resize the forget data
    forget_sample_size = retain_data.edge_index.size(1)
    forget_indices = torch.randperm(forget_data.edge_index.size(1))[:forget_sample_size]
    forget_sampled = forget_data.edge_index[:, forget_indices]
    forget_data = Data(x=num_users + num_books, edge_index=forget_sampled, edge_label_index=forget_sampled)
    forget_data = forget_data.to(device)
    epoch_tracks = []
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

    if config['tuning_type'] == 'gpf':
        prompt = SimplePrompt(config["embedding_dim"]).to(device)
        optimizer = torch.optim.Adam(prompt.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    elif config['tuning_type'] == 'gpf-plus':
        prompt = GPFplusAtt(config["embedding_dim"], config["number_p"]).to(device)
        optimizer = torch.optim.Adam(prompt.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    else:
        raise AssertionError("Invalid tuning type")

    contrastive_loss = ContrastiveLoss(margin=1.0).to(device)
    start_time = time()
    for epoch in range(epochs):
        student.train()
        teacher.eval()
        total_loss = total_examples = 0

        # Training on forget data
        for (retain_index, forget_index) in zip(retain_loader, forget_loader):
            retain_edge_index = retain_data.edge_index[:, retain_index]
            forget_edge_index = forget_data.edge_index[:, forget_index]
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_retain_output = teacher(retain_data.edge_index, retain_edge_index)
            student_retain_output = student(retain_data.edge_index, retain_edge_index, prompt=prompt)

            retain_loss = compute_retain_loss(student_retain_output, teacher_retain_output)
            with torch.no_grad():
                teacher_forget_output = teacher(retain_data.edge_index, forget_edge_index)
            student_forget_output = student(retain_data.edge_index, forget_edge_index, prompt=prompt)
            forget_loss = compute_forget_loss(student_forget_output, teacher_forget_output)

            if config['Contrastive_loss'] is True and config['regularization'] is True:
                # Contrastive loss
                negative_labels = torch.zeros(student_forget_output.size(0), device=device)
                contrastive_loss_forget = contrastive_loss(student_forget_output, teacher_forget_output,
                                                           negative_labels)
                # Total loss
                params = torch.cat([x.view(-1) for x in prompt.parameters()])
                prompt_L2 = torch.norm(params, p=2)
                loss = (config['alpha'] * retain_loss + config['beta'] * forget_loss +
                        config['gamma'] * contrastive_loss_forget + config['delta'] * prompt_L2)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * (forget_edge_index.size(1) + retain_edge_index.size(1))
                total_examples += forget_edge_index.size(1) + retain_edge_index.size(1)
            elif config['Contrastive_loss'] is True and config['regularization'] is False:
                # Contrastive loss
                negative_labels = torch.zeros(student_forget_output.size(0), device=device)
                contrastive_loss_forget = contrastive_loss(student_forget_output, teacher_forget_output,
                                                           negative_labels)
                loss = (config['alpha'] * retain_loss + config['beta'] * forget_loss +
                        config['gamma'] * contrastive_loss_forget)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * (forget_edge_index.size(1) + retain_edge_index.size(1))
                total_examples += forget_edge_index.size(1) + retain_edge_index.size(1)
            elif config['Contrastive_loss'] is False and config['regularization'] is True:
                # Total loss
                params = torch.cat([x.view(-1) for x in prompt.parameters()])
                prompt_L2 = torch.norm(params, p=2)
                loss = (config['alpha'] * retain_loss + config['beta'] * forget_loss + config['delta'] * prompt_L2)
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
        student.eval()
        teacher.eval()
        with torch.no_grad():
            epoch_tracks.append(epoch)
            emb = student.get_embedding(retain_data.edge_index, prompt=prompt)
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
                # for i in range(isin_mat.shape[0]):
                #     if node_count[i] > 0:
                #         ndcg += ndcg_at_k(isin_mat[i].float(), k)
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
    print(f'Running time: {end_time - start_time:.2f}s')
    return student, prompt, epoch_tracks, test_topks


def BPRMF_forget_data_eva(student, prompt, forget_data, num_users, k, batch_size, device='cpu'):
    student.eval()
    forget_data = forget_data.to(device)

    with torch.no_grad():
        emb = student.get_embedding(forget_data.edge_index, prompt=prompt)
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
            # for i in range(isin_mat.shape[0]):
            #         if node_count[i] > 0:
            #             ndcg += ndcg_at_k(isin_mat[i].float(), k)
            for i in range(logits.shape[0]):
                    if node_count[i] > 0:
                        ndcg += ndcg_at_k(logits[i], ground_truth[i], k)
        precision = precision / total_examples
        recall = recall / total_examples
        ndcg = ndcg / total_examples
        print(f'HR@{k}: {precision:.4f}, Recall@{k}: {recall:.4f}, NDCG@{k}: {ndcg:.4f}')

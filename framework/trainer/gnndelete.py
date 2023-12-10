import os
import time
import wandb
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling, k_hop_subgraph
from torch_geometric.loader import GraphSAINTRandomWalkSampler
import numpy as np
import torch.nn.functional as F
from ogb.graphproppred import Evaluator
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score


from .base import Trainer
from ..evaluation import *
from ..utils import *


def BoundedKLD(logits, truth):
    # print('aaaaaaaaa', truth.shape, truth)
    return 1 - torch.exp(-F.kl_div(F.log_softmax(logits, -1), truth.softmax(-1), None, None, 'batchmean'))

def CosineDistance(logits, truth):
    if len(logits.shape) == 1:
        return 1 - F.cosine_similarity(logits.view(1, -1), truth.view(1, -1))
    else:
        return 1 - F.cosine_similarity(logits, truth)

def get_loss_fct(name):
    if name == 'kld':
        loss_fct = BoundedKLD
    elif name == 'mse':
        loss_fct = nn.MSELoss()
    elif name == 'cosine':
        loss_fct = CosineDistance
    else:
        raise NotImplementedError

    return loss_fct
    
class GNNDeleteTrainer(Trainer):

    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        if 'ogbl' in self.args.dataset:
            return self.train_minibatch(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

        else:
            return self.train_fullbatch(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

    def compute_loss(self, model, data, random_loss_fct, compute_random_on, random_layer, local_loss_fct, compute_local_on, local_layer, 
                     z1=None, z2=None, z1_ori=None, z2_ori=None, logits_ori=None, 
                     sdf1_all_pair_without_df_mask=None, sdf2_all_pair_without_df_mask=None):
        
        # Randomness
        loss_r = 0
        if random_layer == '1':
            all_z = [z1]
        elif random_layer == '2':
            all_z = [z2]
        elif random_layer == 'both':
            all_z = [z1, z2]
        else:
            raise NotImplementedError
        
        neg_size = data.df_mask.sum()
        neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=neg_size)

        if compute_random_on == 'edgeprob':       # Compute Randomness on edge probability
            
            for z in all_z:
                df_logits = model.decode(z, data.train_pos_edge_index[:, data.df_mask], neg_edge_index)
                loss_r += random_loss_fct(df_logits[:neg_size], df_logits[neg_size:])

        elif compute_random_on == 'nodeemb':
            for z in all_z:
                z_random_source, z_random_target = z[neg_edge_index[0]], z[neg_edge_index[1]]
                z_source, z_target = z[data.train_pos_edge_index[:, data.df_mask][0]], z[data.train_pos_edge_index[:, data.df_mask][1]]
                loss_r += (random_loss_fct(z_source, z_random_source) + random_loss_fct(z_target, z_random_target))

        elif compute_random_on == 'none':
            loss_r = None

        else:
            raise NotImplementedError


        # Local causality
        loss_l = 0
        if local_layer == '1':
            all_z = [z1]
            all_z_ori = [z1_ori]
            all_sdf_lower_triangular_mask = [sdf1_all_pair_without_df_mask]
            all_sdf_node_mask = [data.sdf_node_1hop_mask]
        elif local_layer == '2':
            all_z = [z2]
            all_z_ori = [z2_ori]
            all_sdf_lower_triangular_mask = [sdf2_all_pair_without_df_mask]
            all_sdf_node_mask = [data.sdf_node_2hop_mask]
        elif local_layer == 'both':
            all_z = [z1, z2]
            all_z_ori = [z1_ori, z2_ori]
            all_sdf_lower_triangular_mask = [sdf1_all_pair_without_df_mask, sdf2_all_pair_without_df_mask]
            all_sdf_node_mask = [data.sdf_node_1hop_mask, data.sdf_node_2hop_mask]
        else:
            raise NotImplementedError


        if compute_local_on == 'edgeprob':

            for z_ori, z, sdf_lower_triangular_mask in zip(all_z_ori, all_z, all_sdf_lower_triangular_mask):
                logits = (z @ z.t())[sdf_lower_triangular_mask].sigmoid()
                logits_ori = (z_ori @ z_ori.t())[sdf_lower_triangular_mask].sigmoid()

                loss_l += local_loss_fct(logits, logits_ori)
        
        elif compute_local_on == 'nodeemb':

            for z_ori, z, sdf_node_mask in zip(all_z_ori, all_z, all_sdf_node_mask):
                print(z_ori.shape, z.shape, sdf_node_mask.shape, sdf_node_mask.sum())
                loss_l += local_loss_fct(z_ori[sdf_node_mask], z[sdf_node_mask])

        elif compute_local_on == 'none':
            loss_l = None

        else:
            raise NotImplementedError


        if compute_random_on == 'none':
            loss = loss_l
        elif compute_local_on == 'none':
            loss = loss_r
        else:
            alpha = 0.5
            loss = alpha * loss_r + (1 - alpha) * loss_l

        return loss, loss_r, loss_l
    @torch.no_grad()
    def eval(self, model, data, stage='val', pred_all=False):
        model.eval()
        pos_edge_index = data[f'{stage}_pos_edge_index']
        neg_edge_index = data[f'{stage}_neg_edge_index']

        if self.args.eval_on_cpu:
            model = model.to('cpu')
        
        if hasattr(data, 'dtrain_mask'):
            mask = data.dtrain_mask
        else:
            mask = data.dr_mask
        
        z = model(data.x, data.train_pos_edge_index[:, mask], mask_1hop=data.sdf_node_1hop_mask, mask_2hop=data.sdf_node_2hop_mask)
        logits = model.decode(z, pos_edge_index, neg_edge_index)
        label = self.get_link_labels(pos_edge_index, neg_edge_index)

        # DT AUC AUP
        loss = F.binary_cross_entropy_with_logits(logits, label).cpu().item()
        dt_auc = roc_auc_score(label.cpu(), logits.sigmoid().cpu())
        dt_aup = average_precision_score(label.cpu(), logits.cpu())

        # DF AUC AUP
        if self.args.unlearning_model in ['original']:
            df_logit = []
        else:
            # df_logit = model.decode(z, data.train_pos_edge_index[:, data.df_mask]).sigmoid().tolist()
            df_logit = model.decode(z, data.directed_df_edge_index).sigmoid().tolist()

        if len(df_logit) > 0:
            df_auc = []
            df_aup = []
        
            # Sample pos samples
            if len(self.df_pos_edge) == 0:
                for i in range(500):
                    mask = torch.zeros(data.train_pos_edge_index[:, data.dr_mask].shape[1], dtype=torch.bool)
                    idx = torch.randperm(data.train_pos_edge_index[:, data.dr_mask].shape[1])[:len(df_logit)]
                    mask[idx] = True
                    self.df_pos_edge.append(mask)
            
            # Use cached pos samples
            for mask in self.df_pos_edge:
                pos_logit = model.decode(z, data.train_pos_edge_index[:, data.dr_mask][:, mask]).sigmoid().tolist()
                
                logit = df_logit + pos_logit
                label = [0] * len(df_logit) +  [1] * len(df_logit)
                df_auc.append(roc_auc_score(label, logit))
                df_aup.append(average_precision_score(label, logit))
        
            df_auc = np.mean(df_auc)
            df_aup = np.mean(df_aup)

        else:
            df_auc = np.nan
            df_aup = np.nan

        # Logits for all node pairs
        if pred_all:
            logit_all_pair = (z @ z.t()).cpu()
        else:
            logit_all_pair = None

        log = {
            f'{stage}_loss': loss,
            f'{stage}_dt_auc': dt_auc,
            f'{stage}_dt_aup': dt_aup,
            f'{stage}_df_auc': df_auc,
            f'{stage}_df_aup': df_aup,
            f'{stage}_df_logit_mean': np.mean(df_logit) if len(df_logit) > 0 else np.nan,
            f'{stage}_df_logit_std': np.std(df_logit) if len(df_logit) > 0 else np.nan
        }

        if self.args.eval_on_cpu:
            model = model.to(device)

        return loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, log

    @torch.no_grad()
    def test(self, model, data, model_retrain=None, attack_model_all=None, attack_model_sub=None, ckpt='best'):
        
        if ckpt == 'best':    # Load best ckpt
            ckpt = torch.load(os.path.join(self.args.checkpoint_dir, 'model_final.pt'))
            model.load_state_dict(ckpt['model_state'])

        if 'ogbl' in self.args.dataset:
            pred_all = False
        else:
            pred_all = True
        loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, test_log = self.eval(model, data, 'test', pred_all)

        self.trainer_log['dt_loss'] = loss
        self.trainer_log['dt_auc'] = dt_auc
        self.trainer_log['dt_aup'] = dt_aup
        self.trainer_log['df_logit'] = df_logit
        self.logit_all_pair = logit_all_pair
        self.trainer_log['df_auc'] = df_auc
        self.trainer_log['df_aup'] = df_aup
        self.trainer_log['auc_sum'] = dt_auc + df_auc
        self.trainer_log['aup_sum'] = dt_aup + df_aup
        self.trainer_log['auc_gap'] = abs(dt_auc - df_auc)
        self.trainer_log['aup_gap'] = abs(dt_aup - df_aup)

        # # AUC AUP on Df
        # if len(df_logit) > 0:
        #     auc = []
        #     aup = []

        #     if self.args.eval_on_cpu:
        #         model = model.to('cpu')
            
        #     z = model(data.x, data.train_pos_edge_index[:, data.dtrain_mask])
        #     for i in range(500):
        #         mask = torch.zeros(data.train_pos_edge_index[:, data.dr_mask].shape[1], dtype=torch.bool)
        #         idx = torch.randperm(data.train_pos_edge_index[:, data.dr_mask].shape[1])[:len(df_logit)]
        #         mask[idx] = True
        #         pos_logit = model.decode(z, data.train_pos_edge_index[:, data.dr_mask][:, mask]).sigmoid().tolist()

        #         logit = df_logit + pos_logit
        #         label = [0] * len(df_logit) +  [1] * len(df_logit)
        #         auc.append(roc_auc_score(label, logit))
        #         aup.append(average_precision_score(label, logit))

        #     self.trainer_log['df_auc'] = np.mean(auc)
        #     self.trainer_log['df_aup'] = np.mean(aup)


        if model_retrain is not None:    # Deletion
            self.trainer_log['ve'] = verification_error(model, model_retrain).cpu().item()
            # self.trainer_log['dr_kld'] = output_kldiv(model, model_retrain, data=data).cpu().item()

        # MI Attack after unlearning
        if attack_model_all is not None:
            mi_logit_all_after, mi_sucrate_all_after = member_infer_attack(model, attack_model_all, data)
            self.trainer_log['mi_logit_all_after'] = mi_logit_all_after
            self.trainer_log['mi_sucrate_all_after'] = mi_sucrate_all_after
        if attack_model_sub is not None:
            mi_logit_sub_after, mi_sucrate_sub_after = member_infer_attack(model, attack_model_sub, data)
            self.trainer_log['mi_logit_sub_after'] = mi_logit_sub_after
            self.trainer_log['mi_sucrate_sub_after'] = mi_sucrate_sub_after
            
            self.trainer_log['mi_ratio_all'] = np.mean([i[1] / j[1] for i, j in zip(self.trainer_log['mi_logit_all_after'], self.trainer_log['mi_logit_all_before'])])
            self.trainer_log['mi_ratio_sub'] = np.mean([i[1] / j[1] for i, j in zip(self.trainer_log['mi_logit_sub_after'], self.trainer_log['mi_logit_sub_before'])])
            print(self.trainer_log['mi_ratio_all'], self.trainer_log['mi_ratio_sub'], self.trainer_log['mi_sucrate_all_after'], self.trainer_log['mi_sucrate_sub_after'])
            print(self.trainer_log['df_auc'], self.trainer_log['df_aup'])

        return loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, test_log
    def train_fullbatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        model = model.to('cuda')
        data = data.to('cuda')
        for name, param in model.named_parameters():
              if "deletion" in name:
                param.requires_grad = True
              else:
                param.requires_grad = False

        best_metric = 0

        # '''Model naming convention: "gnndelete_random_mse_edgeprob_1_local_mse_edgeprob_1" '''
        # _, _, random_loss_fct, compute_random_on, random_layer, _, local_loss_fct, compute_local_on, local_layer = self.args.unlearning_model.split('_')
        # random_loss_fct = get_loss_fct(random_loss_fct)
        # local_loss_fct = get_loss_fct(local_loss_fct)

        # neg_size = 10

        # MI Attack before unlearning
        # if attack_model_all is not None:
        #     mi_logit_all_before, mi_sucrate_all_before = member_infer_attack(model, attack_model_all, data)
        #     self.trainer_log['mi_logit_all_before'] = mi_logit_all_before
        #     self.trainer_log['mi_sucrate_all_before'] = mi_sucrate_all_before
        # if attack_model_sub is not None:
        #     mi_logit_sub_before, mi_sucrate_sub_before = member_infer_attack(model, attack_model_sub, data)
        #     self.trainer_log['mi_logit_sub_before'] = mi_logit_sub_before
        #     self.trainer_log['mi_sucrate_sub_before'] = mi_sucrate_sub_before

        # All node paris in S_Df without Df
        ## S_Df 1 hop all pair mask
        sdf1_all_pair_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        idx = torch.combinations(torch.arange(data.num_nodes)[data.sdf_node_1hop_mask.cpu()], with_replacement=True).t()
        sdf1_all_pair_mask[idx[0], idx[1]] = True
        sdf1_all_pair_mask[idx[1], idx[0]] = True

        assert sdf1_all_pair_mask.sum().cpu() == data.sdf_node_1hop_mask.sum().cpu() * data.sdf_node_1hop_mask.sum().cpu()

        ## Remove Df itself
        sdf1_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][0], data.train_pos_edge_index[:, data.df_mask][1]] = False
        sdf1_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][1], data.train_pos_edge_index[:, data.df_mask][0]] = False

        ## S_Df 2 hop all pair mask
        sdf2_all_pair_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        idx = torch.combinations(torch.arange(data.num_nodes)[data.sdf_node_2hop_mask.cpu()], with_replacement=True).t()
        sdf2_all_pair_mask[idx[0], idx[1]] = True
        sdf2_all_pair_mask[idx[1], idx[0]] = True

        assert sdf2_all_pair_mask.sum().cpu() == data.sdf_node_2hop_mask.sum().cpu() * data.sdf_node_2hop_mask.sum().cpu()

        ## Remove Df itself
        sdf2_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][0], data.train_pos_edge_index[:, data.df_mask][1]] = False
        sdf2_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][1], data.train_pos_edge_index[:, data.df_mask][0]] = False

        ## Lower triangular mask
        idx = torch.tril_indices(data.num_nodes, data.num_nodes, -1)
        lower_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        lower_mask[idx[0], idx[1]] = True

        ## The final mask is the intersection
        sdf1_all_pair_without_df_mask = sdf1_all_pair_mask & lower_mask
        sdf2_all_pair_without_df_mask = sdf2_all_pair_mask & lower_mask
        sdf2_all_pair_without_df_mask = sdf2_all_pair_without_df_mask.to("cuda")

        # print(data.sdf_node_2hop_mask.sum())
        # print(sdf_all_pair_mask.nonzero())
        # print(data.train_pos_edge_index[:, data.df_mask][0], data.train_pos_edge_index[:, data.df_mask][1])
        # print('aaaaaaaaaaaa', data.sdf_node_2hop_mask.sum(), a, sdf_all_pair_mask.sum())
        # print('aaaaaaaaaaaa', lower_mask.sum())
        # print('aaaaaaaaaaaa', sdf_all_pair_without_df_mask.sum())
        # print('aaaaaaaaaaaa', data.sdf_node_2hop_mask.sum())
        # assert sdf_all_pair_without_df_mask.sum() == \
        #         data.sdf_node_2hop_mask.sum().cpu() * (data.sdf_node_2hop_mask.sum().cpu() - 1) // 2 - data.df_mask.sum().cpu()

        # Original node embeddings
        # with torch.no_grad():
        #     z1_ori, z2_ori = model.get_original_embeddings(data.x, data.train_pos_edge_index[:, data.dtrain_mask], return_all_emb=True)

        loss_fct = nn.MSELoss()

        for epoch in trange(args.epochs, desc='Unlearning'):
            model.train()

            start_time = time.time()
            z = model(data.x, data.train_pos_edge_index[:, data.sdf_mask], mask_1hop=data.sdf_node_1hop_mask, mask_2hop=data.sdf_node_2hop_mask)
            # z1, z2 = model(data.x, data.train_pos_edge_index[:, data.sdf_mask], return_all_emb=True)
            # print('current deletion weight', model.deletion1.deletion_weight.sum(), model.deletion2.deletion_weight.sum())
            # print('aaaaaa', z[data.sdf_node_2hop_mask].sum())

            # Effectiveness and Randomness
            neg_size = data.df_mask.sum()
            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=neg_size)

            df_logits = model.decode(z, data.train_pos_edge_index[:, data.df_mask], neg_edge_index).sigmoid()
            loss_r = loss_fct(df_logits[:neg_size], df_logits[neg_size:])
            # df_logits = model.decode(
            #     z, 
            #     data.train_pos_edge_index[:, data.df_mask].repeat(1, neg_size), 
            #     neg_edge_index).sigmoid()
            
            # loss_e = loss_fct(df_logits[:neg_size], df_logits[neg_size:])
            # print('df_logits', df_logits)
            # raise

            # Local causality
            if sdf2_all_pair_without_df_mask.sum() != 0:
                logits_sdf = (z @ z.t())[sdf2_all_pair_without_df_mask].sigmoid()
                loss_l = loss_fct(logits_sdf, logits_ori[sdf2_all_pair_without_df_mask].sigmoid())
                # print('local proba', logits_sdf.shape, logits_sdf, logits_ori[sdf2_all_pair_without_df_mask].sigmoid())
            
            else:
                loss_l = torch.tensor(0)
                print('local proba', 0)


            alpha = 0.5
            loss = alpha * loss_r + (1 - alpha) * loss_l

            # loss, loss_r, loss_l = self.compute_loss(
            #     model, data, random_loss_fct, compute_random_on, random_layer, local_loss_fct, compute_local_on, local_layer,
            #     z1, z2, z1_ori, z2_ori, logits_ori, sdf1_all_pair_without_df_mask, sdf2_all_pair_without_df_mask)
            
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            end_time = time.time()
            epoch_time = end_time - start_time

            step_log = {
                'Epoch': epoch,
                'train_loss': loss.item(),
                'loss_r': loss_r.item(),
                'loss_l': loss_l.item(),
                'train_time': epoch_time
            }
            wandb.log(step_log)
            msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in step_log.items()]
            tqdm.write(' | '.join(msg))

            if (epoch + 1) % self.args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')
                valid_log['epoch'] = epoch

                train_log = {
                    'epoch': epoch,
                    'train_loss': loss.item(),
                    'train_loss_l': loss_l.item(),
                    'train_loss_r': loss_r.item(),
                    'train_time': epoch_time,
                }
                
                for log in [train_log, valid_log]:
                    wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))
                    self.trainer_log['log'].append(log)

                if dt_auc + df_auc > best_metric:
                    best_metric = dt_auc + df_auc
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))

        # Save
        ckpt = {
            'model_state': {k: v.to('cpu') for k, v in model.state_dict().items()},
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))


    def train_minibatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        best_metric = 0
        if 'kld' in args.unlearning_model:
            loss_fct = BoundedKLD
        else:
            loss_fct = nn.MSELoss()
        # neg_size = 10

        # MI Attack before unlearning
        if attack_model_all is not None:
            mi_logit_all_before, mi_sucrate_all_before = member_infer_attack(model, attack_model_all, data)
            self.trainer_log['mi_logit_all_before'] = mi_logit_all_before
            self.trainer_log['mi_sucrate_all_before'] = mi_sucrate_all_before
        if attack_model_sub is not None:
            mi_logit_sub_before, mi_sucrate_sub_before = member_infer_attack(model, attack_model_sub, data)
            self.trainer_log['mi_logit_sub_before'] = mi_logit_sub_before
            self.trainer_log['mi_sucrate_sub_before'] = mi_sucrate_sub_before

        z_ori = self.get_embedding(model, data, on_cpu=True)
        z_ori_two_hop = z_ori[data.sdf_node_2hop_mask]

        data.edge_index = data.train_pos_edge_index
        data.node_id = torch.arange(data.x.shape[0])
        loader = GraphSAINTRandomWalkSampler(
            data, batch_size=args.batch_size, walk_length=2, num_steps=args.num_steps,
        )
        for epoch in trange(args.epochs, desc='Unlearning'):
            model.train()

            # print('current deletion weight', model.deletion1.deletion_weight.sum(), model.deletion2.deletion_weight.sum())

            epoch_loss_e = 0
            epoch_loss_l = 0
            epoch_loss = 0
            epoch_time = 0
            for step, batch in enumerate(tqdm(loader, leave=False)):
                start_time = time.time()
                batch = batch.to('cuda')

                train_pos_edge_index = batch.edge_index
                z = model(batch.x, train_pos_edge_index[:, batch.sdf_mask], batch.sdf_node_1hop_mask, batch.sdf_node_2hop_mask)
                z_two_hop = z[batch.sdf_node_2hop_mask]

                # Effectiveness and Randomness
                neg_size = batch.df_mask.sum()
                neg_edge_index = negative_sampling(
                    edge_index=train_pos_edge_index,
                    num_nodes=z.size(0),
                    num_neg_samples=neg_size)

                df_logits = model.decode(z, train_pos_edge_index[:, batch.df_mask], neg_edge_index)
                loss_e = loss_fct(df_logits[:neg_size], df_logits[neg_size:])

                # Local causality
                # Only take the lower triangular part
                
                # mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
                # mask[batch.node_id[batch.sdf_node_2hop_mask]] = True

                # z_ori_subset = z_ori[mask].to('cuda')
                # num_nodes = z_ori_subset.shape[0]
                # idx = torch.tril_indices(num_nodes, num_nodes, -1)
                # local_lower_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
                # local_lower_mask[idx[0], idx[1]] = True
                # logits_ori = (z_ori_subset @ z_ori_subset.t())[local_lower_mask]#.sigmoid()
                # logits = (z_two_hop @ z_two_hop.t())[local_lower_mask]#.sigmoid()

                edge = batch.edge_index[:, batch.sdf_mask]
                lower_mask = edge[0] < edge[1]
                row, col = edge[0][lower_mask], edge[1][lower_mask]

                logits_ori = (z_ori[row] * z_ori[col]).sum(dim=-1).to('cuda')
                logits = (z[row] * z[col]).sum(dim=-1)

                loss_l = loss_fct(logits, logits_ori)


                # print(loss_e, loss_l, z_ori.device, z.device)
                alpha = 0.5
                if 'ablation_random' in self.args.unlearning_model:
                    loss_l = torch.tensor(0)
                    loss = loss_e
                elif 'ablation_locality' in self.args.unlearning_model:
                    loss_e = torch.tensor(0)
                    loss = loss_l
                else:
                    loss = alpha * loss_e + (1 - alpha) * loss_l
                
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

                end_time = time.time()
                epoch_loss_e += loss_e.item()
                epoch_loss_l += loss_l.item()
                epoch_loss += loss.item()
                epoch_time += end_time - start_time

            epoch_loss_e /= step
            epoch_loss_l /= step
            epoch_loss /= step
            epoch_time /= step

            if (epoch+1) % args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')

                train_log = {
                    'epoch': epoch,
                    'train_loss': epoch_loss / step,
                    'train_loss_l': epoch_loss_e / step,
                    'train_loss_e': epoch_loss_l / step,
                    'train_time': epoch_time / step,
                }
                
                for log in [train_log, valid_log]:
                    wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))
                    self.trainer_log['log'].append(log)

                if dt_auc + df_auc > best_metric:
                    best_metric = dt_auc + df_auc
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
                    torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))

        # Save
        ckpt = {
            'model_state': {k: v.to('cpu') for k, v in model.state_dict().items()},
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))

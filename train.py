import os
import argparse
import time
import logging
from pprint import pprint
import numpy as np
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import dgl
from data.knowledge_graph import load_data

from model import GCN_TransE, GCN_DistMult, GCN_ConvE
from model.kge_models import TransE, DistMult, ConvE, HAKE
from model.hmnce_model import HMNCE
from model.drae_baseline import DRAE as DRAEBaseline
from model.rakge_model import RAKGE
from model.literal_models import TransELiteral_gate, KBLN
from utils import process, TrainDataset, TestDataset




class Runner(object):
    def __init__(self, params):
        self.p = params
        self.prj_path = Path(__file__).parent.resolve()
        if self.p.gpu >= 0:
            if not torch.cuda.is_available():
                raise RuntimeError("--gpu was requested but CUDA is not available; use --gpu -1 for CPU")
            self.device = torch.device(f"cuda:{self.p.gpu}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        # The original RAKGE implementation reads the selected device from params.
        self.p.device = self.device
        self.data = load_data(self.p.dataset, data_root=self.p.data_root)
        self.num_ent, self.train_data, self.valid_data, self.test_data, self.num_rels, self.entity_dict, self.relation_dict, self.negative_data= self.data.num_nodes, self.data.train, self.data.valid, self.data.test, self.data.num_rels, self.data.entity_dict, self.data.relation_dict, self.data.negative
        self.triplets = process({'data': self.data, 'train': self.train_data, 'valid': self.valid_data, 'test': self.test_data,'neg': self.negative_data})
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim  # output dim of gnn
        self.data_iter = self.get_data_iter()
        self.g = self.build_graph().to(self.device)
        self.edge_type, self.edge_norm = self.get_edge_dir_and_norm()
        self.model = self.get_model()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
        self.best_val_mrr, self.best_epoch, self.best_val_results = 0., 0., {}
        self.output_root = Path(self.p.output_root).resolve()
        (self.output_root / 'logs').mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_root / 'logs' / f"{self.p.name}.log", encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        pprint(vars(self.p))

    def fit(self):
        save_root = self.output_root / 'checkpoints'
        save_root.mkdir(parents=True, exist_ok=True)
        save_path = Path(self.p.checkpoint).resolve() if self.p.checkpoint else save_root / f"{self.p.name}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        tolerance = 0
        for epoch in range(1, self.p.max_epochs+1):
            start_time = time.time()
            train_loss = self.train()

            self.logger.info(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Cost: {time.time() - start_time:.2f}s")

            if epoch % 10 == 0:
                val_results = self.evaluate('valid')
                if val_results['mrr'] > self.best_val_mrr:
                    tolerance = 0
                    self.best_val_results = val_results
                    self.best_val_mrr = val_results['mrr']
                    self.best_epoch = epoch
                    self.save_model(save_path)
                else:
                    if tolerance < self.p.tolerance:
                        tolerance += 10

                        if tolerance % 25 == 0:
                            self.load_model(save_path)
                            test_results = self.evaluate('test')
                            self.logger.info(
                                f"MRR: Avg {test_results['mrr']:.5}")
                            self.logger.info(
                                f"MR:  Avg {test_results['mr']:.5}")
                            self.logger.info(
                                f"hits_left@1 = {test_results['hits@1']}")
                            self.logger.info(
                                f"hits_left@3 = {test_results['hits@3']}")
                            self.logger.info(
                                f"hits_left@10 = {test_results['hits@10']}")
                    else:
                        break


                self.logger.info(
                    f"Valid MRR: {val_results['mrr']:.5}, Best Valid MRR: {self.best_val_mrr:.5}")

        self.logger.info(vars(self.p))
        self.load_model(save_path)
        self.logger.info(
            f'Loading best model in {self.best_epoch} epoch, Evaluating on Test data')
        start = time.time()
        test_results = self.evaluate('test')
        end = time.time()
        self.logger.info(f"MRR:  {test_results['mrr']:.5}")
        self.logger.info(f"MR:  {test_results['mr']:.5}")
        self.logger.info(f"hits@1 = {test_results['hits@1']}")
        self.logger.info(f"hits@3 = {test_results['hits@3']}")
        self.logger.info(f"hits@10 = {test_results['hits@10']}")
        self.logger.info("time ={}".format(end-start))

    def train(self):
        self.model.train()
        losses = []
        train_iter = self.data_iter['train']
        for step, (triplets, labels, neg, n_label) in enumerate(train_iter):
            triplets, labels, neg, n_label = (
                triplets.to(self.device), labels.to(self.device), neg.to(self.device), n_label.to(self.device)
            )
            subj, rel, obj  = triplets[:, 0], triplets[:, 1], triplets[:, 2]




            # elif self.p.encoder == 'rgcn':
            if self.p.n_layer > 0 :
                pred = self.model(self.g, subj, rel)
                loss = self.model.calc_loss(pred, labels)
            elif self.p.literal:
                if self.p.model in {'hmnce', 'drae', 'rakge'}:
                    loss = self.model(self.g, subj, rel, labels, neg, n_label)
                else:
                    loss = self.model(self.g, subj, rel, labels)
            else:
                loss = self.model(self.g, subj, rel, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        loss = np.mean(losses)
        return loss


    def evaluate(self, split):
        """
        Function to evaluate the model on validation or test set
        :param split: valid or test, set which data-set to evaluate on
        :return: results['mr']: Average of ranks_left and ranks_right
                 results['mrr']: Mean Reciprocal Rank
                 results['hits@k']: Probability of getting the correct prediction in top-k ranks based on predicted score
        """

        def get_combined_results(left):
            results = dict()
            count_left = float(left['count'])
            results['mr'] = round((left['mr']) / (count_left), 5)
            results['mrr'] = round((left['mrr']) / (count_left), 5)
            for k in [1, 3, 10]:
                results[f'hits@{k}'] = round(left[f'hits@{k}'] / count_left, 5)
            return results

        self.model.eval()
        left_result = self.predict(split, 'tail')
        res = get_combined_results(left_result)
        return res

    def predict(self, split='valid', mode='tail'):
        """
        Function to run model evaluation for a given mode
        :param split: valid or test, set which data-set to evaluate on
        :param mode: head or tail
        :return: results['mr']: Sum of ranks
                 results['mrr']: Sum of Reciprocal Rank
                 results['hits@k']: counts of getting the correct prediction in top-k ranks based on predicted score
                 results['count']: number of total predictions
        """
        with torch.no_grad():
            results = dict()
            test_iter = self.data_iter[f'{split}_{mode}']
            for step, (triplets, labels, neg, n_label) in enumerate(test_iter):
                triplets, labels, neg, n_label = (
                    triplets.to(self.device), labels.to(self.device), neg.to(self.device), n_label.to(self.device)
                )
                subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]


                if self.p.n_layer > 0 :
                    pred = self.model(self.g, subj, rel)

                elif self.p.literal:
                    if self.p.model in {'hmnce', 'drae', 'rakge'}:
                        pred = self.model(self.g, subj, rel, labels, neg, n_label)
                    else:
                        pred = self.model(self.g, subj, rel, labels)

                else:
                    pred = self.model(self.g, subj, rel, labels)

                b_range = torch.arange(pred.shape[0], device=self.device)
                # [batch_size, 1], get the predictive score of obj

                target_pred = pred[b_range, obj]
                # label=>-1000000, not label=>pred, filter out other objects with same sub&rel pair

                pred = torch.where(
                    labels.bool(), -torch.ones_like(pred) * 10000000, pred)
                # copy predictive score of obj to new pred
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]  # get the rank of each (sub, rel, obj)

                ranks = ranks.float()
                results['count'] = torch.numel(
                    ranks) + results.get('count', 0)  # number of predictions

                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0)
                results['mrr'] = torch.sum(
                    1.0 / ranks).item() + results.get('mrr', 0)

                for k in [1, 3, 10]:
                    results[f'hits@{k}'] = torch.numel(
                        ranks[ranks <= k]) + results.get(f'hits@{k}', 0)

        return results

    def save_model(self, path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.
        :param path: path where the model is saved
        :return:
        """
        state = {
            'model': self.model.state_dict(),
            'best_val': self.best_val_results,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p)
        }

        torch.save(state, path)

    def load_model(self, path):
        """
        Function to load a saved model
        :param path: path where model is loaded
        :return:
        """
        state = torch.load(Path(path), map_location=self.device)
        model_state = state.get('model', state)
        incompatible = self.model.load_state_dict(model_state, strict=False)
        allowed_missing = {'numerical_literals', 'literal_mask'}
        real_missing = [key for key in incompatible.missing_keys if key not in allowed_missing]
        if real_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint is incompatible. Missing={real_missing}, unexpected={incompatible.unexpected_keys}"
            )
        self.best_val_results = state.get('best_val', {})
        self.best_val_mrr = self.best_val_results.get('mrr', 0.0)
        self.best_epoch = state.get('best_epoch', 0)
        if 'optimizer' in state and not self.p.eval_only:
            self.optimizer.load_state_dict(state['optimizer'])

    def build_graph(self):
        g = dgl.DGLGraph()
        g.add_nodes(self.num_ent)

        if not self.p.rat:
            g.add_edges(self.train_data[:, 0], self.train_data[:, 2])
            g.add_edges(self.train_data[:, 2], self.train_data[:, 0])
        else:
            if self.p.ss > 0:
                sampleSize = self.p.ss
            else:
                sampleSize = self.num_ent - 1
            g.add_edges(self.train_data[:, 0], np.random.randint(
                low=0, high=sampleSize, size=self.train_data[:, 2].shape))
            g.add_edges(self.train_data[:, 2], np.random.randint(
                low=0, high=sampleSize, size=self.train_data[:, 0].shape))
        return g

    def get_data_iter(self):
        """
        get data loader for train, valid and test section
        :return: dict
        """
        def get_data_loader(dataset_class, split):
            return DataLoader(
                dataset_class(self.triplets[split], self.num_ent, self.p),
                batch_size=self.p.batch_size,
                shuffle=True,
                num_workers=self.p.num_workers,
                drop_last=False
            )
        return {
            'train': get_data_loader(TrainDataset, 'train'),

            'valid_tail': get_data_loader(TestDataset, 'valid_tail'),

            'test_tail': get_data_loader(TestDataset, 'test_tail')
        }
    def get_edge_dir_and_norm(self):
        """
        :return: edge_type: indicates type of each edge: [E]
        """
        in_deg = self.g.in_degrees(range(self.g.number_of_nodes())).float()
        norm = in_deg ** -0.5
        norm[torch.isinf(norm).bool()] = 0
        self.g.ndata['xxx'] = norm
        self.g.apply_edges(
            lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
        norm = self.g.edata.pop('xxx').squeeze().to(self.device)
        edge_type = torch.tensor(np.concatenate(
            [self.train_data[:, 1], self.train_data[:, 1] + self.num_rels]), device=self.device)
        return edge_type, norm

    def get_model(self):
        if self.p.n_layer > 0:
            if self.p.score_func.lower() == 'transe':
                model = GCN_TransE(num_ent=self.num_ent, num_rel=self.num_rels, num_base=self.p.num_bases,
                                   init_dim=self.p.init_dim, gcn_dim=self.p.gcn_dim, embed_dim=self.p.embed_dim,
                                   n_layer=self.p.n_layer, edge_type=self.edge_type, edge_norm=self.edge_norm,
                                   bias=self.p.bias, gcn_drop=self.p.gcn_drop, opn=self.p.opn,
                                   hid_drop=self.p.hid_drop, gamma=self.p.gamma, wni=self.p.wni, wsi=self.p.wsi, encoder=self.p.encoder, use_bn=(not self.p.nobn), ltr=(not self.p.noltr))
            elif self.p.score_func.lower() == 'distmult':
                model = GCN_DistMult(num_ent=self.num_ent, num_rel=self.num_rels, num_base=self.p.num_bases,
                                     init_dim=self.p.init_dim, gcn_dim=self.p.gcn_dim, embed_dim=self.p.embed_dim,
                                     n_layer=self.p.n_layer, edge_type=self.edge_type, edge_norm=self.edge_norm,
                                     bias=self.p.bias, gcn_drop=self.p.gcn_drop, opn=self.p.opn,
                                     hid_drop=self.p.hid_drop, wni=self.p.wni, wsi=self.p.wsi, encoder=self.p.encoder, use_bn=(not self.p.nobn), ltr=(not self.p.noltr))
            elif self.p.score_func.lower() == 'conve':
                model = GCN_ConvE(num_ent=self.num_ent, num_rel=self.num_rels, num_base=self.p.num_bases,
                                  init_dim=self.p.init_dim, gcn_dim=self.p.gcn_dim, embed_dim=self.p.embed_dim,
                                  n_layer=self.p.n_layer, edge_type=self.edge_type, edge_norm=self.edge_norm,
                                  bias=self.p.bias, gcn_drop=self.p.gcn_drop, opn=self.p.opn,
                                  hid_drop=self.p.hid_drop, input_drop=self.p.input_drop,
                                  conve_hid_drop=self.p.conve_hid_drop, feat_drop=self.p.feat_drop,
                                  num_filt=self.p.num_filt, ker_sz=self.p.ker_sz, k_h=self.p.k_h, k_w=self.p.k_w, wni=self.p.wni, wsi=self.p.wsi, encoder=self.p.encoder, use_bn=(not self.p.nobn), ltr=(not self.p.noltr))

            else:
                raise KeyError(
                    f'score function {self.p.score_func} not recognized.')

        elif self.p.literal:
            dataset_dir = Path(self.data.dir)
            literal_path = dataset_dir / 'literals' / 'numerical_literals.npy'
            raw_literals = np.load(literal_path, allow_pickle=True).astype('float32')

            # Preserve the explicit observation mask before normalization.  This
            # avoids confusing a valid normalized zero with a missing value.
            observed_mask = (
                np.random.uniform(0, 1, size=raw_literals.shape) <= self.p.numeric_density
            ).astype('float32')
            max_lit, min_lit = np.max(raw_literals, axis=0), np.min(raw_literals, axis=0)
            numerical_literals = (raw_literals - min_lit) / (max_lit - min_lit + 1e-8)
            numerical_literals = numerical_literals * observed_mask

            if self.p.ft == 'binary':
                numerical_literals = np.float32(np.where(numerical_literals > 0, 1, 0))

            if self.p.model == 'hmnce':
                model = HMNCE(
                    self.num_ent,
                    self.num_rels,
                    numerical_literals,
                    params=self.p,
                    observed_mask=observed_mask,
                )
            elif self.p.model == 'drae':
                model = DRAEBaseline(self.num_ent, self.num_rels, numerical_literals, params=self.p)
            elif self.p.model == 'rakge':
                model = RAKGE(self.num_ent, self.num_rels, numerical_literals, params=self.p)
            elif self.p.model == 'kbln':
                train_path = dataset_dir / 'train.npy'
                if not train_path.exists():
                    raise FileNotFoundError(f"KBLN requires {train_path}")
                x_train = np.load(train_path)
                h = x_train[:, 0].astype('int')
                t = x_train[:, 2].astype('int')
                differences = raw_literals[h, :] - raw_literals[t, :]
                c = np.mean(differences, axis=0).astype('float32')
                var = np.var(differences, axis=0) + 1e-6
                model = KBLN(self.num_ent, self.num_rels, numerical_literals, c, var, params=self.p)
            elif self.p.model == 'literal_gate':
                model = TransELiteral_gate(
                    self.num_ent, self.num_rels, numerical_literals, observed_mask, params=self.p
                )
            else:
                raise ValueError(f'Unknown literal model: {self.p.model}')

        else:
            score_name = self.p.model if self.p.model in {'transe', 'distmult', 'conve', 'hake'} else self.p.score_func.lower()
            if score_name == 'transe':
                model = TransE(self.num_ent, self.num_rels, params=self.p)
            elif score_name == 'distmult':
                model = DistMult(self.num_ent, self.num_rels, params=self.p)
            elif score_name == 'conve':
                model = ConvE(self.num_ent, self.num_rels, params=self.p)
            elif score_name == 'hake':
                model = HAKE(self.num_ent, self.num_rels, params=self.p)

            else:
                raise NotImplementedError

        return model.to(self.device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HMNCE training and evaluation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', default='hmnce',
                        choices=['hmnce', 'drae', 'rakge', 'kbln', 'literal_gate', 'transe', 'distmult', 'conve', 'hake'],
                        help='Model to train. DRAE and RAKGE are numerical baselines.')
    parser.add_argument('--name', default=None,
                        help='Run name for logs and checkpoints; defaults to <model>_<dataset>')
    parser.add_argument('--data', dest='dataset', default='credit',
                        help='Dataset to use, default: credit')
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parent / 'datasets',
                        help='Directory containing dataset folders')
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parent,
                        help='Root directory for logs and checkpoints')
    parser.add_argument('--checkpoint', type=Path,
                        help='Explicit checkpoint path for restore/evaluation or saving')
    parser.add_argument('--eval-only', action='store_true',
                        help='Load --checkpoint and evaluate the test split without training')
    parser.add_argument('--score_func', dest='score_func', default='conve',
                        help='Score Function for Link prediction')
    parser.add_argument('--opn', dest='opn', default='corr',
                        help='Composition Operation to be used in CompGCN')
    parser.add_argument('--batch', dest='batch_size',
                        default=256, type=int, help='Batch size')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--epoch', dest='max_epochs',
                        type=int, default=1000, help='Number of epochs')
    parser.add_argument('--l2', type=float, default=0.0,
                        help='L2 Regularization for Optimizer')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Starting Learning Rate')
    parser.add_argument('--lbl_smooth', dest='lbl_smooth',
                        type=float, default=0.0, help='Label Smoothing')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of processes to construct batches')
    parser.add_argument('--seed', dest='seed', default=12345,
                        type=int, help='Seed for randomization')
    parser.add_argument('--restore', dest='restore', action='store_true',
                        help='Restore from the previously saved model')
    parser.add_argument('--bias', dest='bias', action='store_true',
                        help='Whether to use bias in the model')
    parser.add_argument('--num_bases', dest='num_bases', default=-1, type=int,
                        help='Number of basis relation vectors to use')
    parser.add_argument('--init_dim', dest='init_dim', default=100, type=int,
                        help='Initial dimension size for entities and relations')
    parser.add_argument('--gcn_dim', dest='gcn_dim', default=200,
                        type=int, help='Number of hidden units in GCN')
    parser.add_argument('--embed_dim', dest='embed_dim', default=None, type=int,
                        help='Embedding dimension to give as input to score function')
    parser.add_argument('--n_layer', dest='n_layer', default=0,
                        type=int, help='Number of GCN Layers to use')
    parser.add_argument('--gcn_drop', dest='gcn_drop', default=0.1,
                        type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('--hid_drop', dest='hid_drop',
                        default=0.7, type=float, help='Dropout after GCN')

    parser.add_argument('--gamma', dest='gamma', default=9.0,
                        type=float, help='TransE: Gamma to use')

    # ConvE specific hyperparameters
    parser.add_argument('--conve_hid_drop', dest='conve_hid_drop', default=0.3, type=float,
                        help='ConvE: Hidden dropout')
    parser.add_argument('--feat_drop', dest='feat_drop',
                        default=0.2, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('--input_drop', dest='input_drop', default=0.2,
                        type=float, help='ConvE: Stacked Input Dropout')
    parser.add_argument('--k_w', dest='k_w', default=20,
                        type=int, help='ConvE: k_w')
    parser.add_argument('--k_h', dest='k_h', default=10,
                        type=int, help='ConvE: k_h')
    parser.add_argument('--num_filt', dest='num_filt', default=200, type=int,
                        help='ConvE: Number of filters in convolution')
    parser.add_argument('--ker_sz', dest='ker_sz', default=7,
                        type=int, help='ConvE: Kernel size to use')


    # HAKE specific hyperparameters
    parser.add_argument('--modulus_weight', dest='modulus_weight', default=1.0,
                        type=float, help='HAKE: modulus weight to use')
    parser.add_argument('--phase_weight', dest='phase_weight', default=3.0,
                        type=float, help='HAKE: phase_weight to use')

    parser.add_argument('--rat', action='store_true',
                        default=False, help='random adacency tensors')
    parser.add_argument('--wni', action='store_true',
                        default=False, help='without neighbor information')
    parser.add_argument('--wsi', action='store_true',
                        default=False, help='without self-loop information')
    parser.add_argument('--ss', dest='ss', default=-1,
                        type=int, help='sample size (sample neighbors)')
    parser.add_argument('--nobn', action='store_true',
                        default=False, help='no use of batch normalization in aggregation')
    parser.add_argument('--noltr', action='store_true',
                        default=False, help='no use of linear transformations for relation embeddings')

    parser.add_argument('--encoder', dest='encoder',
                        default='compgcn', type=str, help='which encoder to use')

    # for KGE models
    parser.add_argument('--x_ops', dest='x_ops', default="")
    parser.add_argument('--r_ops', dest='r_ops', default="")

    # for literal models
    parser.add_argument('--literal', action='store_true', default=False)
    parser.add_argument('--tolerance', default=100, type=int)

    # for HMNCE, DRAE, and RAKGE
    parser.add_argument('--att_dim', dest='att_dim', default=200, type=int)
    parser.add_argument('--head_num', dest='head_num', default=5, type=int)
    parser.add_argument('--drop', dest='drop',
                        default=0.7, type=float, help='Dropout for HMNCE/DRAE/RAKGE')
    parser.add_argument('--num_mixture', dest='num_mixture',
                        default=5, type=int, help='number of mixtures')
    ## order score
    parser.add_argument('--order', dest='order', default=0.25, type=float)

    ## contrastive learning
    parser.add_argument('--scale', dest='scale', default=0.25, type=float,
                        help='Coefficient of the HMNCE contrastive loss')

    #4.4.3
    parser.add_argument('--numeric_density', dest='numeric_density', default=0.8, type=float, help='For the real-world setting, we dropped numeric values')

    #4.4.4
    parser.add_argument('--ft', default='numeric', help='binary or numeric')

    parser.add_argument('--top_k', dest='top_k', default=5, type=int,
                        help='Top-k ordinal hard samples for contrastive learning')
    parser.add_argument('--tau', dest='tau', default=0.5, type=float,
                        help='Temperature coefficient for InfoNCE loss')
    parser.add_argument('--disable_me', action='store_true',
                        help='Disable missing-aware numeric encoding for ablation')
    parser.add_argument('--disable_hcl', action='store_true',
                        help='Disable top-k hard contrastive learning for ablation')
    parser.add_argument('--disable_hsm', action='store_true',
                        help='Replace score-based hard mining with random-k sampling for ablation')
    parser.add_argument('--disable_hsa', action='store_true',
                        help='Disable top-k aggregation and use the single hardest samples for ablation')
    parser.add_argument('--disable_srm', action='store_true',
                        help='Disable stable relation-aware mixture for ablation')
    parser.add_argument('--hmnce-score-func', '--drae_score_func', dest='hmnce_score_func', default='transe',
                        choices=['transe', 'conve', 'complex', 'tucker'],
                        help='Internal structural score used by HMNCE')

    args = parser.parse_args()
    if args.checkpoint is not None and (args.eval_only or args.restore):
        saved_state = torch.load(args.checkpoint, map_location='cpu')
        saved_args = saved_state.get('args', {}) if isinstance(saved_state, dict) else {}
        if not isinstance(saved_args, dict):
            saved_args = vars(saved_args)
        structural_keys = {
            'model', 'init_dim', 'att_dim', 'head_num', 'drop', 'num_mixture',
            'gamma', 'order', 'scale', 'top_k', 'numeric_density', 'ft',
            'hmnce_score_func',
        }
        for key in structural_keys:
            if key in saved_args:
                setattr(args, key, saved_args[key])
        if 'hmnce_score_func' not in saved_args and 'drae_score_func' in saved_args:
            args.hmnce_score_func = saved_args['drae_score_func']
        if 'model' not in saved_args and str(saved_args.get('name', '')).lower().startswith('drae'):
            args.model = 'hmnce'
    args.literal = args.literal or args.model in {'hmnce', 'drae', 'rakge', 'kbln', 'literal_gate'}
    if args.name is None:
        args.name = f"{args.model}_{args.dataset}"
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    runner = Runner(args)
    if args.eval_only:
        if args.checkpoint is None:
            parser.error('--eval-only requires --checkpoint')
        runner.load_model(args.checkpoint)
        pprint(runner.evaluate('test'))
    else:
        runner.fit()

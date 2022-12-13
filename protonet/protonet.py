"""Implementation of prototypical networks for Omniglot."""

import argparse
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import tensorboard

import omniglot
# import util

NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 32
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
PRINT_INTERVAL = 10
VAL_INTERVAL = PRINT_INTERVAL * 5
NUM_TEST_TASKS = 600


class ProtoNetNetwork(nn.Module):
    """Container for ProtoNet weights and image-to-latent computation."""

    def __init__(self):
        """Inits ProtoNetNetwork.

        The network consists of four convolutional blocks, each comprising a
        convolution layer, a batch normalization layer, ReLU activation, and 2x2
        max pooling for downsampling. There is an additional flattening
        operation at the end.

        Note that unlike conventional use, batch normalization is always done
        with batch statistics, regardless of whether we are training or
        evaluating. This technically makes meta-learning transductive, as
        opposed to inductive.
        """
        super().__init__()
        
        in_channels = NUM_INPUT_CHANNELS

        # for _ in range(NUM_CONV_LAYERS):
        #     layers.append(
        #         nn.Conv2d(
        #             in_channels,
        #             NUM_HIDDEN_CHANNELS,
        #             (KERNEL_SIZE, KERNEL_SIZE),
        #             padding='same'
        #         )
        #     )
        #     layers.append(nn.BatchNorm2d(NUM_HIDDEN_CHANNELS))
        #     layers.append(nn.ReLU())
        #     layers.append(nn.MaxPool2d(2))
        #     in_channels = NUM_HIDDEN_CHANNELS

        self.conv1 = nn.Conv2d(
            in_channels,
            NUM_HIDDEN_CHANNELS,
            (KERNEL_SIZE, KERNEL_SIZE),
            padding='same'
        )
        torch.nn.init.normal_(self.conv1.weight)
        torch.nn.init.normal_(self.conv1.bias)
        
        # self._layers = nn.Sequential(self.conv1, nn.ReLU(), nn.AvgPool2d(), nn.Flatten())

        self.to(DEVICE)

    # def forward(self, images):
    #     """Computes the latent representation of a batch of images.

    #     Args:
    #         images (Tensor): batch of Omniglot images
    #             shape (num_images, channels, height, width)

    #     Returns:
    #         a Tensor containing a batch of latent representations
    #             shape (num_images, latents)
    #     """
    #     return self._layers(images)

    def forward_(self, images, weight, bias):

        out = torch.nn.functional.conv2d(images, weight, bias, padding='same')
        out = torch.nn.functional.relu(out)
        out = torch.nn.functional.avg_pool2d(out, 2)

        return out

    def constr_(self, images, weight, bias):

        out = torch.nn.functional.conv2d(images, weight, bias, padding='same')
        out = 2*torch.nn.functional.relu(out) - out

        return out


class ProtoNet:
    """Trains and assesses a prototypical network."""

    def __init__(self, learning_rate, log_dir):
        """Inits ProtoNet.

        Args:
            learning_rate (float): learning rate for the Adam optimizer
            log_dir (str): path to logging directory
        """

        self._network = ProtoNetNetwork()
        self._optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=learning_rate
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0

        self.constraints_all = []
        self.distances_query = []
        self.labels_query = []

    def _step(self, task_batch):
        """Computes ProtoNet mean loss (and accuracy) on a batch of tasks.

        Args:
            task_batch (tuple[Tensor, Tensor, Tensor, Tensor]):
                batch of tasks from an Omniglot DataLoader

        Returns:
            a Tensor containing mean ProtoNet loss over the batch
                shape ()
            mean support set accuracy over the batch as a float
            mean query set accuracy over the batch as a float
        """
        loss_batch = []
        accuracy_support_batch = []
        accuracy_query_batch = []
        for i, task in enumerate(task_batch):

            print(f'task: {i}')

            images_support, labels_support, images_query, labels_query = task
            images_support = images_support.to(DEVICE)
            labels_support = labels_support.to(DEVICE)
            images_query = images_query.to(DEVICE)
            labels_query = labels_query.to(DEVICE)

            # ********************************************************
            # ******************* YOUR CODE HERE *********************
            # ********************************************************
            # TODO: finish implementing this method.
            # For a given task, compute the prototypes and the protonet loss.
            # Use F.cross_entropy to compute classification losses.
            # Use util.score to compute accuracies.
            # Make sure to populate loss_batch, accuracy_support_batch, and
            # accuracy_query_batch.

            K = (labels_support==0).sum().item() # K-shot (support)
            K_query = (labels_query==0).sum().item() # K-shot (query)

            N = torch.max(labels_support).item()+1 # N-way
            
            # embeddings_support = self._network.forward(images_support) # (N * K, latents)
            # embeddings_query = self._network.forward(images_query) # (N * K_query, latents)

            # embeddings_support_jac = torch.autograd.functional.jacobian(lambda w, b : self._network.forward_(images_support, w, b), (self._network.conv1.weight, self._network.conv1.bias))
            # embeddings_query_jac = torch.autograd.functional.jacobian(lambda w, b : self._network.forward_(images_query, w, b), (self._network.conv1.weight, self._network.conv1.bias))

            # print("constraints!")

            # support_constr = torch.autograd.functional.jacobian(
            #     lambda w, b : self._network.constr_(images_support, w, b),
            #     (self._network.conv1.weight, self._network.conv1.bias),
            #     strategy='forward-mode'
            # )

            all_images = torch.concat((images_support, images_query))
            all_masks = torch.nn.functional.conv2d(all_images, self._network.conv1.weight, self._network.conv1.bias)
            
            # images_query_masks = torch.nn.functional.conv2d(images_query, self._network.conv1.weight, self._network.conv1.bias)

            # (2*images_support_masks - 1)

            # support_conv = [torch.autograd.functional._jacfwd(
            #     lambda w, b : torch.nn.functional.conv2d(all_images, w, b),
            #     (self._network.conv1.weight[j:j+1], self._network.conv1.bias[j:j+1]),
            #     vectorize=True
            # ) for j in range(NUM_HIDDEN_CHANNELS)]

            all_conv = []

            # lazy generation of of conv2d jacobian wrt weights

            for j in range(NUM_HIDDEN_CHANNELS):

                grad_w, grad_b = torch.autograd.functional._jacfwd(
                    lambda w, b : torch.nn.functional.conv2d(all_images, w, b),
                    (self._network.conv1.weight[j:j+1], self._network.conv1.bias[j:j+1]),
                    vectorize=True
                )

                grad_w = torch.flatten(grad_w, start_dim=4)
                grad_b = torch.flatten(grad_b, start_dim=4)

                all_conv.append(torch.concat((grad_w, grad_b), dim=4))

            all_conv = torch.stack(all_conv, dim=1).squeeze()

            constraint = ((2*all_masks-1).unsqueeze(-1)*all_conv) # num_images * hidden_layers * h * w * weights_per_kernel

            num_images, hidden_layers, h, w, weights_per_kernel = constraint.shape

            self.constraints_all.append(constraint.detach().cpu())

            all_after_relu = all_masks.unsqueeze(-1)*all_conv # num_images * hidden_layers * h * w * weights_per_kernel

            all_before_pooling = all_after_relu.permute(0, 1, 4, 2, 3).reshape(num_images, hidden_layers*weights_per_kernel, h, w)

            all_after_pooling = torch.nn.functional.avg_pool2d(all_before_pooling, 2)

            all_after_pooling = all_after_pooling.permute(0, 2, 3, 1).reshape(num_images, -1, hidden_layers, weights_per_kernel)

            # print(all_after_pooling.shape)
            
            embeddings_support = all_after_pooling[:N*K]
            embeddings_query = all_after_pooling[N*K:]
            
            prototypes = torch.mean(embeddings_support.reshape(N, K, *embeddings_support.shape[1:]), axis=1) # (N, latents)

            distances_query = embeddings_query[:, None, ...] - prototypes
            
            self.distances_query.append(distances_query.detach().cpu())
            self.labels_query.append(labels_query.detach().cpu())

            # print(pos_norm.shape)
            # print(distances_query.shape)

            # exit()

            # print(images_support_masks.shape)
            # print(support_conv[0][0].shape)            
            # exit()

            # support_constr = [torch.autograd.functional._jacfwd(
            #     lambda w, b : self._network.constr_(images_support, w, b),
            #     (self._network.conv1.weight[j:j+1], self._network.conv1.bias[j:j+1]),
            #     vectorize=True
            # ) for j in range(NUM_HIDDEN_CHANNELS)]

            # query_constr = [torch.autograd.functional._jacfwd(
            #     lambda w, b : self._network.constr_(images_query, w, b),
            #     (self._network.conv1.weight[j:j+1], self._network.conv1.bias[j:j+1]),
            #     vectorize=True
            # ) for j in range(NUM_HIDDEN_CHANNELS)]

            # def func(weight, bias):
                
            #     embeddings_support = torch.flatten(self._network.forward_(images_support, weight, bias), start_dim=1)
            #     print(embeddings_support.shape)
            #     embeddings_query = torch.flatten(self._network.forward_(images_query, weight, bias), start_dim=1)

            #     prototypes = torch.mean(embeddings_support.reshape((N, K, -1)), axis=1) # (N, latents)

            #     distances_support = (embeddings_support[:, None, :] - prototypes)
            #     distances_query = (embeddings_query[:, None, :] - prototypes)

            #     return distances_support, distances_query

            # print("norms!")

            # pos_norm, neg_norm = [torch.autograd.functional.jacobian(
            #     func,
            #     (self._network.conv1.weight[j:j+1], self._network.conv1.bias[j:j+1])
            # ) for j in range(NUM_HIDDEN_CHANNELS)]
            
            # print(pos_norm.shape)
            # print(neg_norm.shape)

            # self.constraints.append(support_constr)
            # self.constraints.append(query_constr)
            # self.pos_norms.append(pos_norm)
            # self.neg_norms.append(neg_norm)


    def train(self, dataloader_train, dataloader_val, writer):
        """Train the ProtoNet.

        Consumes dataloader_train to optimize weights of ProtoNetNetwork
        while periodically validating on dataloader_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_train (DataLoader): loader for train tasks
            dataloader_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        print(f'Starting training at iteration {self._start_train_step}.')
        for i_step, task_batch in enumerate(
                dataloader_train,
                start=self._start_train_step
        ):
            # self._optimizer.zero_grad()
            # loss, accuracy_support, accuracy_query = self._step(task_batch)
            # loss.backward()
            # self._optimizer.step()

            self._step(task_batch)
            
            if i_step % PRINT_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    # f'loss: {loss.item():.3f}, '
                    # f'support accuracy: {accuracy_support.item():.3f}, '
                    # f'query accuracy: {accuracy_query.item():.3f}'
                )
                # writer.add_scalar('loss/train', loss.item(), i_step)
                # writer.add_scalar(
                #     'train_accuracy/support',
                #     accuracy_support.item(),
                #     i_step
                # )
                # writer.add_scalar(
                #     'train_accuracy/query',
                #     accuracy_query.item(),
                #     i_step
                # )

            # if i_step % VAL_INTERVAL == 0:
            #     with torch.no_grad():
            #         losses, accuracies_support, accuracies_query = [], [], []
            #         for val_task_batch in dataloader_val:
            #             loss, accuracy_support, accuracy_query = (
            #                 self._step(val_task_batch)
            #             )
            #             losses.append(loss.item())
            #             accuracies_support.append(accuracy_support)
            #             accuracies_query.append(accuracy_query)
            #         loss = np.mean(losses)
            #         accuracy_support = np.mean(accuracies_support)
            #         accuracy_query = np.mean(accuracies_query)
            #     print(
            #         f'Validation: '
            #         f'loss: {loss:.3f}, '
            #         f'support accuracy: {accuracy_support:.3f}, '
            #         f'query accuracy: {accuracy_query:.3f}'
            #     )
            #     writer.add_scalar('loss/val', loss, i_step)
            #     writer.add_scalar(
            #         'val_accuracy/support',
            #         accuracy_support,
            #         i_step
            #     )
            #     writer.add_scalar(
            #         'val_accuracy/query',
            #         accuracy_query,
            #         i_step
            #     )

            # if i_step % SAVE_INTERVAL == 0:
            #     self._save(i_step)

    def test(self, dataloader_test):
        """Evaluate the ProtoNet on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        accuracies = []
        for task_batch in dataloader_test:
            accuracies.append(self._step(task_batch)[2])
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )

    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._network.load_state_dict(state['network_state_dict'])
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves network and optimizer state_dicts as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        torch.save(
            dict(network_state_dict=self._network.state_dict(),
                 optimizer_state_dict=self._optimizer.state_dict()),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')


def main(args):
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/protonet/omniglot.way:{args.num_way}.support:{args.num_support}.query:{args.num_query}.lr:{args.learning_rate}.batch_size:{args.batch_size}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    # writer = tensorboard.SummaryWriter(log_dir=log_dir)
    writer = None

    protonet = ProtoNet(args.learning_rate, log_dir)

    if args.checkpoint_step > -1:
        protonet.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_train = omniglot.get_omniglot_dataloader(
            'train',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            num_training_tasks
        )
        dataloader_val = omniglot.get_omniglot_dataloader(
            'val',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            args.batch_size * 4
        )
        protonet.train(
            dataloader_train,
            dataloader_val,
            writer
        )
        labels_query = torch.stack(protonet.labels_query).numpy()
        constraints_all = torch.stack(protonet.constraints_all).numpy()
        distances_query = torch.stack(protonet.distances_query).numpy()
        weight = protonet._network.conv1.weight.detach().cpu().numpy()
        bias = protonet._network.conv1.bias.detach().cpu().numpy()

        print('saving to disk!')
        np.savez(
            'protonet2.npz',
            labels_query=labels_query,
            constraints_all=constraints_all,
            distances_query=distances_query,
            weight=weight,
            bias=bias
        )
        
    else:
        print(
            f'Testing on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_test = omniglot.get_omniglot_dataloader(
            'test',
            1,
            args.num_way,
            args.num_support,
            args.num_query,
            NUM_TEST_TASKS
        )
        protonet.test(dataloader_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a ProtoNet!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_way', type=int, default=5,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=1,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of query examples per class in a task')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for the network')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=1,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))

    main_args = parser.parse_args()
    main(main_args)

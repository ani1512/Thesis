import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import os
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn
from torchvision import utils as vutils
from PIL import Image

from utils import AverageMeter, get_num_model, show_plot, imshow, save_list_to_file
from custom_loss import PrototypicalLoss

"""
    Trainer encapsulates all the logic necessary for training
    the Siamese Network model.
    All hyperparameters are provided by the user in the
    config file.
"""


class SiameseTrainer:

    def __init__(self, model, data_loader, criterion, optimizer, device, epochs=1, is_pretrained=False):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.is_pretrained = is_pretrained

    def train(self):
        """

        :return model: trained model
        """
        counter = []
        batch_acc = []
        loss_history = []
        iteration_number = 1
        self.model.train()

        # iterate through the epochs
        for epoch in range(self.epochs):

            print('=== Epoch: {} ==='.format(epoch))
            correct = 0
            # iterate over batches
            for i, (img0, img1, label) in enumerate(self.data_loader, 0):

                # move images and labels to gpu
                img0, img1, label = img0.to(self.device), img1.to(self.device), label.to(self.device)

                # zero gradients
                self.optimizer.zero_grad()

                # Pass in the two images into the network and obtain two outputs
                if self.is_pretrained:
                    output1, output2 = self.model.twinforward(img0, img1)
                else:
                    output1, output2 = self.model(img0, img1)

                # Pass in the two outputs and label into the loss function
                loss_contrastive, acc = self.criterion(output1, output2, label)
                batch_acc.append(acc)
                avg_acc = np.mean(batch_acc[:iteration_number])
                print('Avg Train Acc: {}'.format(avg_acc))
                iteration_number += 1

                # Calculate the backpropogation
                loss_contrastive.backward()

                # Optimize
                self.optimizer.step()

                # Every 10 batches print out the loss
                if i % 10 == 0:
                    print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")

                    counter.append(iteration_number)
                    loss_history.append(loss_contrastive.item())

            print('Avg Train Acc: {}'.format(avg_acc))

        show_plot(counter, loss_history)
        return self.model


class SiameseTester:

    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device

    def vis_test(self):
        self.model.eval()
        batch_acc = []

        # grab one image that we are going to test
        with torch.no_grad():
            for i, (x0, x1, label) in enumerate(self.data_loader, 0):
                acc = []
                # Iterate over 64 images and test them with the first image (x0)

                # Concatenate the two images together
                concatenated = torch.cat((x0, x1), 0)

                # send the two images to GPU
                x0, x1 = x0.to(self.device), x1.to(self.device)

                output1, output2 = self.model(x0, x1)

                euclidean_distance = F.pairwise_distance(output1, output2)
                loss_contrastive = (1 - label) * torch.pow(euclidean_distance, 2) + \
                                   label * torch.pow(torch.clamp(2.0 - euclidean_distance, min=0.0), 2)
                for loss_idx in range(len(loss_contrastive)):
                    if loss_contrastive[loss_idx] < 1.0 and label[loss_idx] == 0:
                        acc.append(1)
                    elif loss_contrastive[loss_idx] > 1.0 and label[loss_idx] == 1:
                        acc.append(1)
                    else:
                        print('hit')
                        acc.append(0)
                batch_acc.append(np.mean(acc))
                imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')
            print('Avg Test Acc: {}'.format(np.mean(batch_acc)))




class PrototypicalTrainer:
    """
    Train the model with the prototypical learning algorithm
    """

    def __init__(self, tr_dataloader, model,
                 optimizer, lr_scheduler, device, model_path,
                 iterations, n_support, n_support_val, val_dataloader=None):
        self.tr_dataloader = tr_dataloader
        self.model = model
        self.optim = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.val_dataloader = val_dataloader
        self.path = model_path
        self.iterations = iterations
        self.n_support = n_support
        self.n_support_val = n_support_val

        if self.val_dataloader is None:
            self.best_state = None

        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.best_acc = 0

        self.best_model_path = os.path.join(self.path, 'best_model.pth')
        self.last_model_path = os.path.join(self.path, 'last_model.pth')

    def train(self, epochs):
        """
        train as per prototypical algorithm
        :param epochs:
        :return loss:
        """
        for epoch in range(epochs):
            print('=== Epoch: {} ==='.format(epoch))
            tr_iter = iter(self.tr_dataloader)
            self.model.train()

            for batch in tqdm(tr_iter):
                # zeroing the gradients
                self.optim.zero_grad()

                # getting the x and y
                x, y = batch

                # sending inputs to GPU
                x, y = x.to(self.device), y.to(self.device)

                # getting model output
                model_output = self.model(x)

                criterion = PrototypicalLoss(self.n_support)
                loss, acc = criterion(model_output, y)

                # backpropogate the loss
                loss.backward()

                # run the optimizer
                self.optim.step()

                self.train_loss.append(loss.item())
                self.train_acc.append(acc.item())

            avg_loss = np.mean(self.train_loss[-self.iterations:])
            avg_acc = np.mean(self.train_acc[-self.iterations:])
            print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))

            # step down the learning rate
            self.lr_scheduler.step()

            if self.val_dataloader is None:
                continue

            val_iter = iter(self.val_dataloader)
            self.model.eval()

            for batch in val_iter:
                x, y = batch

                x, y = x.to(self.device), y.to(self.device)
                model_output = self.model(x)

                criterion = PrototypicalLoss(self.n_support_val)
                loss, acc = criterion(model_output, y)

                self.val_acc.append(loss.item())
                self.val_loss.append(acc.item())

            avg_loss = np.mean(self.val_loss[-self.iterations:])
            avg_acc = np.mean(self.val_acc[-self.iterations:])

            postfix = ' (Best)' if avg_acc >= self.best_acc else ' (Best: {})'.format(
                self.best_acc)
            print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
                avg_loss, avg_acc, postfix))

            if avg_acc >= self.best_acc:
                torch.save(self.model.state_dict(), self.best_model_path)
                self.best_acc = avg_acc
                self.best_state = self.model.state_dict()

        torch.save(self.model.state_dict(), self.last_model_path)

        return self.best_state, self.best_acc, self.train_loss, self.train_acc, self.val_loss, self.val_acc

    def test(self, test_dataloader):
        """
        Test the model trained with the prototypical learning algorithm
        """
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        avg_acc = list()
        for epoch in range(10):
            test_iter = iter(test_dataloader)
            for batch in test_iter:
                x, y = batch
                x, y = x.to(device), y.to(device)
                model_output = self.model(x)
                criterion = PrototypicalLoss(self.n_support_val)
                _, acc = criterion(model_output, target=y)
                avg_acc.append(acc.item())
        avg_acc = np.mean(avg_acc)
        print('Test Acc: {}'.format(avg_acc))

        return avg_acc


class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, imsize, is_train, batch_size, epoch, model, device):
        self.is_train = is_train
        self.batch_size = batch_size
        self.device = device
        if self.is_train:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            os.makedirs(self.model_dir)
            os.makedirs(self.image_dir)
            os.makedirs(self.log_dir)

        cudnn.benchmark = True

        self.max_epoch = epoch

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        self.model = model

    def evaluate(self, split_dir, z_dim, test_b_example, it):
        with torch.no_grad():
            # Build and load the generator
            if split_dir == 'test':
                split_dir = 'valid'

            # the path to save generated images
            s_tmp = os.path.join(os.getcwd(), 'generatedImages')
            iteration = it
            save_dir = '%s/iteration%d' % (s_tmp, iteration)

            nz = z_dim
            noise = torch.FloatTensor(self.batch_size, nz)
            if torch.cuda.is_available():
                noise.to(self.device)

            # switch to evaluate mode
            self.model.eval()
            for step, data in enumerate(self.data_loader, 0):
                imgs, t_embeddings, filenames = data
                if torch.cuda.is_available():
                    t_embeddings.to(self.device)
                # print(t_embeddings[:, 0, :], t_embeddings.size(1))

                embedding_dim = t_embeddings.size(1)
                batch_size = imgs[0].size(0)
                noise.data.resize_(batch_size, nz)
                noise.data.normal_(0, 1)

                fake_img_list = []
                for i in range(embedding_dim):
                    fake_imgs, _, _ = self.model(noise, t_embeddings[:, i, :])
                    if test_b_example:
                        # fake_img_list.append(fake_imgs[0].data.cpu())
                        # fake_img_list.append(fake_imgs[1].data.cpu())
                        fake_img_list.append(fake_imgs[2].data.cpu())
                    else:
                        self.save_singleimages(fake_imgs[-1], filenames,
                                               save_dir, split_dir, i, 256)
                        # self.save_singleimages(fake_imgs[-2], filenames,
                        #                        save_dir, split_dir, i, 128)
                        # self.save_singleimages(fake_imgs[-3], filenames,
                        #                        save_dir, split_dir, i, 64)
                    # break
                if test_b_example:
                    # self.save_superimages(fake_img_list, filenames,
                    #                       save_dir, split_dir, 64)
                    # self.save_superimages(fake_img_list, filenames,
                    #                       save_dir, split_dir, 128)
                    self.save_superimages(fake_img_list, filenames,
                                          save_dir, split_dir, 256)

    def save_superimages(self, images_list, filenames,
                         save_dir, split_dir, imsize):
        batch_size = images_list[0].size(0)
        num_sentences = len(images_list)
        for i in range(batch_size):
            s_tmp = '%s/super/%s/%s' % \
                    (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                os.makedirs(folder)
                print('Make a new folder: ', folder)
            #
            savename = '%s_%d.png' % (s_tmp, imsize)
            super_img = []
            for j in range(num_sentences):
                img = images_list[j][i]
                # print(img.size())
                img = img.view(1, 3, imsize, imsize)
                # print(img.size())
                super_img.append(img)
                # break
            super_img = torch.cat(super_img, 0)
            vutils.save_image(super_img, savename, nrow=10, normalize=True)

    def save_singleimages(self, images, filenames,
                          save_dir, split_dir, sentenceID, imsize):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' % \
                    (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                os.makedirs(folder)

            fullpath = '%s_%d_sentence%d.png' % (s_tmp, imsize, sentenceID)
            # range from [-1, 1] to [0, 255]
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)
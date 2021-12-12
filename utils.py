import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.reset()

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class PrototypicalBatchSampler(object):
    """
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.
    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    """

    def __init__(self, labels, data, classes_per_it, num_samples, iterations):
        """
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be inferred from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        """
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = {k: v for k, v in labels.items() if k in data}
        self.labels = {k: v - 1 for k, v in labels.items()}
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(list(self.labels.values()), return_counts=True)
        self.classes = torch.LongTensor(self.classes.astype('int32'))

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in num_el_per_class we store the number of samples for each class/row
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.num_el_per_class = torch.zeros_like(self.classes)

        for idx, label in self.labels.items():
            label_idx = np.argwhere(self.classes == int(label)).item()  # getting the index from classes list
            # using that index as row
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.num_el_per_class[label_idx] += 1

    def __iter__(self):
        """
        :return : yield a batch of indexes
        """

        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:self.classes_per_it]

            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                label_idx = c_idxs[i]
                sample_idxs = torch.randperm(self.num_el_per_class[label_idx])[:self.sample_per_class]
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        """
        returns the number of iterations (episodes) per epoch
        """
        return self.iterations


# Creating some helper functions
def get_num_model(config):
    num_model = config.num_model
    error_msg = "[!] model number must be >= 1."
    assert num_model > 0, error_msg
    return 'exp_' + str(num_model)


def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


def euclidean_dist(x, y):
    """
    Compute euclidean distance between two tensors
    """
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def get_imgs(img_path, imsize, branch_num, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    for i in range(branch_num):
        if i < branch_num:
            re_img = transforms.Resize(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))

    return ret

import torch

from .utils import expand_as_one_hot, flatten, binarize_probabilities, extract_largest_component

EPSILON = 1e-6

METRICS = ['IoU', 'Dice', 'BinaryDice', 'AdjustedIoU', 'AdjustedDice']


class PatchValidator:
    """
    Computes IoU over multiple patches.
    """

    def __init__(self, device, image_shape=None, num_classes=2, only_binary=False):
        if not only_binary:
            self.intersection = torch.zeros(num_classes, device=device, dtype=torch.float)
            self.size = torch.zeros(num_classes, device=device, dtype=torch.float)

        self.binary_intersection = torch.zeros(num_classes, device=device, dtype=torch.float)
        self.binary_size = torch.zeros(num_classes, device=device, dtype=torch.float)
        self.union = torch.zeros(num_classes, device=device, dtype=torch.float)

        if image_shape is not None:
            self.image = torch.zeros(image_shape, dtype=torch.int64)
            self.label = torch.zeros(image_shape, dtype=torch.int64)
        self.only_binary = only_binary
        self.classes = num_classes

    def update(self, output: torch.Tensor, label: torch.Tensor, patch_slice=None):
        """

        :param patch_slice:
        :param output: Output from the network after applying softmax to it. Expected shape 1 x C x Spatial
        :param label: Ground truth label. Expected shape 1 x Spatial
        """

        binary_output = binarize_probabilities(output)

        if patch_slice is not None:
            self.image[patch_slice] = binary_output.argmax(1).squeeze().cpu()
            self.label[patch_slice] = label.squeeze().cpu()

        label = expand_as_one_hot(label, C=output.size()[1]).byte()
        assert output.size() == label.size()
        output = flatten(output)
        binary_output = flatten(binary_output)
        label = flatten(label)

        if not self.only_binary:
            self.intersection += (output * label).sum(1).float()
            self.size += (output * output).sum(1) + (label * label).sum(1).float()

        self.binary_size += (binary_output.sum(1) + label.sum(1)).float()
        self.binary_intersection += (binary_output & label).sum(1).float()
        self.union += (binary_output | label).sum(1).float()

    def calculate_fnc(self, fnc):
        assert fnc in METRICS

        if fnc == 'IoU':
            return self.iou_score()
        if fnc == 'AdjustedIoU':
            return self.iou_after_extract()
        if fnc == 'Dice':
            return self.dice_score()
        if fnc == 'BinaryDice':
            return self.binary_dice_score()
        if fnc == 'AdjustedDice':
            return self.dice_after_extract()

    def iou_score(self):
        return ((self.binary_intersection + EPSILON) / (self.union + EPSILON)).mean()

    def iou_after_extract(self):
        assert self.image is not None and self.label is not None
        extracted = extract_largest_component(self.image).unsqueeze(0).long()  # Expand doesn't work with byte

        image = flatten(expand_as_one_hot(extracted, C=self.classes)).byte()
        label = flatten(expand_as_one_hot(self.label, C=self.classes)).byte()

        intersection = (image & label).sum(1).float()
        union = (image | label).sum(1).float()
        return ((intersection + EPSILON) / (union + EPSILON)).mean()

    def dice_after_extract(self):
        assert self.image is not None and self.label is not None
        extracted = extract_largest_component(self.image).unsqueeze(0).long()  # Expand doesn't work with byte

        image = flatten(expand_as_one_hot(extracted, C=self.classes)).byte()
        label = flatten(expand_as_one_hot(self.label, C=self.classes)).byte()

        intersection = (image & label).sum(1).float()
        size = (image.sum(1) + label.sum(1)).float()
        return ((2 * intersection + EPSILON) / (size + EPSILON)).mean()

    def binary_dice_score(self):
        return ((2 * self.binary_intersection + EPSILON) / (self.binary_size + EPSILON)).mean()

    def dice_score(self):
        assert not self.only_binary
        return ((2 * self.intersection + EPSILON) / (self.size + EPSILON)).mean()

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        pass

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def synch(self, device):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes, task, output_dir, task_id, no_bg_class):
        super().__init__()
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.total_samples = 0
        self.class_list = task.class_list()
        self.output_dir = output_dir
        self.index = 0
        if no_bg_class:
            self.start_bg = 0
        else:
            self.start_bg = 1

        # set-up seperate evaluations
        task_list = task.classes(task_id)
        # need number of elements
        self.base_classes = len(task_list[0]) # task 0
        self.novel_classes = sum(map(len, task_list[1:task_id+1])) # task 1 -> t, all incremental classes
        self.old_classes = sum(map(len, task_list[0:task_id])) if task_id > 0 else self.base_classes # old classes < t-1
        self.new_classes = len(task_list[task_id]) if task_id > 0 else self.base_classes # new classes in t

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
        self.total_samples += len(label_trues)

    def to_str(self, results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU" and k!="Class Acc" and k!="Confusion Matrix":
                string += "%s: %f\n"%(k, v)
        
        string+='Class IoU:\n'
        for k, v in results['Class IoU'].items():
            string += "\tclass %s: %s\n"%(self.class_list[k], str(v))

        string+='Class Acc:\n'
        for k, v in results['Class Acc'].items():
            string += "\tclass %s: %s\n"%(self.class_list[k], str(v))

        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        EPS = 1e-6
        hist = self.confusion_matrix

        gt_sum = hist.sum(axis=1)
        mask = (gt_sum != 0)
        diag = np.diag(hist)

        acc = diag.sum() / hist.sum()
        acc_cls_c = diag / (gt_sum + EPS)
        acc_cls = np.mean(acc_cls_c[mask])
        iu = diag / (gt_sum + hist.sum(axis=0) - diag + EPS)
        mean_iu = np.mean(iu[mask])
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), [iu[i] if m else "X" for i, m in enumerate(mask)]))
        cls_acc = dict(zip(range(self.n_classes), [acc_cls_c[i] if m else "X" for i, m in enumerate(mask)]))

        base_classes = np.full(self.n_classes, False)
        base_classes[self.start_bg:self.base_classes] = mask[self.start_bg:self.base_classes]
        miou_base = np.sum(iu[base_classes]) / (np.sum(base_classes))

        old_classes = np.full(self.n_classes, False)
        old_classes[self.start_bg:self.old_classes] = mask[self.start_bg:self.old_classes]
        miou_old = np.sum(iu[old_classes]) / (np.sum(old_classes)) if np.sum(old_classes) > 0 else 0

        new_classes = np.full(self.n_classes, False)
        new_classes[self.old_classes:] = mask[self.old_classes:]
        miou_new = (np.sum(iu[new_classes]) / (np.sum(new_classes))) if np.sum(new_classes) > 0 else 0

        novel_classes = np.full(self.n_classes, False)
        novel_classes[self.base_classes:] = mask[self.base_classes:]
        miou_novel = (np.sum(iu[novel_classes]) / (np.sum(novel_classes))) if np.sum(novel_classes) > 0 else 0

        return {
                "Total samples":  self.total_samples,
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Base IoU": miou_base,
                "Novel IoU": miou_novel,
                "Old IoU": miou_old,
                "New IoU": miou_new,
                "Class IoU": cls_iu,
                "Class Acc": cls_acc,
                # "Confusion Matrix": self.confusion_matrix_to_fig()
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.total_samples = 0

    def synch(self, device):
        # collect from multi-processes
        confusion_matrix = torch.tensor(self.confusion_matrix).to(device)
        samples = torch.tensor(self.total_samples).to(device)

        torch.distributed.reduce(confusion_matrix, dst=0)
        torch.distributed.reduce(samples, dst=0)

        if torch.distributed.get_rank() == 0:
            self.confusion_matrix = confusion_matrix.cpu().numpy()
            self.total_samples = samples.cpu().numpy()

    def confusion_matrix_to_fig(self):
        cm = self.confusion_matrix.astype('float') / (self.confusion_matrix.sum(axis=1)+0.000001)[:, np.newaxis]
        cm = torch.from_numpy(cm)
        cm = cm.repeat(3, 1, 1)
        cm[1] = 0.25
        cm[2] = 1 - cm[2]
        cm = cm.permute(1,2,0).numpy()
        fig, ax = plt.subplots()
        # ax.imshow(cm) # , interpolation='nearest'), #cmap=plt.cm.binary)

        ax.set(title=f'Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')

        fig.tight_layout()
        ## save confusion matrx
        # fig.savefig(f"{self.output_dir}/confusion_matrix_{self.index}.png")
        # self.index += 1
        return fig


class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]





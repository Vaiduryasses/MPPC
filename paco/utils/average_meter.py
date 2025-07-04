class AverageMeter(object):
    """
    Class for tracking and computing the average values of one or multiple metrics.

    This class supports updates with single numeric values or lists of numeric values.
    It provides methods to retrieve the latest value, count, and computed average.
    """
    def __init__(self, items=None):
        """
        Initialize the AverageMeter.

        Args:
            items: If provided, a list that determines the number of metrics to track.
                   If None, a single metric is tracked.
        """
        self.items = items
        self.n_items = 1 if items is None else len(items)
        self.reset()

    def reset(self):
        """
        Reset the tracked values, sums, and counts to zero.
        """
        self._val = [0] * self.n_items
        self._sum = [0] * self.n_items
        self._count = [0] * self.n_items

    def update(self, values):
        # 支持动态调整大小
        if not isinstance(values, list):
            values = [values]
        
        # 如果当前列表长度不足，扩展列表
        while len(self._val) < len(values):
            self._val.append(0.0)
            self._count.append(0)
            self._sum.append(0.0)
        
        # 如果当前列表长度过大，截断到vals的长度
        if len(self._val) > len(values):
            self._val = self._val[:len(values)]
            self._count = self._count[:len(values)]
            self._sum = self._sum[:len(values)]
        
        for idx, v in enumerate(values):
            self._val[idx] = v
            self._sum[idx] += v
            self._count[idx] += 1

    def val(self, idx=None):
        """
        Retrieve the latest value(s).

        Args:
            idx: Index of the metric. If None and a single metric is tracked,
                 returns that value. If tracking multiple metrics, returns a list
                 of the latest values.

        Returns:
            The latest value or list of latest values.
        """
        if idx is None:
            return self._val[0] if self.items is None else [self._val[i] for i in range(self.n_items)]
        else:
            return self._val[idx]

    def count(self, idx=None):
        """
        Retrieve the count of updates for the value(s).

        Args:
            idx: Index of the metric. If None and a single metric is tracked,
                 returns that count. If tracking multiple metrics, returns a list
                 of counts.

        Returns:
            The count or list of counts.
        """
        if idx is None:
            return self._count[0] if self.items is None else [self._count[i] for i in range(self.n_items)]
        else:
            return self._count[idx]

    def avg(self, idx=None):
        """
        Compute and return the average of the value(s).

        Args:
            idx: Index of the metric. If None and a single metric is tracked,
                 returns the average. If tracking multiple metrics, returns a list
                 of averages.

        Returns:
            The computed average value or a list of average values.
        """
        if idx is None:
            return self._sum[0] / self._count[0] if self.items is None else [
                self._sum[i] / self._count[i] for i in range(self.n_items)
            ]
        else:
            return self._sum[idx] / self._count[idx]
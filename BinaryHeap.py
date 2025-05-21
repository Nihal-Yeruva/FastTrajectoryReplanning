class BinaryHeap:
    def __init__(self):
        self.heap = []

    def isEmpty(self):
        return len(self.heap) == 0

    def heapifyUp(self, index):
        pIdx = (index - 1) // 2
        while pIdx >= 0 and self.heap[pIdx] > self.heap[index]:
            self.heap[pIdx], self.heap[index] = self.heap[index], self.heap[pIdx]
            index = pIdx
            pIdx = (index - 1) // 2

    def heapifyDown(self, index):
        lIdx = 2 * index + 1
        rIdx = 2 * index + 2
        curr = index

        if lIdx < len(self.heap) and self.heap[lIdx] < self.heap[curr]:
            curr = lIdx
        if rIdx < len(self.heap) and self.heap[rIdx] < self.heap[curr]:
            curr = rIdx
        if curr != index:
            self.heap[index], self.heap[curr] = self.heap[curr], self.heap[index]

            self.heapifyDown(curr)
    
    def insert(self, key):
        self.heap.append(key)
        self.heapifyUp(len(self.heap) - 1)
    
    def getMin(self):
        if self.isEmpty():
            return None
        return self.heap[0]
    
    def delMin(self):
        if self.isEmpty():
            return None
        min = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        self.heapifyDown(0)
        return min

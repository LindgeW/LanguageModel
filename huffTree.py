'''
    根据权重序列生成霍夫曼树
'''
import random


class BinTreeNode(object):
    def __init__(self):
        # self.data = data
        self.parent = None
        self.lchild = None
        self.rchild = None


class HuffTreeNode(BinTreeNode):
    def __init__(self, weight):
        super(HuffTreeNode, self).__init__()
        self.weight = weight
        self.code = ''

    def __str__(self):
        return str(self.weight)


# 二叉树的前序遍历(根 -> 左 -> 右)
def prev_order(root: HuffTreeNode):
    if root is not None:
        print(root.weight, root.code)
        prev_order(root.lchild)
        prev_order(root.rchild)


# 二叉树的中序遍历(左 -> 根 -> 右)
def mid_order(root: HuffTreeNode):
    if root is not None:
        prev_order(root.lchild)
        print(root.weight)
        prev_order(root.rchild)


# 二叉树的后序遍历(左 -> 右 -> 根)
def back_order(root: HuffTreeNode):
    if root is not None:
        prev_order(root.lchild)
        prev_order(root.rchild)
        print(root.weight)


# 构建huffman树
def build_huff_tree(huff_tree_nodes: list):
    def insert_node(queue: list, node: HuffTreeNode):
        for i, n in enumerate(queue):
            if node.weight > n.weight:
                queue.insert(i, node)
                return
        queue.insert(len(queue), node)

    # 按照权重降序排列
    node_queue = []
    for tnode in huff_tree_nodes:
        insert_node(node_queue, tnode)

    n = len(huff_tree_nodes) - 1
    for _ in range(n):  # 非叶子节点有n-1个
        # 最后两个为最小权重节点
        min1 = node_queue.pop()
        min2 = node_queue.pop()
        new_node = HuffTreeNode(min1.weight + min2.weight)
        new_node.lchild = min1
        new_node.rchild = min2
        min1.parent = new_node
        min2.parent = new_node

        # 新节点插入原队列中
        insert_node(node_queue, new_node)
    return node_queue[0]  # 根节点


# 生成从根节点到指定叶节点的huffman编码 (左0 右1)
def generate_huff_code(root: HuffTreeNode):
    if root is None:
        return
    if root.parent is not None:
        if root.parent.lchild == root:  # 当前节点是左孩子
            root.code = root.parent.code + '0'
        else:   # 当前节点是右孩子
            root.code = root.parent.code + '1'
    generate_huff_code(root.lchild)
    generate_huff_code(root.rchild)


if __name__ == '__main__':
    huff_nodes = [HuffTreeNode(random.randint(1, 10)) for i in range(5)]
    for node in huff_nodes:
        print(node)

    print("==" * 20)
    root = build_huff_tree(huff_nodes)
    generate_huff_code(root)
    prev_order(root)
    print("=="*20)
    print(root.code)
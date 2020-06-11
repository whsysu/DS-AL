二叉树的前序/中序/后续/层次遍历-python实现

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    #DFS遍历三种：用栈实现
    def __init__(self):
        self.data = []

    def preorder(self, root):
        if not root:
            return
        self.data.append(root.val)
        self.preorder(root.left)
        self.preorder(root.right)

    def preorder_iteration(self,root):
        if not root:
            return []
        stack,result=[root],[]
        while stack:
            node=stack.pop()
            result.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return result

    def postorder(self, root):
        if not root:
            return
        self.preorder(root.left)
        self.preorder(root.right)
        self.data.append(root.val)

    def postorder_iteration(self,root):
        if not root:
            return []
        stack,result=[root],[]
        while stack:
            node=stack.pop()
            result.append(node.val)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return result[::-1]

    def inorder(self, root):
        if not root:
            return
        self.inorder(root.left)
        self.data.append(root.val)
        self.inorder(root.right)

    def inorder_iteration(self, root):  # 二叉搜索树的中序遍历十分重要,结果为有序数组:记住二个while
        if not root:
            return []
        stack, result = [], []
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            result.append(root.val)
            root = root.right
        return result

    #BFS层次遍历：用队列实现
    def levelOrder(self, root):
        if not root:
            return []
        queue,result=[root],[]
        while queue:
            level=[]
            for i in range(len(queue)):
                node=queue.pop(0)
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(level)
        return result
```
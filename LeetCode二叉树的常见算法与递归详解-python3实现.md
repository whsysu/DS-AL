# LeetCode二叉树的常见算法与递归详解-python3实现

## 第一部分：二叉树的前序/中序/后续/层次遍历



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
## 第二部分：二叉树的常见递归算法



```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    #树的高度
    def maxDepth(self, root):            #递归第一要素：定义函数功能和返回值
        if not root:                     #递归第二要素：明确终止条件及当前传入要做到事
            return 0

        left=self.maxDepth(root.left)    #递归第三要素：写递归框架解决问题
        right=self.maxDepth(root.right)

        return max(left,right)+1

    #平衡树
    def isBalanced(self, root): #第一要素
        if not root:            #第二要素
            return True
        left=self.maxDepth(root.left)
        right=self.maxDepth(root.right)
        if abs(left-right)>1:
            return False
        return self.isBalanced(root.left) and self.isBalanced(root.right)  #第三要素

    #一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过也可能不穿过根结点。
    # 思路：只有三种情况：过根节点：左子树最大深度+又子树最大深度+根节点
    # 不过根节点：路径在左子树/右子树中：计算方式和上式一样
    def diameterOfBinaryTree(self, root): #有多个重复计算 #但是递归的本质要掌握
        if not root:
            return 0
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
        deep1 = left + right

        deep2 = self.diameterOfBinaryTree(root.left)
        deep3 = self.diameterOfBinaryTree(root.right)
        return max(deep1, deep2, deep3)

    def diameterOfBinaryTree(self, root):
        self.ans = 1

        def depth(node):
            # 访问到空节点了，返回0
            if not node: return 0
            # 左儿子为根的子树的深度
            L = depth(node.left)
            # 右儿子为根的子树的深度
            R = depth(node.right)
            # 计算d_node即L+R+1 并更新ans
            self.ans = max(self.ans, L + R + 1)
            # 返回该节点为根的子树的深度
            return max(L, R) + 1

        depth(root)
        return self.ans - 1

    # 翻转二叉树
    def invertTree(self, root):
        # 递归第一步：定义函数功能和返回值
        # 递归第二步：明确结束条件和当前输入要做的事
        if not root:
            return
        temp = root.left  # 当前节点要做的事  这里非常容易犯错
        root.left = self.invertTree(root.right)
        root.right = self.invertTree(temp)
        return root

    #合并二叉树：如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。
    def mergeTrees(self, t1, t2):# 第一步，明确函数功能和返回值
        # 第二步 定义结束条件和当前输入要做的事情
        if not t1 and not t2:
            return
        if not t1 and t2:
            return t2
        if t1 and not t2:
            return t1
        t1.val = t1.val + t2.val
        t1.left = self.mergeTrees(t1.left, t2.left)  #第三步 递归框架
        t1.right = self.mergeTrees(t1.right, t2.right)
        return t1

    #判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和
    def hasPathSum(self, root, sum):#第一步：明确函数功能和返回值
        if not root:
            return False
        sum=sum-root.val
        if not root.left and not root.right: #当前输入要做的事
            return sum==0
        left=self.hasPathSum(root.left,sum) #递归框架
        right=self.hasPathSum(root.right,sum)

        return left or right

    #找出路径和等于给定数值的路径总数。路径不需要从根节点开始，也不需要在叶子节点结束
    def path_with_start(self,start,sum):
        #以start为起点，判断和为sum的路径总数
        if not start:
            return 0
        res=0
        sum=sum-start.val
        if sum==0:
            res+=1
        res+=self.path_with_start(start.left,sum)
        res+=self.path_with_start(start.right,sum)
        return res
    def pathSum(self, root, sum) :
        #函数功能：判断总路径树
        if not root:
            return 0
        res=self.path_with_start(root,sum)
        res+=self.pathSum(root.left,sum)
        res+=self.pathSum(root.right,sum)
        return res


    #572. 另一个树的子树，s 的一个子树包括 s 的一个节点和这个节点的所有子孙。
    def is_equal(self,t1,t2):#判断二个树是否相等
        if not t1 and not t2:
            return True
        if not t1 or not t2:
            return False
        if t1.val!=t2.val:
            return False
        left=self.is_equal(t1.left,t2.left)
        right=self.is_equal(t1.right,t2.right)
        return left and right
    def isSubtree(self, s, t):
        #递归第一步：明确函数功能和返回值
        if not s:
            return False
        if self.is_equal(s,t):
            return True
        return self.isSubtree(s.left,t) or self.isSubtree(s.right,t)

    #给定一个二叉树，检查它是否是镜像对称的。
    def ismirror(self, root1, root2):
        if not root1 and not root2:
            return True
        if not root1 or not root2:
            return False
        if root1.val != root2.val:
            return False
        return self.ismirror(root1.left, root2.right) and self.ismirror(root1.right, root2.left)
    def isSymmetric(self, root):
        if not root:
            return True
        return self.ismirror(root.left, root.right)

    #最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
    def minDepth(self, root):
        if not root:
            return 0
        left = self.minDepth(root.left)
        right = self.minDepth(root.right)
        if min(left, right) > 0:
            return min(left, right) + 1
        else:
            return left + 1 if right == 0 else right + 1

    #404计算给定二叉树的所有左叶子之和。
    def sumOfLeftLeaves(self, root):
        #前序遍历二叉树并且判断是否是左叶子节点
        if not root:
            return 0
        res=0
        if root.left and not root.left.left and not root.left.right:
            res+=root.left.val
        res+=self.sumOfLeftLeaves(root.left)
        res+=self.sumOfLeftLeaves(root.right)
        return res

    #687给定一个二叉树，找到最长的路径，这个路径中的每个节点具有相同值。 这条路径可以经过也可以不经过根节点。
    #没懂

    #如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。
    def rob(self, root): #思路没问题，但是超时，树型动态规划问题=》有重复子问题
        # 第一步：定义函数功能和返回值：返回以输入节点为根节点的树可偷取的最大值
        # 则包含二种情况：偷根节点和不偷根节点
        # 第二步：定义结束条件和当前输入要做的事情
        if not root:
            return 0
        res1 = root.val
        if root.left:
            res1 += self.rob(root.left.left) + self.rob(root.left.right)
        if root.right:
            res1 += self.rob(root.right.left) + self.rob(root.right.right)

        res2 = self.rob(root.left) + self.rob(root.right)
        return max(res1, res2)

    #树型动态规划问题
    #这里最关键的是理解：在设计状态的时候，在后面加一维，消除后效性；树的问题，很多时候采用后序遍历。
    def dfs(self,root):
        #dp问题+递归
        #dp[0]以root为根节点的树可投的最大值，但不偷根节点
        #dp[1]以root为根节点的树可偷的最大值，但偷根节点
        res=[0,0]
        if not root:
            return res
        #采用后续遍历
        left=self.dfs(root.left)
        right=self.dfs(root.right)
        res[0]=max(left[0],left[1])+max(right[0],right[1])
        res[1]=root.val+left[0]+right[0]
        return res
    def rob(self, root):
        return max(self.dfs(root))

    #671. 二叉树中第二小的节点：每个节点都是正数且子节点数量只能为 2 或 0。有两个子节点的话，那么这个节点的值不大于它的子节点的值。 
    def findSecondMinimumValue(self, root):
        # 遍历规则
        if not root or not root.left:
            return -1
        left_val = root.left.val
        right_val = root.right.val
        if left_val == root.val:
            left_val = self.findSecondMinimumValue(root.left)
        if right_val == root.val:
            right_val = self.findSecondMinimumValue(root.right)
        if left_val != -1 and right_val != -1:
            return min(left_val, right_val)
        if left_val != -1:
            return left_val
        return right_val    
    




```

## 第三部分：二叉搜索树相关操作

```python

#669，通过修剪二叉搜索树，使得所有节点的值在[L, R]中，可能要改变树的根节点，返回修剪好的二叉搜索树的新的根节点。
def trimBST(self, root, L, R):
    if not root:
        return None
    if root.val > R:
        return self.trimBST(root.left, L, R)
    elif root.val < L:
        return self.trimBST(root.right, L, R)
    else:
        root.left = self.trimBST(root.left, L, R)
        root.right = self.trimBST(root.right, L, R)
    return root


#230给定一个二叉搜索树，编写一个函数 kthSmallest 来查找其中第 k 个最小的元素。1 ≤ k ≤ 二叉搜索树元素个数。
#思路一：二叉搜索树的中序遍历是递增序列  又到了背公式的时间了
def inorder_iteration(self,root,k):
    if not root:
        return []
    stack,result=[],[]
    while root or stack:
        while root:
            stack.append(root)
            root=root.left
        root=stack.pop()
        result.append(root.val)
        if len(result)==k:
            return result[-1]
        root=root.right

#538：BST把它转换成为累加树（Greater Tree)，使得每个节点的值是原来的节点值加上所有大于它的节点值之和。
def __init__(self):
    self.res = 0
def convertBST(self, root):
    # 思路：遍历思路应该为右=>根=>左==>即颠倒的中序遍历
    if not root:
        return
    self.convertBST(root.right)
    root.val = root.val + self.res
    self.res = root.val
    self.convertBST(root.left)
    return root

#235. 二叉搜索树的最近公共祖先
def lowestCommonAncestor(self, root, p, q):
    # 递归第一步：明确函数功能和返回值
    if p.val > q.val:
        p, q = q, p
    if not root:
        return
    if root.val < p.val:
        return self.lowestCommonAncestor(root.right, p, q)
    if root.val > q.val:
        return self.lowestCommonAncestor(root.left, p, q)
    return root

#236. 二叉树的最近公共祖先
def lowestCommonAncestor(self, root, p, q):
    # 第一步;定义函数功能和返回值：找到以root为根的树的公共子节点并返回
    if not root or root == p or root == q:  # 第二步结束条件和输入要做的事
        return root
    left = self.lowestCommonAncestor(root.left, p, q)  # 第三步：遍历框架
    right = self.lowestCommonAncestor(root.right, p, q)
    if not left:
        return right
    if not right:
        return left
    return root

#108. 将一个按照升序排列的有序数组，转换为一棵高度平衡二叉搜索树。
def sortedArrayToBST(self, nums):
    '''思路：bst的中序遍历是有序数组：流程如下：
    数组中间的数是根节点(保证平衡)，左边为左子树，右边为右子树'''
    # 第一步：定义函数功能和返回值
    # 定义结束条件和当前输入要做的事情
    if not nums:
        return None
    if len(nums) == 1:
        return TreeNode(nums[0])
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = self.sortedArrayToBST(nums[0:mid])
    root.right = self.sortedArrayToBST(nums[mid + 1:])
    return root

#109. 有序链表转换高度平衡的二叉搜索树
def sortedListToBST(self, head):
    '''思路1：链表转为数组，再转为平衡bst
        思路2：直接转，先得找到链表中点(快慢指针))'''

#653. 给定二叉搜索树和一个目标结果，如果 BST 中存在两个元素且它们的和等于给定的目标结果，则返回 true。
#思路一：先中序遍历二叉树，得到集合，再从集合中双指针搜索
    def __init__(self):
        self.data=[]
    def inorder_interation(self,root):
        if not root:
            return
        stack=[]
        while root or stack:
            while root:
                stack.append(root)
                root=root.left
            root=stack.pop()
            self.data.append(root.val)
            root=root.right
    def find(self,nums,k):
        i=0
        j=len(nums)-1
        while i<j:
            if nums[i]+nums[j]==k:
                return True
            elif nums[i]+nums[j]<k:
                i+=1
            elif nums[i]+nums[j]>k:
                j-=1
        return False
    def findTarget(self, root, k):
        self.inorder_interation(root)
        return self.find(self.data,k)
```

## 第四部分：Trie: 前缀树/字典树

```python
#208.用于检索字符串数据集中的键。这一高效的数据结构有多种应用：
#https://leetcode-cn.com/problems/implement-trie-prefix-tree/solution/shi-xian-trie-qian-zhui-shu-by-leetcode/
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.dic={}
    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        a=self.dic
        for i in word:
            if i not in a:
                a[i]={}
            a=a[i]
        a['end']=True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        a=self.dic
        for i in word:
            if i not in a:
                return False
            a=a[i]
        if 'end' in a:
            return True
        return False

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        a=self.dic
        for i in prefix:
            if i not in a:
                return False
            a=a[i]
        return True

```


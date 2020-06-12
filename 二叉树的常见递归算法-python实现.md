二叉树的常见递归算法-python实现

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

    #671. 二叉树中第二小的节点：每个节点都是正数且子节点数量只能为 2 或 0。有两个子节点的话，那么这个节点的值不大于它的子节点的值。 
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
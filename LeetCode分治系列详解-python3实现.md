#### 分治算法

```python
#241. 为运算表达式设计优先级：给定一个含有数字和运算符的字符串，为表达式添加括号，
# 改变其运算优先级以求出不同的结果。你需要给出所有可能的组合的结果。有效的运算符号包含 +, - 以及 * 。
def diffWaysToCompute(self, input: str):
    '''思路：显然这道题可以递归的求解，递归有2种：分治和动规，
    这里分解后子问题不重复，故为分治'''
    # 分治3步：
    # 分解：按运算符分成左右两部分，分别求解
    # 解决：实现一个递归函数，输入算式，返回算式解
    # 合并：根据运算符合并左右两部分的解，得出最终解
    if input.isdigit():
        return [int(input)]
    res = []
    for i, char in enumerate(input):
        if char in ['+', '-', '*']:
            left = self.diffWaysToCompute(input[0:i])
            right = self.diffWaysToCompute(input[i + 1:])
            for l in left:
                for r in right:
                    if char == '+':
                        res.append(l + r)
                    elif char == '-':
                        res.append(l - r)
                    else:
                        res.append(r * l)
    return res

#95.不同的二叉搜索树，给定一个整数 n，生成所有由 1 ... n 为节点所组成的 二叉搜索树
def generateTrees(self, n: int):
    '''思路：遍历i=1~n,以i作为根节点，则[1~i-1]和[i+1~n]分别作为左右子树
    ，再递归的构建左右子树
    G(n)表示长为n的序列可构成的不同BST数量
    F(i,n)表示以i为根长为n的序列可构成的不同bst数量
    则：G(n)=sum(F(i,n)), base case:G(0)=0  G(1)=1
    而 F(i,n)=G(i-1)*G(n-i)=> G(n)=sum(G(i-1)*G(n-i))
    至此：此题从毫无头绪变成了递推
    '''
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        for j in range(1, i + 1):
            dp[i] += dp[j - 1] * dp[i - j]
    print(dp[n])

```


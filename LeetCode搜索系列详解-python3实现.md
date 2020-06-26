```python
#LeetCode 搜索算法详解
#深度优先搜索和广度优先搜索广泛运用于树和图中，但是它们的应用远远不止如此。
#广度优先搜索一层一层遍历，每一层得到的所有新节点，要用队列存储起来以备下一层遍历的时候再遍历。

#深度优先搜索在得到一个新节点时立即对新节点进行遍历：从节点 0 出发开始遍历，得到到新节点 6 时，
#立马对新节点 6 进行遍历，得到新节点 4；如此反复以这种方式遍历新节点，直到没有新节点了，此时返回。
#返回到根节点 0 的情况是，继续对根节点 0 进行遍历，得到新节点 2，然后继续以上步骤。
#从一个节点出发，使用 DFS 对一个图进行遍历时，能够遍历到的节点都是从初始节点可达的，DFS 常用来求解这种 可达性 问题。
#在程序实现 DFS 时需要考虑以下问题：
#栈：用栈来保存当前节点信息，当遍历新节点返回时能够继续遍历当前节点。可以使用递归栈。
#标记：和 BFS 一样同样需要对已经遍历过的节点进行标记。

#BFS
#队列：用来存储每一轮遍历得到的节点；
#标记：对于遍历过的节点，应该将它标记，防止重复遍历。
#对于先遍历的节点 i 与后遍历的节点 j，有 di <= dj。利用这个结论，可以求解最短路径等 最优解 问题：
#第一次遍历到目的节点，其所经过的路径为最短路径。应该注意的是，使用 BFS 只能求解无权图的最短路径

from math import *
#1091. 二进制矩阵中的最短路径:0表示可以通过，1表示不可以通过，求从左上角到右下角的最短路径，8个方向都可以走
def shortestPathBinaryMatrix(self, grid) -> int:
    '''思路：首先这是图，每个点代表节点，点和点之间局部相连，和手机图案的9宫格相似。求最短的路径=>找路径常用BFS/DFS/BT搜索+找最值常用动态规划'''
    # 对于BFS，一旦搜索到即最短路径
    n = len(grid)
    # 判断边界条件
    if not grid or grid[0][0] == 1 or grid[n - 1][n - 1] == 1:
        return -1

    queue = [(0, 0, 1)]  # 定义队列，并加入起始位置
    grid[0][0] = 1  # 定义备忘录，更新访问过的位置
    direct = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while queue:
        i, j, step = queue.pop(0)
        for dx, dy in direct:
            new_x, new_y = i + dx, j + dy
            if new_x == n - 1 and new_y == n - 1:
                return step + 1
            elif 0 <= new_x < n - 1 and 0 <= new_y < n - 1 and grid[new_x][new_y] == 0:
                grid[new_x][new_y] = 1
                queue.append((new_x, new_y, step + 1))
    return -1

#279. 完全平方数：找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。
'''思路1：解空间构成N叉树，可用BFS搜索'''
def numSquares(self, n: int) -> int:
    coins = [i * i for i in range(1, int(sqrt(n)) + 1)]
    queue = [n]
    level = 0
    while queue:
        level += 1
        next_queue = set()  # list 会有大量重复
        for i in queue:
            for coin in coins:
                if i == coin:
                    return level
                elif i < coin:
                    break
                else:
                    next_queue.add(i - coin)
        queue = next_queue
    return -1

# 这一题和找零钱的题大致类似=》动态规划
def numSquares(self, n: int) -> int:
    dp = [i for i in range(0, n + 1)]  # dp[i]表示和为i的最少个平方数 初始化为最多的情况
    coins = [i * i for i in range(1, int(sqrt(n)) + 1)]

    for i in range(1, n + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i - coin] + 1, dp[i])
            else:
                break
    return dp[-1]

#127. 单词接龙
#找出一条从 beginWord 到 endWord 的最短路径，每次移动规定为改变一个字符，并且改变之后的字符串必须在 wordList 中。
#将单词之间的转换，理解为一张图，即如果两个单词之间可以转换(相差一个字母)，就说明这两个单词之间存在一条边。因此，问题就变成了从无向无权图中起始单词到终止单词的最短路径。
def ladderLength(self, beginWord: str, endWord: str, wordList):
    '''思考：解空间构成树状结构=>根节点就是beginword，下一层节点是只和beginword差
    一个字母的单词，不断搜索下去，直到某个level含有endword为止
    再思考：将单词之间的转换，理解为一张图，即如果两个单词之间可以转换(相差一个字母)，
    就说明这两个单词之间存在一条边。因此，问题就变成了从无向无权图中起始单词到终止单词的最短路径。
    因此，如果有了这个图，则直接采用BFS即可得到解。故该问题的难点就在于如何构建该图？可以根据字典构建一个邻接表'''

#DFS
#695. 岛屿的最大面积:给定一个包含了一些0,1的非空二维数组.岛屿是由一些相邻的1构成的组合，相邻要求两个1必须在水平或者
# 竖直方向上相邻。可以假设grid四个边缘都被 0包围着。找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为0)
def dfs(self, grid, i, j):
    if i < 0 or j < 0 or i == len(grid) or j == len(grid[0]) or grid[i][j] != 1:
        return 0
    grid[i][j] = 0  # mark一下
    ans = 1
    direct = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    for dx, dy in direct:
        new_x, new_y = i + dx, j + dy
        ans += self.dfs(grid, new_x, new_y)
    return ans
def maxAreaOfIsland(self, grid) -> int:
    '''思路：这是二个二维数组，以及联通问题，首先想到图，再求最大面积，又想到
    动态规划：
    遍历图上的每一个点，作为起点可以构成的岛屿面积，再求最大的面积
    以某个点为起点，就变成了图的连通性求法=>深度或宽度搜索'''
    ans = 0
    for i, l in enumerate(grid):
        for j, n in enumerate(l):
            ans = max(ans, self.dfs(grid, i, j))
    return ans

#dfs+栈
def maxAreaOfIsland(self, grid) -> int:
    ans = 0
    for i, l in enumerate(grid):
        for j, n in enumerate(l):
            cur = 0
            stack = [(i, j)]
            while stack:
                cur_i, cur_j = stack.pop()
                if cur_i < 0 or cur_j < 0 or cur_i == len(grid) or cur_j == len(grid[0]) or grid[cur_i][cur_j] != 1:
                    continue
                cur += 1
                grid[cur_i][cur_j] = 0
                for di, dj in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                    next_i, next_j = cur_i + di, cur_j + dj
                    stack.append((next_i, next_j))
            ans = max(ans, cur)
    return ans
#BFS
import collections
def maxAreaOfIsland(self, grid) -> int:
    ans = 0
    for i, l in enumerate(grid):
        for j, n in enumerate(l):
            cur = 0
            q = collections.deque([(i, j)])
            while q:
                cur_i, cur_j = q.popleft()
                if cur_i < 0 or cur_j < 0 or cur_i == len(grid) or cur_j == len(grid[0]) or grid[cur_i][cur_j] != 1:
                    continue
                cur += 1
                grid[cur_i][cur_j] = 0
                for di, dj in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                    next_i, next_j = cur_i + di, cur_j + dj
                    q.append((next_i, next_j))
            ans = max(ans, cur)
    return ans

#200. 岛屿数量:岛屿总是被水包围，并且每座岛屿只能由水平方向或竖直方向上相邻的陆地连接形成。
#此外，你可以假设该网格的四条边均被水包围。
def dfs(self, grid, i, j):
    grid[i][j] = '0'
    row, col = len(grid), len(grid[0])
    direct = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for dx, dy in direct:
        new_x, new_y = i + dx, j + dy
        if 0 <= new_x < row and 0 <= new_y < col and grid[new_x][new_y] == '1':
            self.dfs(grid, new_x, new_y)
def numIslands(self, grid) -> int:
    '''二维网格的连通性，首先想到图的搜索=>可以将矩阵表示看成一张有向图。
    从某一点开始搜索岛屿并标记记录总数即可'''
    row = len(grid)
    if row == 0:
        return 0
    col = len(grid[0])
    ans = 0
    for i in range(row):
        for j in range(col):
            if grid[i][j] == '1':
                ans += 1
                self.dfs(grid, i, j)
    return ans

#BFS
def numIslands(self, grid) -> int:
    nr = len(grid)
    if nr == 0:
        return 0
    nc = len(grid[0])

    num_islands = 0
    for r in range(nr):
        for c in range(nc):
            if grid[r][c] == "1":
                num_islands += 1
                grid[r][c] = "0"
                neighbors = collections.deque([(r, c)])
                while neighbors:
                    row, col = neighbors.popleft()
                    for x, y in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
                        if 0 <= x < nr and 0 <= y < nc and grid[x][y] == "1":
                            neighbors.append((x, y))
                            grid[x][y] = "0"

    return num_islands


#547. 朋友圈：友谊有传递性。如果已知 A 是 B 的朋友，B 是 C 的朋友，可以认为 A 也是 C 的朋友。朋友圈，是指所有朋友的集合。
def dfs(self, M, visit, i):
    for j in range(len(M)):
        if M[i][j] == 1 and visit[j] == 0:
            visit[j] = 1
            self.dfs(M, visit, j)
def findCircleNum(self, M) -> int:
    '''思考：首先这个矩阵是对称的。步骤：
    1.根据朋友关系的传递性更新矩阵，找到第一个为1的其他朋友=>规则是与当前位置同一行/列
    2.这个矩阵本质为图的邻接矩阵，则找到图的连通分量即可'''
    visit = [0] * len(M)
    count = 0
    for i in range(len(M)):
        if visit[i] == 0:
            self.dfs(M, visit, i)
            count += 1
    return count

def bfs(self, M, visit, i):
    queue = [i]
    while queue:
        s = queue.pop(0)
        visit[s] = 1
        for j in range(len(M)):
            if M[s][j] == 1 and visit[j] == 0:
                queue.append(j)
def findCircleNum(self, M) -> int:
    visit = [0] * len(M)
    count = 0
    for i in range(len(M)):
        if visit[i] == 0:
            self.bfs(M, visit, i)
            count += 1

    return count

#130. 被围绕的区域：二维矩阵，包含 'X' 和 'O'，找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
def dfs(self, grid, i, j):
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == 'X' or grid[i][j] == '#':
        return
    grid[i][j] = '#'
    dircet = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    for dx, dy in dircet:
        x, y = i + dx, j + dy
        self.dfs(grid, x, y)
def solve(self, board) -> None:
    """
    Do not return anything, modify board in-place instead.
    """
    '''思路： 找到连通区域:从非边界的O出发，如果找到的所有连通位置不在边界上，则满足连通要求=>所以是搜索问题=>DFS/BFS=>问题转化为，位于边界上连通的o找到，替换为#，完了再替换一次'''
    if not board:
        return
    row, col = len(board), len(board[0])
    for i in range(row):
        for j in range(col):
            if (i == 0 or j == 0 or i == row - 1 or j == col - 1) and board[i][j] == 'O':
                self.dfs(board, i, j)
    for i in range(row):
        for j in range(col):
            if board[i][j] == '#':
                board[i][j] = 'O'
            elif board[i][j] == 'O':
                board[i][j] = 'X'

#417. 太平洋大西洋水流问题：
#给定一个 m x n 的非负整数矩阵来表示一片大陆上各个单元格的高度。“太平洋”处于大陆的左边界和上边界，而“大西洋”处于大陆的右边界和下边界。
#规定水流只能按照上、下、左、右四个方向流动，且只能从高到低或者在同等高度上流动。
#请找出那些水流既可以流动到“太平洋”，又能流动到“大西洋”的陆地单元的坐标。



#Backtracking（回溯）属于 DFS。
#普通 DFS 主要用在 可达性问题 ，这种问题只需要执行到特点的位置然后返回即可。
#而 Backtracking 主要用于求解 排列组合 问题，例如有 { 'a','b','c' } 三个字符，求解所有由这三个字符排列得到的字符串，
# 这种问题在执行到特定的位置返回之后还会继续执行求解过程。
#因为 Backtracking 不是立即返回，而要继续求解，因此在程序实现时，需要注意对元素的标记问题：
#在访问一个新元素进入新的递归调用时，需要将新元素标记为已经访问，这样才能在继续递归调用时不用重复访问该元素；
#但是在递归返回时，需要将元素标记为未访问，因为只需要保证在一个递归链中不同时访问一个元素，可以访问已经访问过但是不在当前递归链中的元素。

#17. 电话号码的字母组合:给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。
def __init__(self):
    self.phone = {'2': ['a', 'b', 'c'],
                  '3': ['d', 'e', 'f'],
                  '4': ['g', 'h', 'i'],
                  '5': ['j', 'k', 'l'],
                  '6': ['m', 'n', 'o'],
                  '7': ['p', 'q', 'r', 's'],
                  '8': ['t', 'u', 'v'],
                  '9': ['w', 'x', 'y', 'z']}
def backtrack(self, res, length, select, path):
    if len(path) == length:
        res.append(''.join(path[:]))
        return
    for dig in self.phone[select[0]]:
        path.append(dig)
        self.backtrack(res, length, select[1:], path)
        path.pop(-1)
def letterCombinations(self, digits: str) -> List[str]:
    '''解空间构成一颗N叉树，所有组合就是N叉树的所有路径=>
    遍历这棵N叉树即可'''
    if not digits:
        return []
    res = []
    length = len(digits)
    self.backtrack(res, length, digits, [])
    return res


#93. 复原IP地址:给定一个只包含数字的字符串，复原它并返回所有可能的 IP 地址格式。
#有效的 IP 地址正好由四个整数（每个整数位于 0 到 255 之间组成），整数之间用 '.' 分隔。


#79. 单词搜索:给定一个二维网格和一个单词，找出该单词是否存在于网格中。单词必须按照字母顺序，
# 通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。
# 同一个单元格内的字母不允许被重复使用。
#哪里有点问题，且只用了dfs，没有用回溯
def dfs(self, board, row, col, word, visit):
    if len(word) == 0:
        return True
    direct = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for dx, dy in direct:
        x, y = row + dx, col + dy
        if 0 <= x < len(board) and 0 <= y < len(board[0]) and visit[x][y] == 0 and board[x][y] == word[0]:
            visit[x][y] = 1
            if self.dfs(board, x, y, word[1:], visit):
                return True

    return False
def exist(self, board: List[List[str]], word: str) -> bool:
    if not word:
        return True
    row, col = len(board), len(board[0])
    for i in range(row):
        for j in range(col):
            if board[i][j] == word[0]:
                visit = [[0 for _ in range(col)] for _ in range(row)]
                visit[i][j] = 1
                if self.dfs(board, i, j, word[1:], visit):
                    return True
    return False



#257. 二叉树的所有路径
def dfs(self,res,root,path): #回溯算法
    if root:
        path.append(str(root.val))
        if not root.left and not root.right:
            res.append('->'.join(path))
        else:
            self.dfs(res,root.left,path)
            self.dfs(res,root.right,path)
        path.pop(-1) #如果path是字符串则不用pop()字符串是不可变对象
def binaryTreePaths(self, root: TreeNode) -> List[str]:
    res=[]
    self.dfs(res,root,[])
    return res




#46. 全排列:给定一个 没有重复 数字的序列，返回其所有可能的全排列。“全排列”就是一个非常经典的“回溯”算法的应用。
def __init__(self):
    self.res = []
def bt(self, select, track):  # 选择列表 ，路径
    if len(select) == len(track):  # 结束条件
        self.res.append(track[:])
        return
    for i in select:  # 选择列表
        if i in track:  # 判断条件
            continue
        track.append(i)  # 做选择
        self.bt(select, track)
        track.pop(-1)  # 撤销选择
def permute(self, nums):
    track = []
    self.bt(nums, track)
    return self.res

def backtrack(self, select, result, path, visit):
    if len(path) == len(select):
        result.append(path[:])
        return
    for i in range(len(select)):
        if visit[i]:
            continue
        visit[i] = True
        path.append(select[i])
        self.backtrack(select, result, path, visit)
        path.pop(-1)
        visit[i] = False
def permute(self, nums: List[int]) -> List[List[int]]:
    result = []
    path = []
    visit = [0] * len(nums)
    self.backtrack(nums, result, path, visit)
    return result

#47. 全排列 给定一个可包含重复数字的序列，返回所有不重复的全排列。
def backtrack(self, select, result, path, visit):
    if len(path) == len(select):
        result.append(path[:])
        return
    for i in range(len(select)):
        if i != 0 and select[i] == select[i - 1] and not visit[i - 1]:
            continue  # 剪枝操作
        if visit[i]:
            continue
        visit[i] = True
        path.append(select[i])
        self.backtrack(select, result, path, visit)
        path.pop(-1)
        visit[i] = False
def permuteUnique(self, nums: List[int]) -> List[List[int]]:
    nums.sort()
    result = []
    path = []
    visit = [0] * len(nums)
    self.backtrack(nums, result, path, visit)
    return result

#77. 组合：给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。
def backtrack(self, result, path, n, k, start):
    # 到达底部
    if len(path) == k:
        result.append(path[:])
        return
    for i in range(start, n + 1):
        path.append(i)
        self.backtrack(result, path, n, k, i + 1)
        path.pop(-1)
def combine(self, n: int, k: int) -> List[List[int]]:
    '''解空间为树型结构，首先想到搜索'''
    result = []
    if k <= 0 or n <= 0:
        return result
    self.backtrack(result, [], n, k, 1)
    return result

#39. 组合总和：无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
def backtrack(self, result, select, start, target, path):
    if target == 0:
        result.append(path[:])  # list为可变对象，为引用传递，需要把值拷贝出来
        return
    for i in range(start, len(select)):  # 使用start是为了避免重复
        if target - select[i] >= 0:
            path.append(select[i])
            self.backtrack(result, select, i, target - select[i], path)
            path.pop(-1)
def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    '''首先联想到一道类似的题，一个排序数组中找二个数和为target
    这题元素可重复选取，则本质和找零钱一致了，但有一点不同
    本题的解空间依然可以构成树模型，当叶子节点为0时表示这条路径刚好可以'''
    result = []
    if not candidates:
        return result
    # candidates.sort() 无需排序
    self.backtrack(result, candidates, 0, target, [])
    return result

#40. 组合总和：给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。candidates 中的每个数字在每个组合中只能使用一次。(candidates中可以重复)
def backtrack(self, result, select, start, path, target, visit):
    if target == 0:
        result.append(path[:])
        return
    for i in range(start, len(select)):
        if i != 0 and select[i] == select[i - 1] and not visit[i - 1]:
            continue
        if target - select[i] >= 0:
            path.append(select[i])
            visit[i] = 1
            self.backtrack(result, select, i + 1, path, target - select[i], visit)
            visit[i] = 0
            path.pop(-1)
def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
    '''思路：解空间依然构成树，但是candidate可重复，则需要剪枝 '''
    result = []
    candidates.sort()
    visit = [0] * len(candidates)
    self.backtrack(result, candidates, 0, [], target, visit)
    return result

#216. 组合总和：找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。
def backtrack(self, result, path, start, k, target):
    if target == 0 and len(path) == k:
        result.append(path[:])
        return
    for i in range(start, 10):
        if target - i >= 0 and len(path) < k:
            path.append(i)
            self.backtrack(result, path, i + 1, k, target - i)
            path.pop(-1)
def combinationSum3(self, k: int, n: int) -> List[List[int]]:
    '''首先；解空间依然是树结构，且不重复'''
    result = []
    if k <= 0 or n <= 0 or k >= n:
        return result
    self.backtrack(result, [], 1, k, n)
    return result


#78. 子集：给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
'''回溯模板 labuladong
        result = []
        def backtrack(路径, 选择列表):
            if 满足结束条件:
                result.add(路径)
                return
            for 选择 in 选择列表:
                做选择
                backtrack(路径, 选择列表)
                撤销选择
    '''
def backtrack(self,res,select,start,path):
    res.append(path[:])
    for i in range(start,len(select)):
        path.append(select[i])
        self.backtrack(res,select,i+1,path)
        path.pop(-1)
def subsets(self, nums: List[int]) -> List[List[int]]:
    '''思路：组合问题且大问题转为小问题用分治
    迭代和循环二种方式'''
    res = []
    self.backtrack(res,nums,0,[])
    return res

#90. 子集：给定一个可能包含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
def backtrack(self, result, path, start, select, visit):
    result.append(path[:])
    for i in range(start, len(select)):
        if i != 0 and select[i] == select[i - 1] and visit[i - 1] == 0:
            continue
        path.append(select[i])
        visit[i] = 1
        self.backtrack(result, path, i + 1, select, visit)
        visit[i] = 0
        path.pop(-1)
def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
    '''思路：'''
    nums.sort()
    res = []
    visit = [0] * len(nums)
    self.backtrack(res, [], 0, nums, visit)
    return res


#131. 分割回文串：将 s 分割成一些子串，使每个子串都是回文串。返回 s 所有可能的分割方案。


#37. 解数独：编写一个程序，通过已填充的空格来解决数独问题。
'''一个数独的解法需遵循如下规则：
数字 1-9 在每一行只能出现一次。
数字 1-9 在每一列只能出现一次。
数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
空白格用 '.' 表示。'''


#15. N 皇后:在 n*n 的矩阵中摆放 n 个皇后，并且每个皇后不能在同一行，同一列，同一对角线上，求所有的 n 皇后的解。

```

